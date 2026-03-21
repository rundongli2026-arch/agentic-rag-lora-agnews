import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, TypedDict

import pandas as pd
import torch
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


class RewriteQueryOutput(BaseModel):
    rewritten_query: str = Field(
        description="A retrieval-optimized query that preserves intent."
    )


class RelevanceJudgmentOutput(BaseModel):
    relevant_doc_ids: List[int] = Field(
        description="Indices of retrieved docs relevant to the question."
    )
    reasoning: str = Field(description="Why those docs are relevant.")


class RouteDecisionOutput(BaseModel):
    next_step: Literal["retrieve_more", "answer", "abstain"] = Field(
        description="Choose whether to retrieve more, answer now, or abstain."
    )
    reasoning: str = Field(description="Short explanation for the route decision.")


class FinalAnswerOutput(BaseModel):
    answer: str = Field(description="Grounded answer from provided context.")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level based on evidence sufficiency."
    )


class AgentState(TypedDict):
    question: str
    rewritten_query: str
    retrieved_docs: List[Document]
    relevant_docs: List[Document]
    relevance_reasoning: str
    route_decision: str
    route_reasoning: str
    answer: str
    retrieval_round: int
    mode: str


@dataclass
class LoraRelevanceJudge:
    model: object
    tokenizer: object
    device: str

    def judge(self, question: str, doc_text: str) -> bool:
        prompt = (
            "You are a document relevance judge for RAG.\n"
            "Given a question and one document, output exactly one token:\n"
            "relevant or not_relevant.\n\n"
            f"Question: {question}\n\n"
            f"Document: {doc_text}\n\n"
            "Answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
        if "not_relevant" in text or "not relevant" in text:
            return False
        return "relevant" in text


def ensure_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError(
            "OPENAI_API_KEY is not set. Set it in your shell environment before running."
        )


def build_documents(sample_size: int = 3000, random_state: int = 42) -> List[Document]:
    ds = load_dataset("ag_news")
    train_df = pd.DataFrame(ds["train"])
    train_df["label_name"] = train_df["label"].map(LABEL_MAP)
    sample = train_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    sample["combined_text"] = (
        "Category: " + sample["label_name"] + "\n" + "News Article: " + sample["text"]
    )
    docs: List[Document] = []
    for i, row in sample.iterrows():
        docs.append(
            Document(
                page_content=row["combined_text"],
                metadata={
                    "doc_id": int(i),
                    "label": int(row["label"]),
                    "label_name": row["label_name"],
                },
            )
        )
    return docs


def get_or_build_vectorstore(
    index_dir: Path,
    sample_size: int = 3000,
    embedding_model: str = "text-embedding-3-small",
) -> FAISS:
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    index_file = index_dir / "index.faiss"
    if index_file.exists():
        return FAISS.load_local(
            folder_path=str(index_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    docs = build_documents(sample_size=sample_size)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def format_docs_for_prompt(docs: List[Document]) -> str:
    if not docs:
        return "No documents available."
    blocks = []
    for i, doc in enumerate(docs):
        blocks.append(f"[DOC {i}]\nMetadata: {doc.metadata}\nContent:\n{doc.page_content}")
    return "\n\n".join(blocks)


def dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []
    for doc in docs:
        key = (doc.metadata.get("doc_id"), doc.page_content[:150])
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs


def load_lora_judge(base_model_id: str, adapter_dir: Path) -> LoraRelevanceJudge:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter not found at: {adapter_dir}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return LoraRelevanceJudge(model=model, tokenizer=tokenizer, device=device)


class StitchingSystem:
    def __init__(
        self,
        vectorstore: FAISS,
        openai_model: str = "gpt-4o-mini",
        lora_judge: LoraRelevanceJudge | None = None,
        retrieval_k: int = 8,
        max_retrieval_rounds: int = 2,
    ):
        self.vectorstore = vectorstore
        self.retrieval_k = retrieval_k
        self.max_retrieval_rounds = max_retrieval_rounds
        self.lora_judge = lora_judge

        self.llm = ChatOpenAI(model=openai_model, temperature=0)
        self.rewrite_llm = self.llm.with_structured_output(RewriteQueryOutput)
        self.judge_llm = self.llm.with_structured_output(RelevanceJudgmentOutput)
        self.route_llm = self.llm.with_structured_output(RouteDecisionOutput)
        self.answer_llm = self.llm.with_structured_output(FinalAnswerOutput)

        self.agent_graph = self._build_graph()

    def _build_graph(self):
        def rewrite_node(state: AgentState) -> Dict[str, str]:
            result = self.rewrite_llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You rewrite user questions into concise semantic retrieval queries. "
                            "Do not answer."
                        )
                    ),
                    HumanMessage(content=f"Question:\n{state['question']}"),
                ]
            )
            return {"rewritten_query": result.rewritten_query}

        def retrieve_node(state: AgentState) -> Dict[str, object]:
            k = self.retrieval_k + state["retrieval_round"] * 4
            docs = self.vectorstore.similarity_search(state["rewritten_query"], k=k)
            merged = dedupe_docs(state["retrieved_docs"] + docs)
            return {"retrieved_docs": merged, "retrieval_round": state["retrieval_round"] + 1}

        def grade_node(state: AgentState) -> Dict[str, object]:
            if state["mode"] == "advanced_lora":
                if self.lora_judge is None:
                    raise ValueError(
                        "advanced_lora mode requested but LoRA judge is not loaded."
                    )
                selected = []
                for doc in state["retrieved_docs"]:
                    if self.lora_judge.judge(state["question"], doc.page_content):
                        selected.append(doc)
                reason = (
                    "LoRA relevance judge selected documents based on learned topical match."
                    if selected
                    else "LoRA relevance judge found no sufficiently relevant documents."
                )
                return {"relevant_docs": selected, "relevance_reasoning": reason}

            docs_text = format_docs_for_prompt(state["retrieved_docs"])
            result = self.judge_llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a relevance grader. Select only docs that directly help answer."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Question:\n{state['question']}\n\n"
                            f"Rewritten query:\n{state['rewritten_query']}\n\n"
                            f"Retrieved docs:\n{docs_text}\n\n"
                            "Return relevant doc ids and short reasoning."
                        )
                    ),
                ]
            )
            valid_ids = []
            for idx in result.relevant_doc_ids:
                if 0 <= idx < len(state["retrieved_docs"]) and idx not in valid_ids:
                    valid_ids.append(idx)
            relevant_docs = [state["retrieved_docs"][idx] for idx in valid_ids]
            return {"relevant_docs": relevant_docs, "relevance_reasoning": result.reasoning}

        def route_node(state: AgentState) -> Dict[str, str]:
            docs_text = format_docs_for_prompt(state["relevant_docs"])
            result = self.route_llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a route planner for agentic RAG. "
                            "Choose next_step from retrieve_more, answer, abstain."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Question:\n{state['question']}\n\n"
                            f"Retrieval rounds used: {state['retrieval_round']} "
                            f"(limit: {self.max_retrieval_rounds})\n\n"
                            f"Relevance reasoning:\n{state['relevance_reasoning']}\n\n"
                            f"Relevant docs:\n{docs_text}"
                        )
                    ),
                ]
            )
            return {"route_decision": result.next_step, "route_reasoning": result.reasoning}

        def answer_node(state: AgentState) -> Dict[str, str]:
            docs_text = format_docs_for_prompt(state["relevant_docs"])
            result = self.answer_llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You answer user questions strictly from provided documents. "
                            "If evidence is weak, say what is uncertain."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Question:\n{state['question']}\n\n"
                            f"Context:\n{docs_text}\n\n"
                            "Provide a grounded, concise answer."
                        )
                    ),
                ]
            )
            prefix = "[Low confidence] " if result.confidence == "low" else ""
            return {"answer": prefix + result.answer}

        def abstain_node(state: AgentState) -> Dict[str, str]:
            return {
                "answer": (
                    "I do not have enough reliable evidence in the retrieved archive to answer."
                )
            }

        def selector(state: AgentState) -> str:
            if state["retrieval_round"] >= self.max_retrieval_rounds:
                if state["route_decision"] == "retrieve_more":
                    return "answer"
            if state["route_decision"] == "retrieve_more":
                return "retrieve"
            if state["route_decision"] == "abstain":
                return "abstain"
            return "answer"

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("rewrite", rewrite_node)
        graph_builder.add_node("retrieve", retrieve_node)
        graph_builder.add_node("grade", grade_node)
        graph_builder.add_node("route", route_node)
        graph_builder.add_node("answer", answer_node)
        graph_builder.add_node("abstain", abstain_node)

        graph_builder.set_entry_point("rewrite")
        graph_builder.add_edge("rewrite", "retrieve")
        graph_builder.add_edge("retrieve", "grade")
        graph_builder.add_edge("grade", "route")
        graph_builder.add_conditional_edges(
            "route",
            selector,
            {"retrieve": "retrieve", "answer": "answer", "abstain": "abstain"},
        )
        graph_builder.add_edge("answer", END)
        graph_builder.add_edge("abstain", END)
        return graph_builder.compile()

    def base_llm(self, question: str) -> Dict[str, str]:
        msg = [
            SystemMessage(content="You are a concise helpful assistant."),
            HumanMessage(content=question),
        ]
        answer = self.llm.invoke(msg).content
        return {"answer": answer}

    def basic_rag(self, question: str, k: int = 6) -> Dict[str, object]:
        docs = self.vectorstore.similarity_search(question, k=k)
        docs_text = format_docs_for_prompt(docs)
        result = self.answer_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a RAG assistant. Answer only from provided context and note uncertainty."
                    )
                ),
                HumanMessage(
                    content=f"Question:\n{question}\n\nRetrieved docs:\n{docs_text}\n\nAnswer:"
                ),
            ]
        )
        prefix = "[Low confidence] " if result.confidence == "low" else ""
        return {"answer": prefix + result.answer, "retrieved_docs": docs}

    def advanced_agentic(self, question: str, mode: Literal["advanced_base", "advanced_lora"]):
        init_state: AgentState = {
            "question": question,
            "rewritten_query": "",
            "retrieved_docs": [],
            "relevant_docs": [],
            "relevance_reasoning": "",
            "route_decision": "",
            "route_reasoning": "",
            "answer": "",
            "retrieval_round": 0,
            "mode": mode,
        }
        return self.agent_graph.invoke(init_state, config={"recursion_limit": 12})

    def run_mode(self, question: str, mode: str) -> Dict[str, object]:
        if mode == "base_llm":
            return self.base_llm(question)
        if mode == "basic_rag":
            return self.basic_rag(question)
        if mode in {"advanced_base", "advanced_lora"}:
            return self.advanced_agentic(question, mode=mode)  # type: ignore[arg-type]
        raise ValueError(f"Unsupported mode: {mode}")

    def save_mermaid(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.agent_graph.get_graph().draw_mermaid(), encoding="utf-8")


def build_system(
    artifacts_dir: Path,
    openai_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    sample_size: int = 3000,
    retrieval_k: int = 8,
    max_retrieval_rounds: int = 2,
) -> StitchingSystem:
    ensure_openai_key()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = get_or_build_vectorstore(
        index_dir=artifacts_dir / "faiss_index",
        sample_size=sample_size,
        embedding_model=embedding_model,
    )

    lora_dir = artifacts_dir / "lora_agnews_relevance_adapter"
    lora_judge = None
    metadata_path = lora_dir / "metadata.json"
    if lora_dir.exists() and metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        base_model_id = meta.get("base_model_id", "HuggingFaceTB/SmolLM2-360M-Instruct")
        lora_judge = load_lora_judge(base_model_id=base_model_id, adapter_dir=lora_dir)

    return StitchingSystem(
        vectorstore=vectorstore,
        openai_model=openai_model,
        lora_judge=lora_judge,
        retrieval_k=retrieval_k,
        max_retrieval_rounds=max_retrieval_rounds,
    )
