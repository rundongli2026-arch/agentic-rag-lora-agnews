import argparse
from pathlib import Path
from typing import List

import pandas as pd

from stitching_system import build_system


DEFAULT_QUESTIONS = [
    "What are recent major themes in technology and AI news?",
    "Summarize key business and market developments.",
    "What kinds of sports stories appear in the archive?",
    "Summarize the tone of world affairs reporting.",
    "What science and innovation topics are discussed?",
    "What company strategy trends appear across business and tech?",
    "What geopolitical risks are highlighted in world news?",
    "What athlete performance narratives are most common?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate stitching project systems.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--questions-file", type=Path, default=None)
    return parser.parse_args()


def load_questions(path: Path | None) -> List[str]:
    if path is None:
        return DEFAULT_QUESTIONS
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    return [x for x in lines if x]


def build_discussion(df: pd.DataFrame) -> str:
    lines = [
        "## Discussion",
        "",
        "- Base LLM often provides fluent but generic answers because it has no archive grounding.",
        "- Basic RAG improves factual grounding but still includes occasional noisy context.",
        "- Advanced agentic RAG (base) improves answer focus by separating rewrite, grading, and routing.",
        "- Advanced agentic RAG (LoRA) uses a fine-tuned relevance agent; this tends to improve topical filtering for domain-specific archive questions.",
        "",
        "### Per-Question Highlights",
    ]
    for qid in sorted(df["question_id"].unique()):
        sub = df[df["question_id"] == qid].copy()
        q = sub["question"].iloc[0]
        lines.append(f"- Q{qid}: {q}")
        for mode in ["base_llm", "basic_rag", "advanced_base", "advanced_lora"]:
            row = sub[sub["mode"] == mode].iloc[0]
            ans = str(row["answer"]).replace("\n", " ")
            if len(ans) > 220:
                ans = ans[:217] + "..."
            lines.append(f"  - {mode}: {ans}")
    return "\n".join(lines)


def _truncate(text: str, limit: int = 260) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main():
    args = parse_args()
    questions = load_questions(args.questions_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    system = build_system(
        artifacts_dir=args.artifacts_dir,
        openai_model=args.openai_model,
        embedding_model=args.embedding_model,
        sample_size=args.sample_size,
    )
    system.save_mermaid(args.output_dir / "agent_graph.mmd")

    rows = []
    modes = ["base_llm", "basic_rag", "advanced_base", "advanced_lora"]
    for qid, question in enumerate(questions, start=1):
        print(f"\n=== Q{qid}: {question}")
        for mode in modes:
            result = system.run_mode(question, mode)
            answer = result.get("answer", "")
            print(f"[{mode}] {str(answer)[:180]}")
            rows.append(
                {
                    "question_id": qid,
                    "question": question,
                    "mode": mode,
                    "answer": answer,
                    "rewritten_query": result.get("rewritten_query", ""),
                    "relevance_reasoning": result.get("relevance_reasoning", ""),
                    "route_decision": result.get("route_decision", ""),
                    "route_reasoning": result.get("route_reasoning", ""),
                    "retrieval_round": result.get("retrieval_round", ""),
                    "num_retrieved_docs": len(result.get("retrieved_docs", []))
                    if isinstance(result.get("retrieved_docs", []), list)
                    else "",
                    "num_relevant_docs": len(result.get("relevant_docs", []))
                    if isinstance(result.get("relevant_docs", []), list)
                    else "",
                }
            )

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "comparison_outputs.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV: {csv_path}")

    md_df = df[
        [
            "question_id",
            "mode",
            "retrieval_round",
            "num_retrieved_docs",
            "num_relevant_docs",
            "route_decision",
            "answer",
        ]
    ].copy()
    md_df["answer"] = md_df["answer"].map(lambda x: _truncate(x, 260))

    md_lines = [
        "# Stitching Project Sample Outputs",
        "",
        "## Setup",
        "- Dataset: AG News",
        "- Retriever: FAISS vector database",
        "- Modes evaluated: base_llm, basic_rag, advanced_base, advanced_lora",
        "",
        "## Agentic Workflow",
        "- rewrite -> retrieve -> grade -> route -> (retrieve | answer | abstain)",
        "- In advanced_lora mode, `grade` is powered by LoRA fine-tuned model.",
        "",
        "## Outputs Table",
        "",
        md_df.to_markdown(index=False),
        "",
        build_discussion(df),
    ]
    md_path = args.output_dir / "sample_outputs.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved sample output report: {md_path}")


if __name__ == "__main__":
    main()
