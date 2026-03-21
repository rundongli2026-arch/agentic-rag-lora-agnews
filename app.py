import argparse
from pathlib import Path

from stitching_system import build_system


def parse_args():
    parser = argparse.ArgumentParser(description="CLI front-end for stitching project.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument(
        "--mode",
        type=str,
        default="advanced_lora",
        choices=["base_llm", "basic_rag", "advanced_base", "advanced_lora"],
    )
    parser.add_argument("--question", type=str, default="")
    return parser.parse_args()


def run_once(system, mode: str, question: str):
    result = system.run_mode(question, mode)
    print(f"\nMode: {mode}")
    print(f"Question: {question}")
    if result.get("rewritten_query"):
        print(f"\nRewritten query: {result['rewritten_query']}")
    if result.get("route_decision"):
        print(f"Route decision: {result['route_decision']}")
        print(f"Route reasoning: {result.get('route_reasoning', '')}")
    if isinstance(result.get("relevant_docs"), list):
        print(f"Relevant docs used: {len(result['relevant_docs'])}")
    print(f"\nAnswer:\n{result.get('answer', '')}\n")


def main():
    args = parse_args()
    system = build_system(
        artifacts_dir=args.artifacts_dir,
        openai_model=args.openai_model,
        embedding_model=args.embedding_model,
        sample_size=args.sample_size,
    )

    if args.question.strip():
        run_once(system, args.mode, args.question.strip())
        return

    print(
        "Stitching Project CLI ready. Enter a question, or type 'exit' to stop."
    )
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        run_once(system, args.mode, q)


if __name__ == "__main__":
    main()
