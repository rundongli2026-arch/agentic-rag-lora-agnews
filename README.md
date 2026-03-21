# Stitching Project: Advanced Agentic RAG with LoRA (AG News)

This project implements the stitching requirements:
- At least 2 LLM-based agents
- One agent uses a LoRA fine-tuned model
- Vector DB retriever (FAISS)
- CLI front-end
- 4-way evaluation:
  - Base LLM (no RAG)
  - Basic RAG
  - Advanced agentic RAG (base model)
  - Advanced agentic RAG (LoRA-enabled agent)

## Project Files
- `stitching_system.py`: shared system, LangGraph workflow, retriever, and mode runners
- `train_lora.py`: trains LoRA relevance-judge adapter on AG News
- `evaluate.py`: runs 4-system comparison on 5-10 prompts and exports report files
- `app.py`: CLI front-end
- `run_full_pipeline.py`: one-command run (train + evaluate)
- `requirements.txt`: dependencies

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key in environment:
   ```powershell
   $env:OPENAI_API_KEY="your_openai_api_key"
   ```

## Full Run Flow
### Option A: One-command pipeline
```bash
python run_full_pipeline.py
```

### Option B: Step-by-step
1. Train LoRA adapter:
   ```bash
   python train_lora.py --output-dir artifacts/lora_agnews_relevance_adapter --train-pairs 3000 --val-pairs 400 --max-steps 120
   ```

2. Run stitching evaluation (creates report artifacts):
   ```bash
   python evaluate.py --artifacts-dir artifacts --output-dir outputs --sample-size 3000
   ```

3. Run CLI front-end:
   ```bash
   python app.py --artifacts-dir artifacts --mode advanced_lora
   ```

## Model Choice
- Fast CPU demo default: `sshleifer/tiny-gpt2` (used by `run_full_pipeline.py`)
- Higher quality (recommended with GPU): `HuggingFaceTB/SmolLM2-360M-Instruct`
  - Example:
    ```bash
    python train_lora.py --base-model-id HuggingFaceTB/SmolLM2-360M-Instruct --output-dir artifacts/lora_agnews_relevance_adapter --train-pairs 3000 --val-pairs 400 --max-steps 120
    ```

## Outputs
- `outputs/comparison_outputs.csv`
- `outputs/sample_outputs.md`
- `outputs/agent_graph.mmd`

## Submission Notes
- Do not upload API keys.
- Do not upload datasets.
- Include generated report files plus source code and run instructions.
