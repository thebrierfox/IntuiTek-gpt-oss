# IntuiTek-gpt-oss

This repository provides integration guides, scripts and examples for using OpenAI's **gpt-oss** models (`gpt-oss-20b` and `gpt-oss-120b`). These are open weight reasoning models released by OpenAI on 5 Aug 2025. They feature a mixture‑of‑experts architecture, adjustable chain‑of‑thought reasoning, tool‑use capabilities and can run locally on consumer hardware.

## Models

| Model | Parameter counts | Active parameters | Minimum hardware (approx.) | Context length |
| ----- | ---------------- | ---------------- | -------------------------- | -------------- |
| **gpt‑oss‑20b** | 21 billion total parameters | ≈ 3.6 B active | ≥ 16 GB VRAM or unified memory (consumer GPUs, high‑end laptops) | 128 K tokens |
| **gpt‑oss‑120b** | 117 billion total parameters | ≈ 5.1 B active | single high‑end GPU (≥ 80 GB VRAM) | 128 K tokens |

Both models are released under the Apache‑2.0 license and are available for download on [Hugging Face](https://huggingface.co/openai) or via cloud providers like AWS Bedrock/SageMaker. The active parameters are much smaller thanks to the **Mixture‑of‑Experts** architecture, which allows them to run on smaller hardware footprints.

## Features

- **Chain‑of‑thought reasoning** – configurable levels of reasoning effort (low/medium/high) with full chain‑of‑thought output for debugging and transparency.
- **Tool use** – built‑in ability to browse the web, execute Python code, call functions and return structured outputs via the OpenAI Tools API. 
- **Large context window** – handle up to 128 K tokens (~131,072 tokens) enabling long conversations or documents.
- **Fine‑tunable** – the open weights allow you to fine‑tune the models for your specific use‑cases.
- **Agentic capabilities** – function calling, web browsing and Python execution support enable building complex AI agents.

## Getting Started

### Requirements

- Python 3.10+ recommended.
- [`transformers`](https://pypi.org/project/transformers/) >= 4.41.0.
- [`accelerate`](https://pypi.org/project/accelerate/) for efficient hardware use.
- A GPU or Apple Silicon CPU with sufficient memory (see table above).

Install dependencies:

```bash
pip install --upgrade transformers accelerate
```

You can also install `vllm` or `ollama` for faster inference on specific hardware; see the [openai/gpt-oss](https://github.com/openai/gpt-oss) repository for details.

### Downloading Models

Using the Hugging Face CLI:

```bash
# Login to Hugging Face if needed
huggingface-cli login
# Download the 20B model
huggingface-cli download openai/gpt-oss-20b --local-dir ./models/gpt-oss-20b
# Download the 120B model (requires ≥80 GB VRAM)
huggingface-cli download openai/gpt-oss-120b --local-dir ./models/gpt-oss-120b
```

Alternatively you can let the `transformers` library download the model on first use.

### Quick Example

`example_inference.py` demonstrates a simple chat using the 20B model via the `transformers` pipeline API. Run:

```bash
python example_inference.py
```

The script prints a concise answer to a sample question along with the model’s chain‑of‑thought. To use the 120B model, pass its model ID as a command‑line argument:

```bash
python example_inference.py openai/gpt-oss-120b
```

## Repository Structure

- `example_inference.py` – simple script demonstrating how to load a gpt‑oss model via `transformers` and generate an answer.
- `README.md` – this document.
- *(You can add additional notebooks or scripts for benchmarking, fine‑tuning or RAG workflows.)*

## Resources

- [gpt-oss GitHub repository](https://github.com/openai/gpt-oss) – official reference implementations and usage guides.
- [AWS announcement](https://www.aboutamazon.com/news/aws/openai-models-amazon-bedrock-sagemaker) – integration with AWS Bedrock and SageMaker.
- [NVIDIA blog](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss/) – running the models on Nvidia RTX GPUs.

Feel free to open issues or pull requests with suggestions or improvements.
