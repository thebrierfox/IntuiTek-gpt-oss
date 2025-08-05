#!/usr/bin/env python3
"""
example_inference.py - Quick example for loading and using OpenAI's gpt oss models.

This script demonstrates how to load a gpt oss model using Hugging Face
Transformers and run a simple text generation query. It accepts a
model identifier and optional parameters to customise the prompt,
maximum tokens and reasoning effort.

Usage:
    python example_inference.py [model_id] [--prompt PROMPT]
                               [--max_tokens N] [--reasoning_level LEVEL]

Arguments:
    model_id         Identifier of the model on Hugging Face (default: openai/gpt oss 20b)

Options:
    --prompt         Prompt to ask the model (default: example prompt)
    --max_tokens     Maximum number of new tokens to generate (default: 128)
    --reasoning_level
                     Reasoning effort level: low, medium or high (default: medium)

This script constructs a conversation template compatible with gpt oss models,
including a system message specifying the desired reasoning level. It then
encodes the conversation, generates a response and prints the output.

Note: Running gpt oss models locally may require significant GPU memory.
See the repository README for hardware requirements and model download instructions.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_prompt(user_prompt: str, reasoning_level: str) -> str:
    """Build a chat prompt including a reasoning level instruction."""
    return (
        f"<|system|># Reasoning level: {reasoning_level}\n"  # instructs the model how much reasoning to perform
        "You are an AI assistant. Provide chain of thought reasoning before your answer.\n"
        f"<|user|>{user_prompt}\n"
        "<|assistant|>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a gpt oss model")
    parser.add_argument(
        "model_id",
        nargs="?",
        default="openai/gpt-oss-20b",
        help="Hugging Face model ID to load (e.g., openai/gpt-oss-20b or openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the benefits of the mixture of experts architecture.",
        help="Prompt/question to send to the model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--reasoning_level",
        choices=["low", "medium", "high"],
        default="medium",
        help="Desired chain of thought reasoning level",
    )
    args = parser.parse_args()

    # Build chat prompt with reasoning level instruction
    chat_prompt = build_prompt(args.prompt, args.reasoning_level)

    # Load model and tokenizer.  Device mapping uses GPU if available.
    print(f"Loading model {args.model_id}…")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")

    # Encode prompt
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    # Generate a response; tune sampling parameters as desired.
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Print the full conversation including chain of thought reasoning and answer
    print("\n--- Model Output ---\n")
    print(output)


if __name__ == "__main__":
    main()
