"""Script for querying a Generating the response.

This script allows users to:
- Initalizes the LLM.
- Creates a fn that processes the query and generates the response.
"""

from typing import Any

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def setup_model_and_processor(
    model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
) -> tuple[Any, Any, torch.device]:
    """Initializes and returns the model and processor.

    Parametsers: model_name: The llama model being used
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=
        torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor, device


def generate_response(
        query: str, model: AutoModelForVision2Seq, 
        processor: AutoProcessor, 
        device: str
) -> str:
    """Function processes the user's query using LLaMA processor.

    It generates a response using the LLM.
    The response generation hyper parameters are optimized for coherent
    and contextually appropriate outputs.
    Parameters:
        query (str): The user's query
        model: Llama-3.2-11B-Vision-Instruct
        processor: LLaMA-associated processor
        device: CPU or GPU

    Returns:
        str: The generated response.
    """
    full_prompt = query.strip()

    inputs = processor(
        text=[full_prompt], return_tensors="pt", padding=True, truncation=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=300,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        temperature=0.8,
        top_k=10,
        top_p=0.9,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    decoded_output = processor.batch_decode(
        output_ids, skip_special_tokens=True
    )
    response_text = decoded_output[0].strip()

    sources = []
    if "Sources:" in response_text:
        response, sources_section = response_text.split("Sources:", 1)
        response_text = response.strip()
        sources = [
            source.strip() for source 
            in sources_section.splitlines() 
            if source.strip()
        ]
    else:
        response = response_text

    return {"response": response_text, "sources": sources}


def main() -> None:
    """This function initializes the model, processor, and device.

    It then enters an interactive session where users can input prompts.
    The model processes each prompt and generates a response.
    Users can exit the session by typing 'exit'.
    """
    # Load LLaMA model and processor
    print("Loading model and processor...")
    model, processor, device = setup_model_and_processor()
    print("Model loaded successfully!")

    # Start interactive loop
    print("Enter a prompt (or 'exit' to quit):")
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == "exit":
            break

        # Generate and display the response
        print("\nProcessing query...\n")
        response = generate_response(prompt, model, processor, device)

        print("Response:", response["response"])
        if response["sources"]:
            print("\nSources:")
            for source in response["sources"]:
                print(f"- {source}")
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
