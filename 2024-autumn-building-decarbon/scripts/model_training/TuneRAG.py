"""This script loads and fine tunes a RAG model given fine-tuning data.

This script provides functionality for interacting with a Chroma
database, including:
- Loading and saving the model.
- Uses a simple query1 = answer 1 for fine-tuning.
- Provides a CLI interface with various controls for adjusting params.
- CLI also provides the option to bypass fine-tuning and load the model as is + save.

However, more work is needed on re-calling the loaded model to properly use.
This includes defining a class architecture to reload the model from the
fine-tuned weights.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn.functional as nn
from huggingface_hub import HfFolder
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizerBase

API_KEY = os.getenv("HUGGING_FACE_API_KEY")
HfFolder.save_token(API_KEY)


def main() -> None:
    """Parses CLI arguments, optionally fine-tunes, and saves the model."""
    parser = argparse.ArgumentParser(
        description="Load and optionally fine-tune a SentenceTransformer model."
    )
    parser.add_argument(
        "--fine_tune", action="store_true", help="Set to fine-tune the model."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer (only relevant if fine-tuning).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Margin for contrastive loss (only relevant if fine-tuning).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (only relevant if fine-tuning).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training (only relevant if fine-tuning).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="fine_tuned_model.pth",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--query_file",
        type=str,
        help="Path to the queries file (required if fine-tuning).",
    )
    parser.add_argument(
        "--passage_file",
        type=str,
        help="Path to the passages file (required if fine-tuning).",
    )

    args = parser.parse_args()

    model = load_model()

    if args.fine_tune:
        if not args.query_file or not args.passage_file:
            parser.error(
                """--query_file and --passage_file are required 
                when --fine_tune is set."""
            )

        print("Starting fine-tuning...")
        print("Configuration:")
        print(f"  Learning Rate: {args.learning_rate}")
        print(f"  Margin: {args.margin}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Save Path: {args.save_path}")

        queries, passages = load_queries_and_passages(
            args.query_file, args.passage_file
        )

        model, loss_fn, optimizer, device = configure_fine_tuning(
            args.margin, args.learning_rate
        )

        tokenizer = model.tokenizer
        train_loader = load_data(model, queries, passages, args.batch_size, tokenizer)

        fine_tuned_model = trained_model(
            model, optimizer, loss_fn, device, train_loader, args.epochs
        )
        print("Fine-tuning completed.")
    else:
        print("Fine-tuning skipped.")
        fine_tuned_model = model

    save_model(fine_tuned_model, args.save_path)
    print(f"Model loaded and saved at: {args.save_path}")


def load_queries_and_passages(
    query_file: str, passage_file: str
) -> tuple[list[Any], list[Any]]:
    """Loads queries and passages from the specified JSON files.

    Args:
        query_file (str): Path to the file containing queries.
        passage_file (str): Path to the file containing passages.

    Returns:
        (list, list): Tuple containing a list of queries and list of passages.
    """
    with Path(query_file).open() as qf, Path(passage_file).open() as pf:
        queries = json.load(qf)
        passages = json.load(pf)
    return queries, passages


def load_model() -> SentenceTransformer:
    """Initializes NVIDIA-Embed-v2 from HuggingFace.

    Returns:
        SentenceTransformer: The loaded SentenceTransformer model.
    """
    model = SentenceTransformer("nvidia/NV-Embed-v2", trust_remote_code=True)
    model.max_seq_length = 32768
    model.tokenizer.padding_side = "right"
    print("Model loaded.")
    return model


def add_eos(
        input_texts: list[str], tokenizer: PreTrainedTokenizerBase
    ) -> list[str]:
    """Appends the EOS token to each text in the provided list of input texts.

    Args:
        input_texts (list of str): The input texts.
        tokenizer: The tokenizer with an eos_token attribute.

    Returns:
        list of str: The modified texts with the EOS token appended.
    """
    return [text + tokenizer.eos_token for text in input_texts]


def configure_fine_tuning(
    margin_contrastive_loss: float, learning_rate: float
) -> tuple[SentenceTransformer, nn.Module, AdamW, str]:
    """Configures the model for fine-tuning by loading and setting it to train.

    Initializes the contrastive loss, and sets up the optimizer and device.

    Args:
        margin_contrastive_loss (float): The margin for the contrastive loss.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        (SentenceTransformer, nn.Module, AdamW, str): The fine-tuning model,
        the contrastive loss function, the optimizer, and the device used.
    """
    model = load_model()
    model.train()
    contrastive_loss_fn = nn.CosineEmbeddingLoss(margin=margin_contrastive_loss)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, contrastive_loss_fn, optimizer, device


def load_data(
    model: SentenceTransformer,
    queries: list[str],
    passages: list[str],
    batches: int,
    tokenizer: PreTrainedTokenizerBase,
) -> DataLoader:
    """Encodes queries and passages using the model and prepares a DataLoader.

    Args:
        model (SentenceTransformer): The model used for encoding.
        queries (list of str): The queries to be encoded.
        passages (list of str): The passages to be encoded.
        batches (int): The batch size for the DataLoader.
        tokenizer: The tokenizer with an eos_token attribute.

    Returns:
        DataLoader: DataLoader for training with encoded queries and passages.
    """
    task_name_to_instruct = {
        "example": """
        Given a question, retrieve passages that answer the questio
        """
    }
    query_prefix = "Instruct" + task_name_to_instruct["example"] + "\nQuery: "

    query_embeddings = model.encode(
        add_eos(queries, tokenizer),
        batch_size=2,
        prompt=query_prefix,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    passage_embeddings = model.encode(
        add_eos(passages, tokenizer),
        batch_size=2,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )

    train_dataset = TensorDataset(query_embeddings, passage_embeddings)
    train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=False)

    return train_loader


def trained_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
    train_loader: DataLoader,
    number_of_epochs: int,
) -> torch.nn.Module:
    """Trains the model for a specified number of epochs.
    
    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating
        the model parameters.
        loss_fn (nn.Module): The loss function to be used.
        device (str): The device ("cuda" or "cpu") on which training will occur.
        train_loader (DataLoader): A DataLoader providing training batches.
        number_of_epochs (int): The number of epochs to train the model.

    Returns:
        nn.Module: The trained model.
    """
    for epoch in range(number_of_epochs):
        for _, (query_batch, passage_batch, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            query_batch = query_batch.to(device).requires_grad_(True)
            passage_batch = passage_batch.to(device).requires_grad_(True)

            query_batch = F.normalize(query_batch, p=2, dim=1)
            passage_batch = F.normalize(passage_batch, p=2, dim=1)

            labels = labels.to(device)

            loss = loss_fn(query_batch, passage_batch, labels)

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")
    print("Fine-tuning completed.")
    return model


def save_model(model: torch.nn.Module, save_path: str) -> None:
    """Saves the model's state to the specified file path.

    Args:
        model (nn.Module): The model to be saved.
        save_path (str): The path where the model's state dict should be saved.
    """
    dir_name = Path(save_path).parent
    dir_name.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at: {save_path}")


if __name__ == "__main__":
    main()
