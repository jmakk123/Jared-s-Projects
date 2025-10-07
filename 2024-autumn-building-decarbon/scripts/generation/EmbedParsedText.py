"""Initializes the embedding function for RAG queries.

This script allows users to:
- Create the instance of the embedding function that is used for
  populating the database in PopulateDatabase.py.
- Embed queries to do retrieval in RetrieveTextFromQuery.py.

Note: This implementation uses a Nvidia-Embed-v2; focusing solely 
on embedding. 
"""

import os

import torch
from huggingface_hub import HfFolder, hf_hub_download
from sentence_transformers import SentenceTransformer

API_KEY = os.getenv("HUGGING_FACE_API_KEY")
HfFolder.save_token(API_KEY)


class CustomSentenceTransformer:
    """A wrapper class for SentenceTransformer to customize embedding functionality."""

    def __init__(
        self, model_name: str = "nvidia/NV-Embed-v2", device: str = "cuda"
    ) -> None:
        """Initializes the SentenceTransformer model with the given name.

        Args:
            model_name (str): The name of the SentenceTransformer model to load.
            device (str): The device to run the model on ("cuda" or "cpu").
        """
        self.device = device
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"
        self.model = self.model.to(device)

    def add_eos(self, input_texts: list[str]) -> list[str]:
        """Adds the end-of-sequence token to the input texts.

        Args:
            input_texts: A list of input texts.

        Returns:
            A list of texts with the EOS token appended.
        """
        eos_token = self.model.tokenizer.eos_token or ""
        return [text + eos_token for text in input_texts]

    def embed_documents(self, input_texts: list[str]) -> list[list[float]]:
        """Encodes the input texts into embeddings using the model.

        Args:
            input_texts: The input text or list of texts to encode.

        Returns:
            The embeddings for the input texts.
        """
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        task_name_to_instruct = {
            "example": """
            Given a question, retrieve passages that answer the question
            """
        }
        query_prefix = (
        "Instruct: " + task_name_to_instruct["example"] + "\nQuery: "
        )

        texts_with_eos = self.add_eos(input_texts)

        embeddings = self.model.encode(
            texts_with_eos,
            batch_size=2,
            prompt=query_prefix,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=self.device
        )

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return embeddings

    def embed_query(self, query_text: str) -> list[float]:
        """Encodes a query text into an embedding using the model.

        Args:
            query_text: The query text to encode.

        Returns:
            The embedding for the query text.
        """
        return self.embed_documents([query_text])[0]


def get_embedding_function(device: str = "cuda") -> CustomSentenceTransformer:
    """Creates and returns a CustomSentenceTransformer object.

    Args:
        device (str): The device to run the model on ("cuda" or "cpu").

    Returns:
        CustomSentenceTransformer: An initialized custom embedding model.
    """
    return CustomSentenceTransformer(model_name="nvidia/NV-Embed-v2", device=device)

