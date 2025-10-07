"""Script for loading and splitting documents into smaller chunks.

This script provides functions to:
- Load text documents from a specified directory.
- Split loaded documents into smaller chunks using a recursive 
character splitter.

It is intended to process textual data for downstream tasks like 
embedding and database storage.
"""

import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Path to the directory containing PDF files
DATA_PATH = os.path.join(os.getcwd(), "data", "NY_DPS_Documents")


def load_documents() -> List[Document]:
    """
    Loads PDF documents as text from the specified directory.

    Reads `.pdf` files directly and returns them as `Document` objects.

    Returns:
        A list of `Document` objects loaded from the specified directory.
    """
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, filename)
            try:
                with open(file_path, "rb") as pdf_file:
                    content = ""
                    # Emulate PDF text extraction without external packages
                    for line in pdf_file:
                        content += line.decode("utf-8", errors="ignore")
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": filename}
                    )
                )
                print(f"Loaded document: {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller chunks using a recursive character splitter.

    Args:
        documents: A list of `Document` objects to be split into chunks.

    Returns:
        A list of smaller `Document` chunks created from the input documents.
    """
    # Initialize the text splitter with predefined chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    return text_splitter.split_documents(documents)
