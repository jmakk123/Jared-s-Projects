"""Script to manage Chroma database operations.

This script provides functionality for interacting with a Chroma
database, including:
- Checking existing collections.
- Adding new document chunks with embeddings to the database.
- Clearing the database if needed.

It uses external modules for splitting and embedding text from
parsed PDFs.
"""

import argparse
import shutil
import os
import chromadb
from langchain.schema import Document

import EmbedParsedText as embedder
import SplitParsedPDF as splitter

CHROMA_PATH = "/net/scratch2/2024-autumn-building-decarbon/data/chromadb"
COLLECTION_NAME = "bdc_collection"

def check_existing_collections(chroma_path: str) -> list[str]:
    """Lists existing collections in the Chroma database.

    Args:
        chroma_path: Path to the Chroma database.

    Returns:
        A list of names of the existing collections in the database.
    """
    client = chromadb.PersistentClient(path=chroma_path)
    existing_collections = client.list_collections()
    collection_names = [collection.name for collection in existing_collections]
    print("Existing Collections:", collection_names)
    return collection_names


def add_to_chroma(chunks: list[Document]) -> None:
    """Adds new chunks to the Chroma db after finding the document ID's.

    Args:
        chunks: List of `Document` objects to be added to the database.
    """
    # Initialize Chroma database client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    db = client.get_or_create_collection(name=COLLECTION_NAME)

    # Calculate unique IDs for each chunk
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Check for existing documents in the database
    existing_items = db.get(include=["documents"])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    print("Documents in DB before population:", existing_items["documents"])

    # Filter out already existing chunks
    new_chunks = [
        chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids
    ]

    # Debugging: Check new chunks
    print(f"New chunks to add: {len(new_chunks)}")
    if not new_chunks:
        print("No new documents to add.")
        return

    # Add new chunks to the database with embeddings
    embedding_function = embedder.get_embedding_function()
    for i, chunk in enumerate(new_chunks):
        print(f"Processing chunk {i + 1} of {len(new_chunks)}...")
        embedding = embedding_function.embed_documents([chunk.page_content])
        db.add(
            documents=[chunk.page_content],
            ids=[chunk.metadata["id"]],
            embeddings=embedding,
            metadatas=[
                {"id": chunk.metadata["id"], "collection_id": COLLECTION_NAME}
            ],
        )
    print("New documents added successfully.")


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Calculates unique IDs for each chunk based on source, page, and index.

    Args:
        chunks: List of `Document` objects for which IDs need to be
        calculated.

    Returns:
        The input list of `Document` objects with updated `id` in
        their metadata.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown_source")
        page = chunk.metadata.get("page", "unknown_page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def clear_database(chroma_path: str) -> None:
    """Clears the Chroma database by deleting its directory.

    Args:
        chroma_path: Path to the Chroma database directory.
    """
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        print("Database cleared successfully.")
    else:
        print("Database path does not exist, no action taken.")


def main() -> None:
    """Main entry point for managing the Chroma database."""
    parser = argparse.ArgumentParser(description="Manage Chroma database operations.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    # Reset the database if --reset is passed
    if args.reset:
        print("Clearing database...")
        clear_database(CHROMA_PATH)

    # Load documents and split them into chunks
    print("Loading documents...")
    documents = splitter.load_documents()

    if not documents:
        print("No documents found. Ensure the data directory is populated correctly.")
        return

    print(f"Loaded {len(documents)} documents.")
    print("Splitting documents into chunks...")
    chunks = splitter.split_documents(documents)

    if not chunks:
        print("No chunks generated. Check document splitting logic.")
        return

    print(f"Generated {len(chunks)} chunks.")
    print("Adding chunks to Chroma database...")
    add_to_chroma(chunks)


if __name__ == "__main__":
    main()
