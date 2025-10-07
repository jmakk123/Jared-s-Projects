"""Script for querying a ChromaDB database using a query text.

This script allows users to:
- Query a ChromaDB database for relevant context based on a
user-provided query text.
- Retrieve and display the most relevant results with similarity scores.

Note: This implementation does not use an LLM; it focuses solely
on retrieving relevant context.
"""

import argparse
import chromadb
import EmbedParsedText as embedder
from langchain_community.vectorstores import Chroma

# Constants for the ChromaDB path and collection name
CHROMA_PATH = "data/chromadb"
COLLECTION_NAME = "bdc_collection"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str) -> str:
    """Queries the ChromaDB database with the given query text.

    Args:
        query_text: The input query to search for relevant context.

    Returns:
        A string message indicating the completion of the query process.

    Example:
        >>> query_rag("What is opinion on data acquisition?")
        "Query completed successfully."

        but printed output is, e.g.:
        # Content: collected. We must have the opportunity to collect
        # system-wide and individual system data in order to analyze
        # Coefficients of Performance (COP) and electric demand
        # profiles in order to measure the efficacy of UTENs. We
        # strongly recommend NYSDPS staff consult with data scientists
        # already working on data acquisition for TENs across North
        # America to determine appropriate sampling sizes. The Learning
        # from the Ground Up (LeGUp) initiative is a collaboration of
        # various organizations and national laboratories seeking key
        # data points, establishing a database, and publishing findings
        # and insights from TEN pilot projects. We have submitted
        # information that should be collected in a previous filing in
        # this matter. We recommend all New York UTEN Pilots join the
        # LeGUp effort and follow the work
        # Metadata: {'id': 'data/BDC_Letter.txt:None:3'}
        # Similarity Score (Distance): 1.3391986759725651
        # --------------------------------------------------
    """
    # Initialize embedding function and ChromaDB client
    embedding_function = embedder.get_embedding_function()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Initialize Chroma retriever
    print("Loading retriever...")
    chromadb_retriever = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        client=client,
    )
    print("Retriever loaded successfully.")

    # Perform the query
    print("Querying vector database...")
    results = chromadb_retriever.similarity_search_with_score(query=query_text, k=10)

    return results


def main() -> None:
    """Main entry point for the script.

    Parses command-line arguments and invokes the query function
    with the user-provided query text.
    """
    # Create a command-line interface
    parser = argparse.ArgumentParser(
        description="Query ChromaDB with a given query text."
    )
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    # Execute the query
    query_rag(args.query_text)


if __name__ == "__main__":
    main()
