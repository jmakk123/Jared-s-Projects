"""Script for creating and preparing the propmt tempalte for the LLM.

This script allows users to:
- Prepare the prompt + instruct the LLM.
- Process relevant docs to include as part of the prompt.
- Combines prompt + relevant docs to create one prompt for the LLM.
"""

from RetrieveTextFromQuery import query_rag


def prepare_prompt(relevant_docs_text: str, user_query: str) -> str:
    """Prepares a prompt for the Llama model.
    
    Does this by combining the provided document content and user query.

    Parameters:
        relevant_docs_text (str): Combined text of the relevant documents
        user_query (str): The user's query

    Returns:
        str: The prepared prompt for the Llama model
    """
    system_instruction = (
        "You are an AI chatbot for the Building Decarbonization Coalition, "
        "using a database of documents from the NY Department of Public Service. "
        "Answer queries by integrating the provided documents with the query. "
        "Respond in English only, using information from the relevant documents "
        f"provided alongside this prompt. Here is the user query: {user_query}"
        f"to answer this query, use information in {relevant_docs_text}"
    )

    return (
        f"{system_instruction}\n\n"
        f"Relevant Context:\n{relevant_docs_text}\n\n"
        f"Query:\n{user_query}\n\n"
    )


def process_relevant_docs(query_text: str) -> str:
    """Queries the ChromaDB database and processes relevant document content.

    Parameters:
        query_text (str): The input query to search for relevant context.

    Returns:
        str: Combined document text for prompt preparation.
    """
    print("\nQuerying ChromaDB...")
    results = query_rag(query_text)

    relevant_docs = [
        line[len("Content:") :].strip()
        for line in results.split("\n")
        if line.startswith("Content:")
    ]
    return "\n\n".join(relevant_docs).strip()


def generate_pipeline_output(user_query: str) -> str: 
    """Combines relevant document content and user query.
    
    This generates the final GenAI prompt.

    Parameters:
        user_query (str): The user's query.

    Returns:
        str: The prepared prompt for the Llama model.
    """
    relevant_docs_text = process_relevant_docs(user_query)
    response = prepare_prompt(relevant_docs_text, user_query)
    return response

if __name__ == "__main__":
    user_query = input("Enter your query: ")

    llama_prompt = generate_pipeline_output(user_query)

    print("\nPrepared Prompt for Llama Model:")
    print(llama_prompt)