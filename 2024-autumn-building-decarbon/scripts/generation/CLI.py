"""Command-line interface for querying the RAG pipeline and LLaMA model."""

import CreatePrompt as prompt_builder
import GenerateResponse as generator
import RetrieveTextFromQuery as retriever

def main() -> None:
    """Main function to handl the command-line interface.

    Integrates RAG, ChromaDB retrieval, and LLaMA model for
    Building Decarbonization Coalition queries.
    """
    # Load the model and processor
    print("Loading model and processor...")
    model, processor, device = generator.setup_model_and_processor()
    print("Model loaded successfully!\n")

    print(
        "Welcome to the command-line interface for the "
        "BDC chatbot, designed to answer questions regarding "
        "the Building Decarbonization Coalition using information "
        "from pilot projects in New York!"
    )
    print("Type 'exit' to quit the program.\n")

    while True:
        # Prompt the user for a query
        query = input("Enter your question or type 'exit' to exit: ").strip()

        # Exit condition
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve relevant documents using the RAG pipeline
        print("\nRetrieving relevant documents...\n")
        retrieved_docs = retriever.query_rag(query)
        if not retrieved_docs:
            print(
                """No relevant documents found in the database. 
                Please try rephrasing your query."""
            )
            continue

        # Build the prompt using the retrieved documents and user query
        print("Building prompt with retrieved context...\n")
        context = "\n\n".join([doc.page_content for doc, _ in retrieved_docs])
        sources = [
            doc.metadata.get("id", "Unknown Source") 
            for doc, _ in retrieved_docs
        ]

        combined_prompt = prompt_builder.prepare_prompt(
            user_query=query,
            relevant_docs_text=context,
        )

        # Generate a response using the LLaMA model
        print("Generating response...\n")
        response = generator.generate_response(
            combined_prompt, model, processor, device
        )

        print("Response:")
        print(response["response"])
        if response["sources"]:
            print("\nSources:")
            for source in response["sources"]:
                print(f"- {source}")
        elif sources:
            print("\nSources:")
            for source in sources:
                print(f"- {source}")
        print("-" * 50)


if __name__ == "__main__":
    main()
