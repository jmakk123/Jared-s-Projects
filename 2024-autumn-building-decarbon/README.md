# 2024-2025 Building Decarbonization



# Building Decarbonization Coalition (BDC) Chatbot

The BDC chatbot is an AI-driven solution developed to assist stakeholders in efficiently retrieving insights and data about Thermal Energy Networks (TENs) from New York City's pilot projects. It uses advanced retrieval and language modeling techniques to provide reliable answers and document references.

---

## Project Background

The BDC is dedicated to promoting climate-friendly energy solutions. TENs represent a major innovation in HVAC technology, offering improved energy efficiency compared to traditional systems. Publicly available data on NYC's TEN pilot projects is stored in the NY Department of Public Service database, but its unstructured nature presents challenges for stakeholders looking to evaluate their success.

This project focuses on addressing those challenges by creating a conversational AI chatbot capable of retrieving insights directly from the database.

---

## Features

- **Web Scraper**: Extracts and parses PDFs from the NY Department of Public Service database.
- **NVIDIA RAG Model**: Generates embeddings for document retrieval.
- **Llama Instruct Model**: Processes user queries in conjunction with relevant document data.
- **LangChain CLI**: Combines the components into a seamless Command Line Interface (CLI).

---

## Project Goals

1. Provide accurate and efficient retrieval of TEN-related data from the BDC.
2. Answer stakeholder queries with precise document references.
3. Lay the groundwork for future fine-tuning to enhance AI performance and accuracy.

---

## System Components

1. **Word Preprocessing**:
   - PDF files are converted into textual data using a web scraper.
   - Documents are split into smaller chunks for efficient storage and retrieval.

2. **Retrieval Augmented Generation**
   - Uses NVIDIA RAG technology to retrieve contextually relevant documents and generating factually-correct responses.  

3. **ChromaDB**:
   - Chunks are stored and indexed for quick similarity-based retrieval.

4. **Command Line Interface**:
   - A user query is combined with retrieved document text to generate an AI prompt.
   - Llama Instruct Model processes the prompt and generates a detailed response.

---

## Repository Structure

### `scripts`
- **preprocessing**:
  - `Webscraper.py`: Downloads and parses PDF files from the NY Department of Public Service.
  - `SplitParsedPDF.py`: Splits parsed documents into manageable text chunks.
  - `EmbedParsedText.py`: Generates embeddings for document chunks.
  - `PopulateDatabase.py`: Populates the ChromaDB with text chunks and embeddings.

- **model_training**:
  - `TuneGen.py`: Empty file made for future fine tuning of the Llama model.
  - `TuneRAG.py`: Empty file made for future fine tuning of RAG model.

- **generation**:
  - `RetrieveTextFromQuery.py`: Queries the database for relevant document text.
  - `CreatePrompt.py`: Combines user queries with retrieved text to prepare AI prompts.
  - `GenerateResponse.py`: Uses Llama Instruct Model to process AI prompts.

### `data`
Contains the raw and processed data used in the project:
- `NY_DPS_Documents/`: PDF files downloaded from the NY Department of Public Service.
- link to data source: https://documents.dps.ny.gov/public/MatterManagement/CaseMaster.aspx?MatterCaseNo=22-m-0429&CaseSearch=Search
- `chromadb/`: The Chroma database containing document embeddings and metadata.

### `utils`
Reusable utilities and helper functions.

### `notebooks`
Jupyter notebooks demonstrating system functionality.

### `output`
Generated outputs from running the system.

---

## Usage

### Prerequisites

- Install Python dependencies:

        pip install -r requirements.txt

- Separately, run these in the terminal as well:
      
      pip install transformers -U
      
      pip install --upgrade torchvision

- Ensure the following environment variables are set:

        export HUGGING_FACE_API_KEY=YOUR_HUGGING_FACE_API_KEY

- You need to make an account with Hugging Face and request access to Nvidia-Embed-v2 and 
meta-llama/Llama-3.2-11B-Vision-Instruct. 
- You will get one API_KEY that you export as
an ENV variable before you move to the next step. 

- Make sure the NY_DPS_Documents directory is populated with PDF files by running the Web Scraper.

        python scripts/generation/Webscraper.py

- Populate the ChromaDB with text chunks:

        python scripts/generation/PopulateDatabase.py

- Use the CLI for querying the chatbot:

        python scripts/generation/CLI.py

- Input your query when prompted, and the chatbot will return relevant answers and references.

## Next Steps: 
 
1. Fine-tune the Llama model for improved response accuracy.
2. Create a web-based interface for enhanced user experience.

## Contributors: 

- **Jared Maksoud**  
  - Email: jmaksoud@uchicago.edu
  
- **Andres Guerrero**  
  - Email: andresg@uchicago.edu  

- **Andrew Kaboski**  
  - Email: akaboski@uchicago.edu  

- **Victor Zhang**  
  - Email: victorzhang@uchicago.edu  

For questions or support, contact the contributors listed above.

If having issues with chrome driver, please see this link: https://www.nuget.org/packages/selenium.webdriver.chromedriver/

- To to get permissions for chromedriver run chmod +x /pathtochrome