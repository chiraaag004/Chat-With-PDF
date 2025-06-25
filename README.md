# 📄🤖 Chat with PDF (Multimodal RAG)

A Streamlit app and utility toolkit for **multimodal retrieval-augmented generation (RAG)** with PDFs.  
Upload a PDF, extract and summarize its text, tables, and images, and chat with it using advanced LLMs (Groq, Gemini) — all with context-aware, multimodal answers!

---

## Features

- **PDF Parsing:** Extracts text, tables, and images from uploaded PDFs.
- **Summarization:** Summarizes text, tables, and images using LLMs (Groq for text/tables, Gemini for images).
- **Vector Store:** Stores summaries and original content for efficient retrieval.
- **Multimodal RAG:** Answers questions using both text and images as context.
- **Conversational Memory:** Remembers previous chat turns for context-aware Q&A.
- **Streamlit UI:** Simple, interactive web interface.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/chat_with_pdf.git
cd chat_with_pdf
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root with your API keys:

```
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## Usage

1. **Upload a PDF** using the file uploader.
2. The app will extract and summarize the content.
3. **Ask questions** about your PDF in the chat interface.
4. The assistant will answer using both text and images from your document, remembering previous messages.

---

## Project Structure

```
chat_with_pdf/
│
├── app.py                # Streamlit app
├── utils.py              # Core PDF, summarization, vectorstore, and prompt utilities
├── requirements.txt      # Python dependencies
├── .env                  # Your API keys (not included in repo)
└── README.md             # This file
```

---

## Core Files

### `app.py`

- Streamlit UI for uploading PDFs and chatting.
- Handles chat history and displays summaries.

### `utils.py`

- `process_pdf`: Extracts texts, tables, and images from PDFs.
- `summarize_texts_and_tables`: Summarizes text and tables with Groq LLM.
- `summarize_images`: Summarizes images with Gemini.
- `build_vectorstore`, `add_documents_to_retriever`: Sets up vector store and adds content.
- `parse_docs`, `build_prompt`: Prepares multimodal context and prompts for Gemini.

---

## Requirements

See [`requirements.txt`](./requirements.txt) for all dependencies.

---

## API Keys

- **GOOGLE_API_KEY**: For Google Generative AI (Gemini models and embeddings).
- **GROQ_API_KEY**: For fast Llama-3 summarization.
- **LANGCHAIN_API_KEY**: For LangChain integrations and enhanced features.

---


## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)
- [Groq](https://groq.com/)
- [Google Generative AI](https://ai.google.dev/)
