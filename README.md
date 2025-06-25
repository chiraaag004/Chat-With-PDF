# 🦾 RagBot! — Multimodal PDF Chat with RAG

A Streamlit app for **chatting with your PDFs** using Retrieval-Augmented Generation (RAG) and multimodal LLMs.  
Upload one or more PDFs, extract and summarize their text, tables, and images, and ask questions with full conversational memory!

---

## Features

- **Multiple PDF Upload:** Upload and process several PDFs at once.
- **Content Extraction:** Extracts text, tables, and images from PDFs.
- **Summarization:** Summarizes text, tables, and images using LLMs (Groq for text/tables, Gemini for images).
- **Vector Store (ChromaDB):** Stores summaries and original content for efficient retrieval.
- **Multimodal RAG:** Answers questions using both text and images as context.
- **Conversational Chat:** Remembers previous chat turns for context-aware Q&A.
- **ChromaDB Inspector:** Inspect and test your vectorstore from the sidebar.
- **Export Chat:** Download your chat history.

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

### 3. Set up API keys

Create a `.env` file in the project root with:

```
GROQ_API_KEY="your_groq_api_key"
GOOGLE_API_KEY="your_google_api_key"
LANGCHAIN_API_KEY="your_langchain_api_key"
```

Or, if using Streamlit Cloud, add these to your app’s **Secrets**.

### 4. Run the app

```bash
streamlit run index.py
```

---

## Usage

1. **Upload PDFs** using the sidebar uploader.
2. Click **Submit to DB** to process and summarize.
3. Inspect your vectorstore in the sidebar.
4. **Chat** with your PDFs using the chat input at the bottom.
5. Download your chat history if desired.

---

## Project Structure

```
chat_with_pdf/
│
├── index.py                  # Main Streamlit app
├── modules/
│   ├── __init__.py           # Package initialization
│   ├── pdf_handler.py        # PDF upload and saving
│   ├── pdf_processing.py     # PDF parsing and image extraction
│   ├── summarization.py      # Summarization logic
│   ├── vectorstore_utils.py  # Vectorstore and retriever setup
│   ├── chroma_inspector.py   # ChromaDB inspector sidebar
│   ├── chat.py               # Chat logic and chat history
│   ├── prompt_utils.py       # Prompt and doc parsing utilities
│   └── llm_models.py         # LLM and embedding model setup
├── packages.txt
├── requirements.txt
├── .env                      # Your API keys (not included in repo)
└── README.md
```

---

## Requirements

See [`requirements.txt`](./requirements.txt) & [`packages.txt`](./packages.txt) for all dependencies.


---

## API Keys

- **GROQ_API_KEY**: For Groq Llama-3 summarization.
- **GOOGLE_API_KEY**: For Gemini models and embeddings.
- **LANGCHAIN_API_KEY**: For LangChain integrations.

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)
- [Groq](https://groq.com/)
- [Google Generative AI](https://ai.google.dev/)