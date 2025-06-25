import os
import io
import uuid
import base64
import tempfile
from dotenv import load_dotenv
from base64 import b64decode

from unstructured.partition.pdf import partition_pdf
from langchain.schema.document import Document

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

load_dotenv()

# Model names
GROQ_MODEL = "llama-3.1-8b-instant"
GEMINI_MODEL = "gemini-1.5-flash-8b"
EMBEDDING_MODEL = "models/embedding-001"
ID_KEY = "doc_id"

# Initialize LLMs and embeddings
groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)
gemini_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
embed_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)



def process_pdf(file_bytes):
    # Extracts texts, tables, and images from a PDF file (bytes).
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        chunks = partition_pdf(
            filename=tmp_path,
            infer_table_structure=True,
            strategy='hi_res',
            extract_image_block_types=['Image'],
            extract_image_block_to_payload=True,
            chunking_strategy='by_title',
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )
    finally:
        os.remove(tmp_path)

    texts, tables = [], []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    images = get_images_base64(chunks)

    return texts, tables, images



def get_images_base64(chunks):
    # Extracts base64-encoded images from PDF chunks.
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64



def summarize_texts_and_tables(texts, tables):
    # Summarizes text and table chunks using Groq LLM.

    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element": lambda x: x} | prompt | groq_llm | StrOutputParser()

    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

    return text_summaries, table_summaries



def summarize_images(images):
    # Summarizes images using Gemini multimodal model.

    genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    prompt_template = """Describe the image in detail.
    Do not start your message by saying "Here is a description" or anything like that.
    Do not hallucinate or make up details that are not present in the image."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ])
    chain = prompt | gemini_llm | StrOutputParser()

    image_summaries = chain.batch(images)
    return image_summaries



def build_vectorstore():
    # Creates a Chroma vectorstore and in-memory docstore for RAG.
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=embed_model,
        persist_directory=None  # or just omit this argument
    )
    store = InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=ID_KEY,
    )
    return retriever


def add_documents_to_retriever(retriever, texts, text_summaries, tables, table_summaries, images, image_summaries):
    # Adds summarized documents and originals to the retriever.
    
    # Texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={ID_KEY: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    if summary_texts:
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

    
    # Tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={ID_KEY: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    if summary_tables:
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))

    
    # Images
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_imgs = [
        Document(page_content=summary, metadata={ID_KEY: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    if summary_imgs:
        retriever.vectorstore.add_documents(summary_imgs)
        retriever.docstore.mset(list(zip(img_ids, images)))

import base64
from base64 import b64decode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

def parse_docs(docs):
    
    # Split retrieved docs into base64 images and text.
    # Returns a dict: {"images": [...], "texts": [...]}
    
    b64 = []
    text = []
    for doc in docs:
        # If doc is a Document, get its content
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        try:
            # Try to decode as base64 (image)
            b64decode(content)
            b64.append(content)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}



def build_prompt(kwargs):
    
    # Build a multimodal prompt for Gemini with context (text, tables, images) and user question.
    
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            # If Document, get text; else use as is
            context_text += text_element.page_content if hasattr(text_element, "page_content") else str(text_element)

    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )
