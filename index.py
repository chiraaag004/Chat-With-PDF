import warnings
import logging
import streamlit as st

# Local modules
from modules.pdf_handler import upload_pdfs, save_uploaded_files
from modules.pdf_processing import process_pdf
from modules.summarization import summarize_texts_and_tables, summarize_images
from modules.vectorstore_utils import build_vectorstore, add_documents_to_retriever
from modules.chroma_inspector import inspect_chroma
from modules.chat import display_chat_history, handle_user_input, download_chat_history


# Silence noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(
    page_title="RagBot!",     
)

# App title
st.title("Ask Ragbot! ")

# Step 1: Upload PDFs + wait for submit
uploaded_files, submitted = upload_pdfs()

# Step 2: If user clicks submit, update vectorstore
if submitted and uploaded_files:
    file_paths = save_uploaded_files(uploaded_files)
    all_texts, all_tables, all_images = [], [], []
    with st.spinner(" Processing PDFs..."):
        for file_path in file_paths:
            texts, tables, images = process_pdf(file_path)
            all_texts.extend(texts)
            all_tables.extend(tables)
            all_images.extend(images)

        # Summarize texts and tables
        text_summaries, table_summaries = summarize_texts_and_tables(all_texts, all_tables)
        image_summaries = summarize_images(all_images) if all_images else [] 

    with st.spinner(" Building vectorstore..."):
        # Build vectorstore
        vectorstore = build_vectorstore()
        add_documents_to_retriever(
            vectorstore, all_texts, text_summaries, all_tables, table_summaries, all_images, image_summaries
        )
        
        # Store in session state
        st.session_state.vectorstore = vectorstore


# Step 3: Display vectorstore inspector (Sidebar)
if "vectorstore" in st.session_state:
    inspect_chroma(st.session_state.vectorstore)

# Step 4: Display old chat messages
display_chat_history()

# Step 5: Handle new prompt input
if "vectorstore" in st.session_state:
    handle_user_input({"retriever": st.session_state.vectorstore})

# Step 6: Chat history export
download_chat_history()