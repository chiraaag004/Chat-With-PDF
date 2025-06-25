import streamlit as st

from utils import (
    process_pdf,
    summarize_texts_and_tables,
    summarize_images,
    build_vectorstore,
    add_documents_to_retriever,
    parse_docs,
    build_prompt,
    GEMINI_MODEL,
)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Chat with PDF (Multimodal)", layout="wide")
st.title("📄🤖 Chat with PDF (Multimodal RAG)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

def process_and_store_pdf(pdf_bytes):
    texts, tables, images = process_pdf(pdf_bytes)
    text_summaries, table_summaries = summarize_texts_and_tables(texts, tables)
    image_summaries = summarize_images(images) if images else []
    retriever = build_vectorstore()
    add_documents_to_retriever(
        retriever, texts, text_summaries, tables, table_summaries, images, image_summaries
    )
    st.session_state['pdf_data'] = {
        "texts": texts,
        "tables": tables,
        "images": images,
        "text_summaries": text_summaries,
        "table_summaries": table_summaries,
        "image_summaries": image_summaries,
        "retriever": retriever,
    }

if uploaded_file and "pdf_data" not in st.session_state:
    with st.spinner("Processing PDF..."):
        pdf_bytes = uploaded_file.read()
        process_and_store_pdf(pdf_bytes)
    st.success("PDF processed! You can now ask questions about it.")

if "pdf_data" in st.session_state:
    pdf_data = st.session_state["pdf_data"]
    # Show summaries
    with st.expander("Show extracted summaries"):
        st.subheader("Text Summaries")
        for i, summary in enumerate(pdf_data["text_summaries"]):
            st.markdown(f"**Text {i+1}:** {summary}")

        st.subheader("Table Summaries")
        for i, summary in enumerate(pdf_data["table_summaries"]):
            st.markdown(f"**Table {i+1}:** {summary}")

        st.subheader("Image Summaries")
        for i, summary in enumerate(pdf_data["image_summaries"]):
            st.markdown(f"**Image {i+1}:** {summary}")

    # Chat interface
    st.header("Ask a question about your PDF")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Your question:")

    if st.button("Send") and user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        conversation = ""
        for msg in st.session_state.chat_history[-10:]:
            prefix = "User:" if msg["role"] == "user" else "Assistant:"
            conversation += f"{prefix} {msg['content']}\n"

        with st.spinner("Generating answer..."):
            chain = (
                {
                    "context": pdf_data["retriever"] | RunnableLambda(parse_docs),
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(build_prompt)
                | ChatGoogleGenerativeAI(model=GEMINI_MODEL)
                | StrOutputParser()
            )
            answer = chain.invoke(conversation)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style="background-color:#e3f2fd; padding:10px; border-radius:8px; margin-bottom:8px;">
                    <strong style="color:#1565c0;">You:</strong>
                    <span style="color:#0d47a1;">{msg['content']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:#f1f8e9; padding:10px; border-radius:8px; margin-bottom:8px;">
                    <strong style="color:#388e3c;">Assistant:</strong>
                    <span style="color:#1b5e20;">{msg['content']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
