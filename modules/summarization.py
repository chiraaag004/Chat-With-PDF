from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google import genai
import streamlit as st
import os

from modules.llm_models import groq_llm, gemini_llm

def summarize_texts_and_tables(texts, tables):
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

    try:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
    except Exception as e:
        import traceback
        st.error("Summarization failed!")
        st.code(traceback.format_exc())
    
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
    
    return text_summaries, table_summaries

def summarize_images(images):
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