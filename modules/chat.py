import streamlit as st
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.prompt_utils import parse_docs, build_prompt
from modules.llm_models import GEMINI_MODEL

def display_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        st.chat_message("user" if msg["role"] == "user" else "assistant").markdown(msg["content"])

def handle_user_input(pdf_data):
    user_question = st.chat_input("Your question:")
    if not user_question:
        return

    # Display user message immediately
    st.chat_message("user").markdown(user_question)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # Build conversation context (last 10 exchanges)
    conversation = ""
    for msg in st.session_state.chat_history[-10:]:
        prefix = "User:" if msg["role"] == "user" else "Assistant:"
        conversation += f"{prefix} {msg['content']}\n"

    try:
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

        # Display assistant message immediately
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.error(f"Error: {str(e)}")
        

def download_chat_history():
    if st.session_state.get("chat_history"):
        content = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
        st.download_button("💾 Download Chat History", content, file_name="chat_history.txt", mime="text/plain")