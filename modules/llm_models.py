from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

GROQ_MODEL = "llama-3.1-8b-instant"
GEMINI_MODEL = "gemini-1.5-flash-8b"
EMBEDDING_MODEL = "models/embedding-001"

groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)
gemini_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
embed_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)