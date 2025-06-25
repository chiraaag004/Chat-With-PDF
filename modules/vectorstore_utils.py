import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

from modules.llm_models import embed_model

ID_KEY = "doc_id"

def build_vectorstore():
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=embed_model
    )
    store = InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=ID_KEY,
    )
    return retriever

def add_documents_to_retriever(retriever, texts, text_summaries, tables, table_summaries, images, image_summaries):
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={ID_KEY: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    if summary_texts:
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={ID_KEY: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    if summary_tables:
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))

    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_imgs = [
        Document(page_content=summary, metadata={ID_KEY: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    if summary_imgs:
        retriever.vectorstore.add_documents(summary_imgs)
        retriever.docstore.mset(list(zip(img_ids, images)))