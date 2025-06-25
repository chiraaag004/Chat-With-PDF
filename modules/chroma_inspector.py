import streamlit as st

def inspect_chroma(retriever):
    st.sidebar.markdown("🧪 **ChromaDB Inspector**")

    # Try to access the underlying vectorstore
    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore is None:
        st.sidebar.error("No vectorstore found in retriever.")
        return

    # Show basic info
    try:
        # Use the public API to get the number of documents
        doc_count = len(vectorstore.get()["ids"])
        st.sidebar.success(f"🔎 {doc_count} documents stored in ChromaDB.")
    except Exception as e:
        st.sidebar.error("Could not fetch document count.")
        st.sidebar.code(str(e))

    # Search inside the vectorstore
    query = st.sidebar.text_input("🔍 Test a query against ChromaDB")

    if query:
        try:
            results = vectorstore.similarity_search(query, k=3)
            st.sidebar.markdown("### Top Matching Chunks:")
            for i, doc in enumerate(results):
                st.sidebar.markdown(f"**Result {i+1}:**")
                st.sidebar.markdown(doc.page_content[:300] + "...")
                st.sidebar.markdown("---")
        except Exception as e:
            st.sidebar.error("Error querying ChromaDB")
            st.sidebar.code(str(e))