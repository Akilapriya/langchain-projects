import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# 1. Setup API Keys and Environment
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.title("RAG Document Q&A")

# 2. Initialize LLM
llm = ChatGroq(model="llama3-8b-8192")

# 3. Define the Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)


# 4. Function to initialize Vector Embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama3")
        st.session_state.loader = WebBaseLoader("https://langchain.com")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(st.session_state.docs)
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )


# 5. UI Elements
user_prompt = st.text_input("Enter your query from the documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Database is ready")

import time

# 6. The Retrieval Logic (Fixed Typo)
if "vectors" in st.session_state:
    # Creates the chain to pass documents to LLM
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Converts vector store into a retriever
    retriever = st.session_state.vectors.as_retriever()

    # Creates the final retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    if user_prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})

        st.write(f"Response time: {time.process_time() - start} seconds")
        st.write(response["answer"])

        # With a streamlit expander for context snippets
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
