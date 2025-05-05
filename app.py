import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question:{input}
"""
)

# Streamlit UI
st.title("PDF Q&A RAG APPLICATION")
st.write("Please provide the PDF files")

# Sidebar
st.sidebar.title("Settings")
hf_api_key = st.sidebar.text_input("Enter your HuggingFace API key", type="password")
groq_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
models = st.sidebar.selectbox('Select the Language Model',["gemma2-9b-it","llama-3.1-8b-instant","llama3-8b-8192","meta-llama/llama-4-maverick-17b-128e-instruct"])
temp = st.sidebar.slider("Temperature",min_value=0.0,max_value = 1.0,value=0.7)
tokens = st.sidebar.slider('Max Tokens',min_value=50,max_value=300,value=150)
# File uploader
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

# Function to process PDFs and create vector DB
def create_vectors_embeddings_from_uploads(uploaded_files, groq_key, hf_api_key,temp,tokens,models):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file to proceed.")
        return

    if not hf_api_key:
        st.warning("Please enter your HuggingFace API key first.")
        return
    
    if not groq_key:
        st.warning("Please enter your GROQ API key first.")
        return


    # Set environment variable
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

    # Initialize models
    llm = ChatGroq(api_key=groq_key, model_name=models,temperature=temp,max_tokens=tokens)
    embedding = HuggingFaceEmbeddings(model_name="johnpaulbin/jina-embeddings-v3-128")

    # Store in session state if not already
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embedding

        # Load and process documents
        documents = []
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())
            loader = PyPDFLoader(uploaded_file.name)
            documents.extend(loader.load())

        st.session_state.docs = documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents[:50])
        st.session_state.final_documents = final_documents
        st.session_state.vectors = FAISS.from_documents(final_documents, embedding)
        st.session_state.llm = llm
        st.write("✅ Vector Database is ready")

# Button to create embeddings
if st.button("Document Embedding"):
    create_vectors_embeddings_from_uploads(uploaded_files, groq_key, hf_api_key,temp,tokens,models)

# Query input and response
query = st.text_input("Enter your query based on the documents")

if query:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' to load the documents first.")
    else:
        llm = st.session_state.llm
        retriever = st.session_state.vectors.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = rag_chain.invoke({'input': query})
        end = time.process_time()

        st.write("🧠 Answer:")
        st.write(response['answer'])
        st.caption(f"⏱️ Response time: {end - start:.2f} seconds")

        # Expandable for sources
        with st.expander("📄 Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("---")
