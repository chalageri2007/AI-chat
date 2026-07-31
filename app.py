import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .doc-stat {
        background-color: #F1F5F9;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


def extract_text_from_files(uploaded_files):
    """Extract text from PDF, DOCX, and TXT files."""
    text_content = []
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_text = ""
        
        try:
            if file_name.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                for page_num, page in enumerate(reader.pages):
                    extracted = page.extract_text()
                    if extracted:
                        file_text += f"\n--- Page {page_num + 1} ({file_name}) ---\n" + extracted
            
            elif file_name.endswith(".docx"):
                doc = Document(uploaded_file)
                file_text += f"\n--- Document: {file_name} ---\n"
                for para in doc.paragraphs:
                    if para.text.strip():
                        file_text += para.text + "\n"
            
            elif file_name.endswith(".txt"):
                file_text += f"\n--- Document: {file_name} ---\n"
                file_text += uploaded_file.read().decode("utf-8")
                
            if file_text.strip():
                text_content.append({"filename": file_name, "content": file_text})
            else:
                st.warning(f"No readable text found in {file_name}")
                
        except Exception as e:
            st.error(f"Error reading {file_name}: {str(e)}")
            
    return text_content


def build_vector_store(text_chunks, api_key):
    """Create FAISS vector store from text chunks using Gemini Embeddings with fallback."""
    candidate_embeddings = [
        "models/gemini-embedding-001",
        "models/gemini-embedding-2",
        "gemini-embedding-001",
        "models/text-embedding-004",
        "embedding-001"
    ]
    
    embeddings = None
    last_error = None
    
    for model_name in candidate_embeddings:
        try:
            temp_emb = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key
            )
            # Test query to verify model works for API key
            temp_emb.embed_query("test")
            embeddings = temp_emb
            break
        except Exception as e:
            last_error = e
            continue
            
    if embeddings is None:
        raise RuntimeError(f"Could not initialize embedding model: {last_error}")

    vector_store = FAISS.from_texts(
        texts=[chunk.page_content for chunk in text_chunks],
        embedding=embeddings,
        metadatas=[chunk.metadata for chunk in text_chunks]
    )
    return vector_store


def format_docs(docs):
    """Helper to format retrieved documents into context string."""
    return "\n\n".join(f"[{doc.metadata.get('source', 'Unknown')}]: {doc.page_content}" for doc in docs)


def get_rag_chain(vector_store, api_key):
    """Create LCEL retrieval QA chain using available Gemini Chat model with fallback."""
    candidate_models = [
        "gemini-flash-latest",
        "models/gemini-flash-latest",
        "gemini-pro-latest",
        "models/gemini-pro-latest",
        "gemini-2.0-flash-lite",
        "models/gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "models/gemini-2.0-flash"
    ]
    
    llm = None
    last_error = None
    
    for model_name in candidate_models:
        try:
            temp_llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.3
            )
            # Test model with simple query
            temp_llm.invoke("Hi")
            llm = temp_llm
            break
        except Exception as e:
            last_error = e
            continue
            
    if llm is None:
        raise RuntimeError(f"Could not initialize LLM model: {last_error}")
    
    prompt = ChatPromptTemplate.from_template("""
You are an intelligent AI Document Assistant.
Answer the user's question accurately using only the provided context below.
If the context does not contain enough information to answer the question, clearly state that the document does not contain that information.

<context>
{context}
</context>

Question: {question}
Answer:""")
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    env_api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_api_key if env_api_key and env_api_key != "your_gemini_api_key_here" else "",
        type="password",
        help="Get your API key from Google AI Studio (https://aistudio.google.com/)"
    )
    
    active_api_key = api_key_input or (env_api_key if env_api_key != "your_gemini_api_key_here" else None)
    
    st.divider()
    st.header("📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    process_btn = st.button("🚀 Process Documents", use_container_width=True, type="primary")

# --- Initialize Session State ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []

# --- Main App Header ---
st.markdown('<div class="main-title">📄 AI Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload your documents and ask questions powered by Google Gemini RAG</div>', unsafe_allow_html=True)

# Document Processing Handler
if process_btn:
    if not active_api_key:
        st.error("Please enter a valid Google Gemini API Key in the sidebar or `.env` file.")
    elif not uploaded_files:
        st.warning("Please upload at least one document before processing.")
    else:
        with st.spinner("Extracting text and building vector database..."):
            extracted_docs = extract_text_from_files(uploaded_files)
            
            if extracted_docs:
                # Chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                
                all_chunks = []
                for doc in extracted_docs:
                    chunks = text_splitter.create_documents(
                        texts=[doc["content"]],
                        metadatas=[{"source": doc["filename"]}]
                    )
                    all_chunks.extend(chunks)
                
                try:
                    # Build Vector Store
                    vector_store = build_vector_store(all_chunks, active_api_key)
                    st.session_state.vector_store = vector_store
                    st.session_state.processed_docs = [d["filename"] for d in extracted_docs]
                    st.session_state.messages = [] # Reset chat for new docs
                    st.success(f"Successfully processed {len(extracted_docs)} document(s) into {len(all_chunks)} text chunks!")
                except Exception as e:
                    st.error(f"Failed to build vector store: {str(e)}")

# Display current document stats
if st.session_state.processed_docs:
    docs_list = ", ".join(st.session_state.processed_docs)
    st.markdown(f'<div class="doc-stat"><b>Active Documents:</b> {docs_list}</div>', unsafe_allow_html=True)

# --- Chat Interface ---
# Render existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User query input
if user_question := st.chat_input("Ask a question about your uploaded documents..."):
    # Display user message
    st.chat_message("user").write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    if not active_api_key:
        response_text = "⚠️ Please provide a Google Gemini API Key in the sidebar."
        st.chat_message("assistant").write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    elif st.session_state.vector_store is None:
        response_text = "⚠️ Please upload and process documents first before asking questions."
        st.chat_message("assistant").write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document content..."):
                try:
                    rag_chain, retriever = get_rag_chain(st.session_state.vector_store, active_api_key)
                    answer = rag_chain.invoke(user_question)
                    
                    st.write(answer)
                    
                    # Source document inspection
                    relevant_docs = retriever.invoke(user_question)
                    if relevant_docs:
                        with st.expander("🔍 View Referenced Sources"):
                            for i, doc in enumerate(relevant_docs):
                                source = doc.metadata.get("source", "Unknown")
                                st.markdown(f"**Source {i+1} ({source}):**")
                                st.caption(doc.page_content[:300] + "...")
                                
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"An error occurred while answering: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})




