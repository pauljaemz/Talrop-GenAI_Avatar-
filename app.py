import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import backend modules
from vector_store import VectorStoreManager
from chat import build_rag_chain

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - "The Study Room" aesthetic - Dark Theme
st.markdown("""
<style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Black background */
    .stApp {
        background-color: #000000;
    }
    
    /* Sidebar - The Bookshelf */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #2a2a2a;
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    /* Main area - The Notebook */
    .main .block-container {
        background-color: #0a0a0a;
        padding: 2rem;
        border-radius: 8px;
        border: 1px solid #2a2a2a;
        max-width: 900px;
        margin: 2rem auto;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .stChatMessage * {
        color: #e0e0e0 !important;
    }
    
    /* Input field */
    .stChatInputContainer {
        border-top: 1px solid #2a2a2a;
        padding-top: 1rem;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        color: #e0e0e0;
    }
    
    .stButton button:hover {
        background-color: #2a2a2a;
        border-color: #3a3a3a;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Text */
    p, span, label {
        color: #e0e0e0;
    }
    
    /* Input fields */
    input, textarea {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        color: #e0e0e0 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
    }
    
    /* Remove extra padding */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Dividers */
    hr {
        border-color: #2a2a2a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vsm = None
    st.session_state.chain = None
    st.session_state.chat_history = []
    st.session_state.api_key_set = False

# Sidebar - "The Bookshelf"
with st.sidebar:
    st.title("📚 Study Materials")
    st.markdown("---")
    
    # API Key Section
    st.subheader("🔑 API Configuration")
    
    # Check for existing API key
    existing_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if existing_key:
        st.success("✅ API Key configured")
        st.session_state.api_key_set = True
    else:
        st.warning("⚠️ API Key required")
        api_key = st.text_input(
            "HuggingFace API Token",
            type="password",
            placeholder="hf_...",
            help="Get your free token from https://huggingface.co/settings/tokens"
        )
        
        if api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
            # Save to .env file
            with open(".env", "w") as f:
                f.write(f"HUGGINGFACEHUB_API_TOKEN={api_key}\n")
            st.session_state.api_key_set = True
            st.rerun()
    
    st.markdown("---")
    
    # File Upload Section
    st.subheader("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Add PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload study materials to chat about"
    )
    
    if uploaded_files and st.session_state.api_key_set:
        if st.button("📤 Process Documents", use_container_width=True, type="primary"):
            with st.spinner("Processing documents..."):
                # Initialize VectorStoreManager if not already done
                if not st.session_state.initialized:
                    try:
                        st.session_state.vsm = VectorStoreManager()
                        st.session_state.initialized = True
                    except Exception as e:
                        st.error(f"Error initializing: {e}")
                        st.stop()
                
                # Save uploaded files to temp directory
                temp_files = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_files.append(tmp_file.name)
                
                # Add documents to vector store
                try:
                    success, count = st.session_state.vsm.add_documents(temp_files)
                    
                    # Clean up temp files
                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    
                    if success:
                        # Build RAG chain
                        st.session_state.chain = build_rag_chain(st.session_state.vsm.vectorstore)
                        st.toast(f"✅ Added {count} chunks from {len(uploaded_files)} file(s)", icon="✅")
                    else:
                        st.toast("⚠️ No new documents added", icon="⚠️")
                        
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                    # Clean up temp files on error
                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
    
    st.markdown("---")
    
    # Statistics Section
    if st.session_state.initialized and st.session_state.vsm:
        st.subheader("📊 Statistics")
        stats = st.session_state.vsm.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", stats.split("|")[0].split(":")[1].strip())
        with col2:
            st.metric("Files", stats.split("|")[1].split(":")[1].strip())
    
    st.markdown("---")
    
    # Clear Memory Section
    st.subheader("🗑️ Clear Memory")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.toast("Chat history cleared", icon="🗑️")
            st.rerun()
    
    with col2:
        if st.button("Clear DB", use_container_width=True):
            if st.session_state.vsm:
                st.session_state.vsm.clear_database()
                st.session_state.chain = None
                st.session_state.chat_history = []
                st.toast("Database cleared", icon="🗑️")
                st.rerun()

# Main Area - "The Notebook"
st.title("📖 Study Assistant")
st.markdown("*Ask questions about your uploaded documents*")
st.markdown("---")

# Check if ready to chat
if not st.session_state.api_key_set:
    st.info("� Please configure your API key in the sidebar to begin")
elif not st.session_state.chain:
    st.info("👈 Please upload and process documents in the sidebar to begin")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the RAG chain
                    result = st.session_state.chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history[:-1]  # Exclude current message
                    })
                    
                    response = result["answer"]
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add AI response to history
                    st.session_state.chat_history.append(AIMessage(content=response))
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.chat_history.pop()  # Remove user message on error

# Footer
st.markdown("---")
st.caption("� Tip: Upload multiple documents for better context")
