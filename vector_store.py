import os
import shutil
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_DB_PATH = "./chroma_db"


def _ensure_hf_token() -> None:
    """
    Ensure HUGGINGFACEHUB_API_TOKEN exists.
    - loads .env
    - if missing, prompt user once
    """
    load_dotenv()
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if api_key:
        return

    api_key = input("Enter your HuggingFace API Token: ").strip()
    if not api_key:
        raise RuntimeError("HuggingFace API token is required.")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key


def _get_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_document(file_path: str):
    """Load a document based on its file type."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.split(".")[-1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: pdf, txt")

    return loader.load()



def split_documents_semantic(
    documents,
    embeddings: HuggingFaceEmbeddings,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: int = 90,
    min_chunk_size: int = 200,
):
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        min_chunk_size=min_chunk_size,
    )
    return chunker.split_documents(documents)


@dataclass
class VectorStoreManager:
    chroma_path: str = CHROMA_DB_PATH
    processed_files: List[str] = field(default_factory=list)
    processed_file_hashes: Dict[str, str] = field(default_factory=dict)

    embeddings: Optional[HuggingFaceEmbeddings] = None
    vectorstore: Optional[Chroma] = None

    def __post_init__(self):
        os.makedirs(self.chroma_path, exist_ok=True)

        _ensure_hf_token()

        # Initialize embeddings once
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load/attach to existing persisted Chroma directory
        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings,
        )

    def add_documents(
        self,
        file_paths: List[str],
    ) -> Tuple[bool, int]:
        """
        Process and add documents to the vectorstore.
        Returns: (success, added_chunks_count)
        """
        all_chunks = []

        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue

            if file_path in self.processed_files:
                print(f"⚠️ File '{file_path}' has already been processed. Skipping.")
                continue

            file_hash = _get_file_hash(file_path)
            if file_hash in self.processed_file_hashes:
                original_file = self.processed_file_hashes[file_hash]
                print(f"⚠️ File '{file_path}' has identical content to '{original_file}'. Skipping.")
                continue

            try:
                print(f"🔄 Processing: {file_path}")
                documents = load_document(file_path)

                chunks = split_documents_semantic(documents, embeddings=self.embeddings)

                all_chunks.extend(chunks)
                self.processed_files.append(file_path)
                self.processed_file_hashes[file_hash] = file_path
                print(f"✅ Successfully processed: {file_path} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")

        if not all_chunks:
            return (False, 0)

        print("🔄 Adding to vector database...")
        # vectorstore already exists (persist dir attached), so just add
        self.vectorstore.add_documents(all_chunks)
        print(f"✅ Added {len(all_chunks)} chunks to database.")
        return (True, len(all_chunks))

    def clear_database(self) -> None:
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
        os.makedirs(self.chroma_path, exist_ok=True)

        # Reattach fresh empty store
        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings,
        )

        self.processed_files = []
        self.processed_file_hashes = {}
        print("✅ Database cleared successfully!")

    def get_chunk_count(self) -> int:
        if not self.vectorstore:
            return 0
        data = self.vectorstore.get()
        return len(data.get("documents", []))

    def get_stats(self) -> str:
        return f"📊 Database: {self.get_chunk_count()} chunks | Files: {len(self.processed_files)}"