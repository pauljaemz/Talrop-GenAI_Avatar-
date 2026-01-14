import os
import sys
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Personality presets
PERSONALITIES = {
    "professional": {
        "name": "Professional",
        "description": "Formal, precise, and business-like responses",
        "prompt": """You are a professional assistant for question-answering tasks. 
Maintain a formal and precise tone. Be thorough but concise. 
Use proper terminology and avoid casual language.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, clearly state that the information is not available in the provided documents.

Context:
{context}"""
    },
    "friendly": {
        "name": "Friendly",
        "description": "Warm, approachable, and conversational",
        "prompt": """You are a friendly and helpful assistant! 😊
Be warm, approachable, and conversational in your responses.
Use a casual but helpful tone, and feel free to use emojis occasionally.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say so in a friendly way!

Context:
{context}"""
    },
    "concise": {
        "name": "Concise",
        "description": "Brief, to-the-point responses",
        "prompt": """You are a concise assistant. Give brief, direct answers.
No unnecessary words. Get straight to the point.
Use bullet points when listing items.
Use the following context to answer. Say "Unknown" if not found.

Context:
{context}"""
    },
    "teacher": {
        "name": "Teacher",
        "description": "Educational, explains concepts thoroughly",
        "prompt": """You are an educational assistant and patient teacher.
Explain concepts thoroughly and break down complex ideas into simpler parts.
Use examples and analogies when helpful. Encourage learning.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, suggest what the user might look into to learn more.

Context:
{context}"""
    },
    "creative": {
        "name": "Creative",
        "description": "Imaginative, uses metaphors and storytelling",
        "prompt": """You are a creative and imaginative assistant!
Use metaphors, analogies, and vivid language to make your answers engaging.
Don't be afraid to be a little poetic or use storytelling elements.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, creatively acknowledge the gap in knowledge.

Context:
{context}"""
    },
    "custom": {
        "name": "Custom",
        "description": "User-defined personality",
        "prompt": """{custom_personality}

Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say so appropriately.

Context:
{context}"""
    }
}


class RAGChatbot:
    def __init__(self):
        self.vectorstore = None
        self.conversation = None
        self.messages = []  # Chat history for LCEL
        self.processed_files = []
        self.embeddings = None
        self.current_personality = "professional"
        self.custom_personality = ""

        # Check for HuggingFace API key
        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            api_key = input("Enter your HuggingFace API Token: ").strip()
            if api_key:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
            else:
                print("❌ HuggingFace API token is required.")
                sys.exit(1)

        # Initialize embeddings
        print("🔄 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✅ Embedding model loaded.")

        # Try to load existing vectorstore
        self._load_existing_vectorstore()

    def _load_existing_vectorstore(self):
        """Load existing vectorstore if available."""
        if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
            try:
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_DB_PATH,
                    embedding_function=self.embeddings
                )
                if len(self.vectorstore.get()["documents"]) > 0:
                    self.conversation = self._create_conversation_chain()
                    print(f"✅ Loaded existing database with {len(self.vectorstore.get()['documents'])} chunks.")
                    return True
            except Exception as e:
                print(f"⚠️ Could not load existing database: {str(e)}")
        return False

    def load_document(self, file_path: str):
        """Load a document based on its file type."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path.split(".")[-1].lower()

        if file_extension == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported: pdf, txt")

        return loader.load()

    def split_documents(self, documents):
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def _format_docs(self, docs):
        """Format documents for context."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _create_conversation_chain(self):
        """Create the conversational retrieval chain using LCEL."""
        endpoint = HuggingFaceEndpoint(
            repo_id="HuggingFaceTB/SmolLM3-3B",
            temperature=0.7,
            max_new_tokens=512,
        )

        llm = ChatHuggingFace(llm=endpoint)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Contextualize question prompt
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

        def contextualized_question(input_dict):
            if input_dict.get("chat_history"):
                return contextualize_q_chain.invoke(input_dict)
            return input_dict["input"]

        # Answer question prompt - uses current personality
        qa_system_prompt = self.get_personality_prompt()

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        def retrieve_and_format(input_dict):
            question = contextualized_question(input_dict)
            docs = retriever.invoke(question)
            return {
                "context": self._format_docs(docs),
                "input": input_dict["input"],
                "chat_history": input_dict.get("chat_history", []),
                "source_docs": docs
            }

        rag_chain = (
            RunnableLambda(retrieve_and_format)
            | RunnablePassthrough.assign(
                answer=lambda x: (qa_prompt | llm | StrOutputParser()).invoke({
                    "context": x["context"],
                    "input": x["input"],
                    "chat_history": x["chat_history"]
                })
            )
        )

        return rag_chain

    def add_documents(self, file_paths: list):
        """Process and add documents to the vectorstore."""
        all_chunks = []

        for file_path in file_paths:
            if file_path in self.processed_files:
                print(f"⚠️ File '{file_path}' has already been processed. Skipping.")
                continue

            try:
                print(f"🔄 Processing: {file_path}")
                documents = self.load_document(file_path)
                chunks = self.split_documents(documents)
                all_chunks.extend(chunks)
                self.processed_files.append(file_path)
                print(f"✅ Successfully processed: {file_path} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")

        if all_chunks:
            print("🔄 Adding to vector database...")
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=all_chunks,
                    embedding=self.embeddings,
                    persist_directory=CHROMA_DB_PATH
                )
            else:
                self.vectorstore.add_documents(all_chunks)

            self.conversation = self._create_conversation_chain()
            print(f"✅ Added {len(all_chunks)} chunks to database.")
            return True

        return False

    def chat(self, user_question: str) -> str:
        """Handle user input and generate response."""
        if self.conversation is None:
            return "❌ Please add documents first using the 'add' command."

        try:
            response = self.conversation.invoke({
                "input": user_question,
                "chat_history": self.messages
            })

            # Update chat history
            self.messages.append(HumanMessage(content=user_question))
            self.messages.append(AIMessage(content=response["answer"]))

            return response["answer"]

        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def clear_database(self):
        """Clear the ChromaDB database and reset state."""
        import shutil

        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
            os.makedirs(CHROMA_DB_PATH)

        self.vectorstore = None
        self.conversation = None
        self.messages = []
        self.processed_files = []
        print("✅ Database cleared successfully!")

    def get_stats(self):
        """Get database statistics."""
        if self.vectorstore:
            count = len(self.vectorstore.get()["documents"])
            return f"📊 Database: {count} chunks | Files: {len(self.processed_files)}"
        return "📊 Database: Empty"

    def set_personality(self, personality_type: str):
        """Set the chatbot's personality."""
        if personality_type in PERSONALITIES:
            self.current_personality = personality_type
            if personality_type == "custom":
                # For custom personality, ask the user to input the personality prompt
                custom_prompt = input("Enter your custom personality prompt: ").strip()
                self.custom_personality = custom_prompt
            # Recreate conversation chain with new personality
            if self.vectorstore is not None:
                self.conversation = self._create_conversation_chain()
            print(f"✅ Personality set to: {PERSONALITIES[personality_type]['name']}")
        else:
            print("❌ Invalid personality type. Available types: " + ", ".join(PERSONALITIES.keys()))

    def get_personality_prompt(self):
        """Get the prompt for the current personality."""
        personality = PERSONALITIES[self.current_personality]
        prompt = personality["prompt"]
        if self.current_personality == "custom":
            prompt = prompt.format(custom_personality=self.custom_personality)
        return prompt

    def list_personalities(self):
        """List all available personalities."""
        print("\n🎭 Available Personalities:")
        print("-" * 50)
        for key, value in PERSONALITIES.items():
            current = " (current)" if key == self.current_personality else ""
            print(f"  • {key}: {value['description']}{current}")
        print("-" * 50)
        print(f"\nCurrent personality: {PERSONALITIES[self.current_personality]['name']}")
        print("Use 'personality <type>' to change.\n")


def print_help():
    """Print help message."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    🤖 RAG Chatbot Commands                   ║
╠══════════════════════════════════════════════════════════════╣
║  add <file1> [file2] ...  - Add PDF or TXT files            ║
║  chat                     - Start chatting (or just type)   ║
║  stats                    - Show database statistics         ║
║  clear                    - Clear the database               ║
║  history                  - Show chat history                ║
║  reset                    - Reset chat history               ║
║  help                     - Show this help message           ║
║  exit / quit              - Exit the chatbot                 ║
║                                                              ║
║  🎭 Personality Commands:                                    ║
║  personality <type>       - Set chatbot personality          ║
║  personalities            - List available personalities     ║
║                                                              ║
║  Available personalities: professional, friendly, concise,   ║
║                          teacher, creative, custom           ║
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    """Main function to run the terminal chatbot."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              🤖 RAG Chatbot - Terminal Edition               ║
║     Upload PDF or TXT files and ask questions about them     ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Initialize chatbot
    chatbot = RAGChatbot()

    print_help()
    print(chatbot.get_stats())
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Parse commands
            parts = user_input.split()
            command = parts[0].lower()

            if command in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break

            elif command == "help":
                print_help()

            elif command == "add":
                if len(parts) < 2:
                    print("❌ Usage: add <file1> [file2] ...")
                else:
                    file_paths = parts[1:]
                    chatbot.add_documents(file_paths)

            elif command == "stats":
                print(chatbot.get_stats())

            elif command == "clear":
                confirm = input("⚠️ Are you sure you want to clear the database? (yes/no): ").strip().lower()
                if confirm == "yes":
                    chatbot.clear_database()
                else:
                    print("Cancelled.")

            elif command == "history":
                if not chatbot.messages:
                    print("📜 No chat history yet.")
                else:
                    print("\n📜 Chat History:")
                    print("-" * 50)
                    for msg in chatbot.messages:
                        if isinstance(msg, HumanMessage):
                            print(f"You: {msg.content}")
                        else:
                            print(f"Bot: {msg.content}")
                        print()
                    print("-" * 50)

            elif command == "reset":
                chatbot.messages = []
                print("✅ Chat history reset.")

            elif command == "chat":
                print("💬 Chat mode. Type your questions (type 'back' to exit chat mode)")
                while True:
                    question = input("You: ").strip()
                    if question.lower() == "back":
                        break
                    if question:
                        print("🤔 Thinking...")
                        response = chatbot.chat(question)
                        print(f"\n🤖 Bot: {response}\n")

            elif command == "personality":
                if len(parts) < 2:
                    print("❌ Usage: personality <type>")
                else:
                    personality_type = parts[1].lower()
                    chatbot.set_personality(personality_type)

            elif command == "personalities":
                chatbot.list_personalities()

            else:
                # Treat as a question
                print("🤔 Thinking...")
                response = chatbot.chat(user_input)
                print(f"\n🤖 Bot: {response}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
