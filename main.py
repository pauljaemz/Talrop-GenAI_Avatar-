from langchain_core.messages import HumanMessage, AIMessage
from vector_store import VectorStoreManager
from chat import build_rag_chain


def print_help():
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
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              🤖 RAG Chatbot - Terminal Edition               ║
║     Upload PDF or TXT files and ask questions about them     ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # state
    messages = []
    rag_chain = None

    # init vector store manager (loads embeddings + attaches to persistent Chroma dir)
    vsm = VectorStoreManager()

    print_help()
    print(vsm.get_stats())
    print()

    def ensure_chain_ready():
        nonlocal rag_chain
        if rag_chain is None:
            # match chat.py defaults (k=5)
            rag_chain = build_rag_chain(vsm.vectorstore, k=5)

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

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
                    continue

                file_paths = parts[1:]
                ok, added = vsm.add_documents(file_paths)
                if ok and added > 0:
                    ensure_chain_ready()

            elif command == "stats":
                print(vsm.get_stats())

            elif command == "clear":
                confirm = input("⚠️ Are you sure you want to clear the database? (yes/no): ").strip().lower()
                if confirm == "yes":
                    vsm.clear_database()
                    rag_chain = None
                    messages = []
                else:
                    print("Cancelled.")

            elif command == "history":
                if not messages:
                    print("📜 No chat history yet.")
                else:
                    print("\n📜 Chat History:")
                    print("-" * 50)
                    for msg in messages:
                        if isinstance(msg, HumanMessage):
                            print(f"You: {msg.content}\n")
                        else:
                            print(f"Bot: {msg.content}\n")
                    print("-" * 50)

            elif command == "reset":
                messages = []
                print("✅ Chat history reset.")

            elif command == "chat":
                if vsm.get_chunk_count() == 0:
                    print("❌ Please add documents first using the 'add' command.")
                    continue

                ensure_chain_ready()
                print("💬 Chat mode. Type your questions (type 'back' to exit chat mode)")
                while True:
                    question = input("You: ").strip()
                    if question.lower() == "back":
                        break
                    if not question:
                        continue

                    print("🤔 Thinking...")
                    result = rag_chain.invoke({"input": question, "chat_history": messages})
                    answer = result["answer"]

                    messages.append(HumanMessage(content=question))
                    messages.append(AIMessage(content=answer))

                    print(f"\n🤖 Bot: {answer}\n")

            else:
                # treat as question
                if vsm.get_chunk_count() == 0:
                    print("❌ Please add documents first using the 'add' command.")
                    continue

                ensure_chain_ready()
                print("🤔 Thinking...")
                result = rag_chain.invoke({"input": user_input, "chat_history": messages})
                answer = result["answer"]

                messages.append(HumanMessage(content=user_input))
                messages.append(AIMessage(content=answer))

                print(f"\n🤖 Bot: {answer}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()