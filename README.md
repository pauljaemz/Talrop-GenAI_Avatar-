# 🤖 RAG Chatbot - Terminal Edition

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, and HuggingFace that allows you to add PDF and TXT files and ask questions about their content via the terminal.

## Features

- **Document Upload**: Support for PDF and TXT files
- **Persistent Storage**: Uses ChromaDB to store document embeddings locally
- **Conversational Memory**: Maintains chat history for context-aware responses
- **Multiple File Support**: Add and process multiple documents at once
- **Free Models**: Uses HuggingFace's free inference API
- **Terminal Interface**: Simple command-line interface

## Installation

1. Clone the repository or navigate to the project directory

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your HuggingFace API token:
   - Copy `.env.example` to `.env`
   - Add your HuggingFace API token:
     ```
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
     ```
   - Get your token from: https://huggingface.co/settings/tokens

## Usage

1. Run the chatbot:
   ```bash
   python main.py
   ```

2. **Available Commands**:
   | Command | Description |
   |---------|-------------|
   | `add <file1> [file2] ...` | Add PDF or TXT files to the database |
   | `chat` | Enter chat mode |
   | `stats` | Show database statistics |
   | `clear` | Clear the database |
   | `history` | Show chat history |
   | `reset` | Reset chat history |
   | `help` | Show help message |
   | `exit` / `quit` | Exit the chatbot |

3. **Example Session**:
   ```
   You: add document.pdf notes.txt
   🔄 Processing: document.pdf
   ✅ Successfully processed: document.pdf (15 chunks)
   🔄 Processing: notes.txt
   ✅ Successfully processed: notes.txt (8 chunks)
   ✅ Added 23 chunks to database.

   You: What is the main topic of the document?
   🤔 Thinking...
   🤖 Bot: The document discusses...
   ```

## Project Structure

```
├── main.py              # Main terminal application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API keys)
├── .env.example         # Example environment file
├── chroma_db/           # ChromaDB persistent storage
└── README.md            # This file
```

## Configuration

You can modify these settings in `main.py`:

- `CHROMA_DB_PATH`: Directory for ChromaDB storage (default: `./chroma_db`)
- `CHUNK_SIZE`: Size of text chunks for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- LLM Model: Currently uses `mistralai/Mistral-7B-Instruct-v0.3` (can be changed to other HuggingFace models)
- Embeddings Model: Uses `sentence-transformers/all-MiniLM-L6-v2` (runs locally)

## Requirements

- Python 3.9+
- HuggingFace API token (free)
- Dependencies listed in `requirements.txt`

## How It Works

1. **Document Processing**: Documents are split into smaller chunks using RecursiveCharacterTextSplitter
2. **Embedding Generation**: HuggingFace sentence-transformers create embeddings locally for each chunk
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient retrieval
4. **Query Processing**: User questions are embedded and matched against stored documents
5. **Response Generation**: Relevant document chunks are used as context for the HuggingFace LLM to generate answers

## License

MIT License
