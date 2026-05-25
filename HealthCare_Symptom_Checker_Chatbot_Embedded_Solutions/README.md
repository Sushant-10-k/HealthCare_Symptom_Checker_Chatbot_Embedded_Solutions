# Healthcare Symptom Checker Chatbot 🏥🤖

An intelligent healthcare symptom checker chatbot built using **Streamlit**, **LangChain**, **FAISS**, and **Groq LLM**. This application uses Retrieval-Augmented Generation (RAG) to provide accurate healthcare information based on medical documents.

## 🌟 Features

- **Interactive Chat Interface**: Built with Streamlit for a user-friendly experience
- **RAG Implementation**: Combines document retrieval with Large Language Model responses
- **FAISS Vector Database**: Fast similarity search for medical information
- **Groq LLM Integration**: Uses Llama-3.1-8b-instant model for natural language processing
- **PDF Document Processing**: Automatically processes medical PDFs to create knowledge base
- **Session Memory**: Maintains conversation history during the session

## 🏗️ Architecture

```
📁 Project Structure
├── medibot.py                    # Main Streamlit application
├── create_memory_for_llm.py      # Script to create FAISS vector store from PDFs
├── connect_memory_with_llm.py    # Console-based Q&A interface
├── data/                         # Directory for PDF medical documents
├── vectorstore/                  # FAISS vector database (generated)
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration (UV/Poetry)
├── .env                         # Environment variables (not tracked)
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key ([Get one here](https://console.groq.com/))
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/HealthCare_Symptom_Checker_Chatbot_Embedded_Solutions.git
   cd HealthCare_Symptom_Checker_Chatbot_Embedded_Solutions
   ```

2. **Set up virtual environment**
   
   Using UV (recommended):
   ```bash
   uv venv
   uv pip sync requirements.txt
   ```
   
   Or using traditional pip:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Prepare your medical documents**
   
   - Place your PDF medical documents in the `data/` directory
   - Run the memory creation script:
   ```bash
   python create_memory_for_llm.py
   ```
   This will process the PDFs and create the FAISS vector database.

5. **Run the application**
   ```bash
   streamlit run medibot.py
   ```

## 💻 Usage

### Web Interface (Recommended)

1. Start the Streamlit app: `streamlit run medibot.py`
2. Open your browser to `http://localhost:8501`
3. Type your health-related questions in the chat interface
4. Get AI-powered responses based on your medical document knowledge base

### Console Interface

For command-line usage:
```bash
python connect_memory_with_llm.py
```

## 🔧 Configuration

### Supported Models

The application uses Groq's Llama-3.1-8b-instant model by default. You can change this in the code:

```python
GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # or other supported models
```

### Embedding Model

The project uses `sentence-transformers/all-MiniLM-L6-v2` for creating document embeddings. This provides a good balance of performance and quality.

### Document Processing Parameters

- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Retrieval K**: 3 most similar documents

## 📋 Dependencies

### Core Libraries
- `streamlit` - Web interface
- `langchain` - LLM framework
- `langchain-groq` - Groq integration
- `langchain-huggingface` - HuggingFace embeddings
- `langchain-community` - Community extensions
- `faiss-cpu` - Vector similarity search
- `python-dotenv` - Environment variable management
- `pypdf` - PDF processing

### Full requirements
See `requirements.txt` for the complete list of dependencies.

## ⚠️ Important Notes

1. **Medical Disclaimer**: This chatbot is for informational purposes only and should not replace professional medical advice.

2. **API Keys**: Keep your Groq API key secure and never commit it to version control.

3. **Document Quality**: The quality of responses depends on the medical documents you provide in the `data/` directory.

4. **Vector Store**: The `vectorstore/` directory contains processed embeddings and can be large. It's excluded from git by default.

## 🛠️ Development

### Adding New Documents

1. Place new PDF files in the `data/` directory
2. Run `python create_memory_for_llm.py` to update the vector database
3. Restart the Streamlit application

### Customizing the Prompt

The application uses LangChain's built-in retrieval-qa-chat prompt. You can customize it by modifying the prompt template in the code.

### Environment Setup for Development

The project uses UV for dependency management. Key files:
- `pyproject.toml` - Project metadata and dependencies
- `uv.lock` - Lock file for reproducible installs
- `.python-version` - Python version specification

## 🐛 Troubleshooting

### Common Issues

1. **"Failed to load the vector store"**
   - Ensure you've run `create_memory_for_llm.py` first
   - Check that PDF files exist in the `data/` directory

2. **Groq API Errors**
   - Verify your API key is correct in the `.env` file
   - Check your Groq API quota and rate limits

3. **Memory/Performance Issues**
   - Consider reducing the chunk size or number of retrieved documents
   - Monitor RAM usage, especially with large document collections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain** - For the excellent LLM framework
- **Groq** - For fast LLM inference
- **Streamlit** - For the intuitive web framework
- **FAISS** - For efficient vector similarity search
- **HuggingFace** - For the embedding models

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information about your problem

---

**⚕️ Remember**: This tool is designed to assist with health information queries but should never replace professional medical consultation.