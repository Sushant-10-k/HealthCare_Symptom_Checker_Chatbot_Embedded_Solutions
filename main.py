import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ── Constants ────────────────────────────────────────────────────────────────
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# ── App-level state (loaded once at startup) ─────────────────────────────────
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store and build the RAG chain once at startup."""
    global rag_chain

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    # 1. Embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. FAISS vector store
    db = FAISS.load_local(
        DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )

    # 3. Groq LLM
    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0.5,
        max_tokens=512,
        api_key=groq_api_key,
    )

    # 4. RAG chain  (same pattern as your connect_memory_with_llm.py)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(
        db.as_retriever(search_kwargs={"k": 3}), combine_docs_chain
    )

    print("✅ RAG chain ready.")
    yield  # app runs here

    # Cleanup (nothing needed for these objects)
    rag_chain = None


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Healthcare Symptom Checker API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response schemas ────────────────────────────────────────────────
class SymptomRequest(BaseModel):
    symptoms: list[str]


class SymptomResponse(BaseModel):
    input_symptoms: list[str]
    answer: str
    source_excerpts: list[str]  # short snippets from retrieved docs


class ChatRequest(BaseModel):
    query: str  # free-text question, e.g. from a chat UI


class ChatResponse(BaseModel):
    answer: str
    source_excerpts: list[str]


# ── Helper ────────────────────────────────────────────────────────────────────
def _run_rag(query: str) -> tuple[str, list[str]]:
    """
    Invoke the RAG chain and return (answer, source_excerpts).
    Raises HTTPException(503) if the chain is not yet initialised.
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialised yet.")

    response = rag_chain.invoke({"input": query})
    answer = response.get("answer", "I could not find an answer in the knowledge base.")

    # Pull short excerpts from the retrieved source documents
    source_excerpts = [
        doc.page_content[:300] for doc in response.get("context", [])
    ]
    return answer, source_excerpts


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def interface():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "API running", "rag_ready": rag_chain is not None}


@app.post("/predict", response_model=SymptomResponse)
def predict(data: SymptomRequest):
    """
    Accepts a list of symptoms, builds a natural-language query,
    and returns the RAG answer plus the source excerpts used.

    Example body:
        { "symptoms": ["fever", "headache", "stiff neck"] }
    """
    if not data.symptoms:
        raise HTTPException(status_code=422, detail="symptoms list must not be empty.")

    # Turn the symptom list into a plain-English query for the RAG chain
    symptoms_str = ", ".join(data.symptoms)
    query = (
        f"A patient is experiencing the following symptoms: {symptoms_str}. "
        "What could be the possible condition or disease? "
        "What are the recommended next steps or treatments?"
    )

    answer, source_excerpts = _run_rag(query)

    return SymptomResponse(
        input_symptoms=data.symptoms,
        answer=answer,
        source_excerpts=source_excerpts,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):
    """
    Free-text chat endpoint — mirrors what medibot.py does in Streamlit.

    Example body:
        { "query": "What are the symptoms of dengue fever?" }
    """
    if not data.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty.")

    answer, source_excerpts = _run_rag(data.query)

    return ChatResponse(answer=answer, source_excerpts=source_excerpts)