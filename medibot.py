import os
import sys
from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
RETRIEVAL_K = 3

MEDICAL_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful healthcare information assistant. Use only the "
            "medical encyclopedia context below to answer. If the context does "
            "not contain enough information, say what is missing and recommend "
            "speaking with a qualified clinician. Do not diagnose the user.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{input}"),
    ]
)


@lru_cache(maxsize=1)
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(str(DB_FAISS_PATH), embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template: str) -> PromptTemplate:
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


@lru_cache(maxsize=1)
def get_rag_chain():
    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise RuntimeError("Failed to load the vector store")

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0.5, max_tokens=512, api_key=GROQ_API_KEY)

    combine_docs_chain = create_stuff_documents_chain(llm, MEDICAL_QA_PROMPT)
    return create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        combine_docs_chain,
    )


def format_source(doc) -> dict[str, str | int | None]:
    source = doc.metadata.get("source")
    page = doc.metadata.get("page")

    return {
        "source": Path(source).name if source else "Medical encyclopedia",
        "page": page + 1 if isinstance(page, int) else page,
        "excerpt": doc.page_content[:320].strip(),
    }


def run_medical_query(prompt: str) -> dict[str, object]:
    try:
        rag_chain = get_rag_chain()
        response = rag_chain.invoke({"input": prompt})
        docs = response.get("context", [])

        return {
            "answer": response.get("answer", "").strip(),
            "sources": [format_source(doc) for doc in docs],
        }
    except RuntimeError as error:
        if "GROQ_API_KEY" not in str(error):
            raise

        docs = get_vectorstore().similarity_search(prompt, k=RETRIEVAL_K)
        excerpts = "\n\n".join(doc.page_content.strip()[:500] for doc in docs)
        return {
            "answer": (
                "I found related encyclopedia entries, but GROQ_API_KEY is not configured, "
                "so I cannot generate a full response.\n\n"
                f"{excerpts}"
            ).strip(),
            "sources": [format_source(doc) for doc in docs],
        }


def run_query_once(prompt: str) -> str:
    return str(run_medical_query(prompt).get("answer", ""))


def main():
    print("Healthcare Symptom Checker Chatbot (CLI) - Groq LLM + FAISS")
    print("Type a question and press Enter. Type 'exit' or Ctrl+C to quit." )

    messages = []
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            messages.append({"role": "user", "content": user_input})

            try:
                answer = run_query_once(user_input)
                print("Assistant:")
                print(answer)
                messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()
