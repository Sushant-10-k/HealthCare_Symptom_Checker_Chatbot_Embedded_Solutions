import os
import sys
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

load_dotenv()


DB_FAISS_PATH = "vectorstore/db_faiss"


@lru_cache(maxsize=1)
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template: str) -> PromptTemplate:
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


def run_query_once(prompt: str) -> str:
    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise RuntimeError("Failed to load the vector store")

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    GROQ_MODEL_NAME = "llama-3.1-8b-instant"
    llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0.5, max_tokens=512, api_key=GROQ_API_KEY)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 3}), combine_docs_chain)

    response = rag_chain.invoke({"input": prompt})
    return response.get("answer", "")


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