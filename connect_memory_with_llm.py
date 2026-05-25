# import os
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# # Step 1: Setup LLM (Mistral 7B Instruct model with huggingface)

# HF_TOKEN=os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         task="conversational",
#         temperature=0.5,
#         max_new_tokens=512,
#         huggingfacehub_api_token=HF_TOKEN
#     )
#     return llm
  
# # Step 2: Connect LLM with FAISS vector database (created in create_memory_for_llm.py)

# # this is a custom prompt template for Q&A , now this will be used in the chain ,this prompt will be passed to the LLM and the LLM will generate response based on this prompt.
# # Context will be the relevant chunks retrieved from the FAISS vector db based on the user query and question will be the user query.
# DB_FAISS_PATH = "vectorstore/db_faiss"
# CUSTOM_PROMPT_TEMPLATE = """                                                                     
# Use the pieces of information provided in the context to answer user's question.
# If you dont know the answer, just say that you dont know, dont try to make up an answer. 
# Dont provide anything out of the given context

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk please.
# """

# def set_custom_prompt_template(CUSTOM_PROMPT_TEMPLATE):
#     prompt = PromptTemplate(
#         template=CUSTOM_PROMPT_TEMPLATE,
#         input_variables=["context", "question"]
#     )
#     return prompt

# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)   # Load the FAISS vector db locally dangerous_deserialization= True is for safety and security by langchain.

# # Step 3: Create Chain for Q&A
# qa_chain = RetrievalQA.from_chain_type(
#    llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k':3}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt':set_custom_prompt_template(CUSTOM_PROMPT_TEMPLATE)}
# )

# # Now invoke with a single query
# user_query=input("Write Query Here: ")
# response=qa_chain.invoke({'query': user_query})
# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])


import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

# Step 1: Setup Groq LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # Change to any supported Groq model


llm = ChatGroq(
    model=GROQ_MODEL_NAME,
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)


# Step 2: Connect LLM with FAISS and Create chain

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 3: Build RAG chain
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Document combiner chain (stuff documents into prompt)
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

# Retrieval chain (retriever + doc combiner)
rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)



# Now invoke with a single query
user_query=input("Write Query Here: ")
response=rag_chain.invoke({'input': user_query})
print("RESULT: ", response["answer"])
print("\nSOURCE DOCUMENTS:")
for doc in response["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")