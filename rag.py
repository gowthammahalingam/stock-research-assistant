from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = OllamaLLM(model="llama3")

query = input("Ask a question: ")

docs = db.similarity_search(query, k=2)

context = "\n".join([doc.page_content for doc in docs])

prompt = f"""
Answer using only the context below.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response)