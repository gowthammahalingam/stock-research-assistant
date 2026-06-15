from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

files = [
    "data/apple.pdf",
    "data/microsoft.pdf",
    "data/nvidia.pdf"
]

documents = []

for file in files:
    print("Loading:", file)

    loader = PyPDFLoader(file)

    documents.extend(loader.load())

print("Pages Loaded:", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Chunks Created:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(
    chunks,
    embeddings
)

db.save_local("financial_vector_db")

print("Financial Vector DB Created!")