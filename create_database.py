# Load pdf
# Split the pdf into chunks
# Create Embeddings
# Store the embeddings in a vector database

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

data = PyPDFLoader("document/deep_learning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store =  Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    persist_directory= "chroma_db"
)