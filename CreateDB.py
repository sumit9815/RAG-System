# code of CreateDB.py
#todo 1.loading the document(pdf)
from langchain_community.document_loaders import PyPDFLoader
#todo 2.Create Chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
#todo 3.Create the embeddings
from langchain_huggingface import HuggingFaceEmbeddings
#todo 4.Store into the Database (Chroma)
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

loader=PyPDFLoader("deeplearning.pdf")
docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
) 
chunks=splitter.split_documents(docs)

embeddings_model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2"
)
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="Chroma-VDB"
) 