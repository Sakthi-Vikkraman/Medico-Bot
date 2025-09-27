from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

#Extract text from pdf
def load_pdf(file_path):
    loader = DirectoryLoader(
        file_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        )
    documents = loader.load()
    return documents

#remove unwated data in the documents
def clean_documents(documents: List[Document]) -> List[Document]:
    cleaned_docs = []
    for doc in documents:
        src = doc.metadata.get("source")
        cleaned_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return cleaned_docs

# split the documents into smaller chunks
def split_documents(cleaned_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )
    text_chunks = text_splitter.split_documents(cleaned_docs)
    return text_chunks

#create embeddings from the text chunks
def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings