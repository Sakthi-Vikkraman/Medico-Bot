from dotenv import load_dotenv
import os
from src.helper import load_pdf, clean_documents, split_documents, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

# load pinecone db access keys from .env file
load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


extracted_data = load_pdf("data")
cleaned_data = clean_documents(extracted_data)
texts_chunks = split_documents(cleaned_data)

embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
PC = Pinecone(api_key=pinecone_api_key)


index_name = "medico-bot"
if not PC.has_index(index_name):
    PC.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = PC.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    index_name=index_name,
    embedding=embeddings, 
)