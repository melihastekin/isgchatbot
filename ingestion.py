from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

import os

load_dotenv()

pdf_folder = "C:\\Users\\mkast\\Downloads\\tez\\PROJE\\dokumanlar"

docs = []
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
        docs.extend(loader.load())  # Sayfalara ayrılmış dökümanları listeye ekle

print(f"Yüklenen {len(docs)} döküman sayfası")

# 📌 Metinleri bölme işlemi (Chunking)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
    persist_directory="./.chroma",
)

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()

print("Vektör veritabanı başarıyla oluşturuldu!")
