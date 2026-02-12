import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

class RAGengine:
    def __init__(self,pdf_paths,api_key):
        self.api_key = api_key

        self.docs = []
        for path in pdf_paths:
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                self.docs.extend(loader.load())
            else:
                print(f"Arquivo não encontrado: {path}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        self.split = self.text_splitter.split_documents(self.docs)

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model='models/gemini-embedding-001',
            google_api_key=self.api_key
        )

        self.vector_store = FAISS.from_documents(
            documents=self.split, embedding=self.embeddings
            )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k":3})
        print("RAG Engine inicializado com sucesso.")

    def buscar_contexto(self,query):
        docs = self.retriever.invoke(query)

        return "\n\n".join([doc.page_content for doc in docs])