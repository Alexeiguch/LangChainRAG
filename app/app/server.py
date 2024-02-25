#!/usr/bin/env python
"""Example LangChain server exposes a retriever."""
from fastapi import FastAPI
from langchain_core.runnables import RunnableLambda
from langchain.document_loaders.url_playwright import PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langserve import add_routes

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.url_playwright import PlaywrightURLLoader
from langchain.text_splitter import TextSplitter
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


class RAGRetriever:
    def __init__(self, splitter: TextSplitter, loader: BaseLoader, llama_model = "llama2") -> None:
        self.loader = loader
        self.splitter = splitter

        self.llm = Ollama(model=llama_model)

    def load_context(self) -> List[Document]:
        content = self.loader.load()
        documents = self.splitter.split_documents(content)
        return documents
    
    def vectorize(self, documents: List[Document]) -> FAISS:
        embeddings = OllamaEmbeddings()
        vectors = FAISS.from_documents(documents, embeddings)
        return vectors

    def create_retrieval_chain(self, vectors: FAISS, prompt: str):
        prompt_temp = ChatPromptTemplate.from_template(prompt)

        document_chain = create_stuff_documents_chain(self.llm, prompt_temp)

        retriever = vectors.as_retriever()
        self.retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    def invoke(self, input: str):
        assert self.retrieval_chain != None
        res = self.retrieval_chain.invoke({"input": input})
        print(res)
        return res
        

if __name__ == "__main__":
    playwright = PlaywrightURLLoader([
        "https://www.promptior.ai/", 
        "https://www.promptior.ai/about/"], headless=False)
    text_splitter = RecursiveCharacterTextSplitter()
    model = RAGRetriever(text_splitter, playwright, llama_model="llama2")

    docs = model.load_context()
    vectors = model.vectorize(docs)

    prompt = """Answer the following question based only on the provided context:

            <context>
            {context}
            </context>

            Question: {input}"""

    model.create_retrieval_chain(vectors, prompt)

    response = model.invoke("Qué servicios ofrece Promptior?")
    print(response["answer"])

    response2 = model.invoke("Cuándo fue fundada la empresa Promptior?")
    print(response2["answer"])

app = FastAPI(
    title="LangChain Server - LLama2 Retriever",
    version="1.0",
    description="Run up a simple api server using Langchain's Runnable interfaces that instantiates a Llama2 Retriever Model to answer questions about Promptior.",
)

playwright = PlaywrightURLLoader([
    "https://www.promptior.ai/", 
    "https://www.promptior.ai/about/"], headless=False)

text_splitter = RecursiveCharacterTextSplitter()

model = RAGRetriever(text_splitter, playwright)
docs = model.load_context()
vectors = model.vectorize(docs)

prompt = """Answer the following question based only on the provided context:

            <context>
            {context}
            </context>

            Question: {input}"""
model.create_retrieval_chain(vectors, prompt)

def invoke(input: str):
    return model.invoke(input)

retriever = RunnableLambda(invoke)

# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
add_routes(app, retriever, path="/llama")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)