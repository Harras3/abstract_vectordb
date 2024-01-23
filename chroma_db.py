

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import numpy as np
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from helper_functions import load_pdf




class chroma_db:
    def __init__ (self,collection_name,embedding_function):
        self.chroma_cliet = chromadb.Client()
        self.embedding_function = SentenceTransformerEmbeddingFunction()
        self.chroma_collection = chroma_cliet.create_collection(name=collection_name, embedding_function=embedding_function)
        return chroma_collection
    def insert_in_db(self,chunks):
        ids = [str(i) for i in range(len(chunks))]
        chroma_collection.add(ids=ids, documents=chunks)
        return
    def query(self,query,n,include=['documents', 'embeddings']):
        results = chroma_collection.query(query_texts=query, n_results=n, include=include)
        return results



        






