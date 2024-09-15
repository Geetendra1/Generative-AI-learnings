from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore



def create_retriever(index_name="gen-ai"):
    vector_store = PineconeVectorStore(index_name=index_name, embedding=OpenAIEmbeddings(model="text-embedding-3-large"))
    retriever = vector_store.as_retriever()
    return retriever