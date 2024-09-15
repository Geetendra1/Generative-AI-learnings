from langchain_openai import ChatOpenAI
from functools import lru_cache


@lru_cache(maxsize=2)
def create_llm(model="gpt-3.5-turbo-0125", temperature=0):
    return ChatOpenAI(model=model, temperature=temperature)