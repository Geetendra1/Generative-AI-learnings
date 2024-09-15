from typing import TypedDict, List
import time
import logging
import concurrent.futures
from langchain.schema import Document

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import START, END, StateGraph

from retriever import create_retriever
from functools import lru_cache
from llm import create_llm
from prompts import rag_prompt, grading_prompt
web_search_tool = TavilySearchResults()


retriever = create_retriever()
llm = create_llm()
rag_chain = rag_prompt | llm | StrOutputParser()
retrieval_grader = grading_prompt | llm | JsonOutputParser()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    start_time = time.time()

    question = state["question"]
    documents = retriever.invoke(question)
    state["documents"] = documents
    state["steps"].append("retrieve_documents")
    logger.info(f"Retrieve time: {time.time() - start_time:.2f} seconds")
    return state


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    start_time = time.time()

    question = state["question"]
    documents = state["documents"]
    docs_content = "\n".join(doc.page_content for doc in documents[:3])  # Limit to top 3 documents
    # RAG generation
    state["generation"] = rag_chain.invoke({"documents": docs_content, "question": question})
    state["steps"].append("generate_answer")
    logger.info(f"Generate time: {time.time() - start_time:.2f} seconds")
    return state

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    start_time = time.time()

    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    search_needed = False
    batch_size = 20


    def process_batch(batch):
        batch_inputs = [{"question": question, "document": d.page_content} for d in batch]
        return retrieval_grader.batch(batch_inputs)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, documents[i:i+batch_size]) for i in range(0, len(documents), batch_size)]

        for future in concurrent.futures.as_completed(futures):
            batch_scores = future.result()
            for doc, score in zip(documents, batch_scores):
                if score["score"] == "yes":
                    filtered_docs.append(doc)
                else:
                    search_needed = True

            if len(filtered_docs) >= 5: # Top 5 docs
                break

    state["documents"] = filtered_docs[:5]  # Limit to top 5 relevant documents
    state["search"] = "Yes" if search_needed and len(filtered_docs) < 3 else "No"
    state["steps"].append("grade_document_retrieval")
    logger.info(f"Grade documents time: {time.time() - start_time:.2f} seconds")
    return state




@lru_cache(maxsize=100)
def cached_web_search(question: str):
    logger.info("Web search cache miss")
    return web_search_tool.invoke({"query": question})


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    start_time = time.time()
    web_results = cached_web_search(state["question"])
    state["documents"].extend([Document(page_content=d["content"], metadata={"url": d["url"]}) for d in web_results])
    state["steps"].append("web_search")
    logger.info(f"Web search time: {time.time() - start_time:.2f} seconds")
    return state


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    return "search" if state.get("search") == "Yes" and len(state["documents"]) < 3 else "generate"


def add_nodes(workflow: StateGraph):
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("web_search", web_search)

def build_graph(workflow: StateGraph):
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)


def create_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)
    add_nodes(workflow=workflow)
    build_graph(workflow=workflow)
    return workflow.compile()


rag_workflow = create_workflow()

def process_query(question: str) -> dict:
    initial_state = GraphState(question=question, steps=[])
    result = rag_workflow.invoke(initial_state)
    return result