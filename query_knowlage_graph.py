import os 
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph.KnowledgeGraph import KnowledgeGraph
from llm.factory import LLMFactory
from dotenv import load_dotenv
from graph.qdrant_db import QdrantVectorDB

load_dotenv()

def naive_query(query):
    llm = LLMFactory.get_llm()
    knowledge_graph = KnowledgeGraph(llm=llm)
    answer, result = knowledge_graph.naive_query(query)
    return answer, result

def graph_query(query):
    llm = LLMFactory.get_llm()
    knowledge_graph = KnowledgeGraph(llm=llm)
    answer, result = knowledge_graph.query(query)
    return answer, result

if __name__ == "__main__":

    query = "What timesteps rachel appeared in wedding gown?"
    print("Query Aware Agent:\n")
    answer, result = graph_query(query)
    print(answer)
    print(result)