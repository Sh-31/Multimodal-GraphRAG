import os 
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph.KnowledgeGraph import KnowledgeGraph
from llm.factory import LLMFactory
from dotenv import load_dotenv
from graph.qdrant_db import QdrantVectorDB

load_dotenv()

def naive_query(query):
    print("Naive Query:")
    llm = LLMFactory.get_llm()
    knowledge_graph = KnowledgeGraph(llm=llm)
    answer, result = knowledge_graph.naive_query(query)
    print("\n")
    print(f"Answer: {answer}\n\n")
    print(f"Result: {result}")
    return answer, result

def graph_query(query):
    print("Query Aware Agent:\n")
    llm = LLMFactory.get_llm()
    knowledge_graph = KnowledgeGraph(llm=llm)
    answer, result = knowledge_graph.query(query)
    print("\n")
    print(f"Answer: {answer}\n\n")
    print(f"Result: {result}")
    return answer, result

if __name__ == "__main__":

       qurey = "Find all timesteps and episode name did Joey hold mug?"
       answer, result = graph_query(qurey)

    # # qurey = "Find all timesteps did any character kiss? and name of episode?"
    # # answer, result = graph_query(qurey)

    # # qurey = "What kinds of food did Joey eat across these episodes?"
    # # answer, result = graph_query(qurey)

    # qurey = "Who smoked that most?"
    # answer, result = graph_query(qurey)

