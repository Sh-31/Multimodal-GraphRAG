import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

class Neo4jClient:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri 
        self.username = user 
        self.password = password 

        try:
            self.graph = Neo4jGraph(
                url=self.uri, 
                username=self.username, 
                password=self.password,
            )

        except Exception as e:
            raise ConnectionError(f"Could not connect to Neo4j database at {self.uri}. Error: {e}")

    def add_graph_documents(self, graph_documents, include_source=True):
        self.graph.add_graph_documents(
            graph_documents, 
            include_source=include_source
        )

    def query(self, query, params=None):
        return self.graph.query(query, params)
    
    def refresh_schema(self):
        self.graph.refresh_schema()

    def verify_connection(self):
        try:
            self.graph.query("RETURN 1") # A simple query to verify connection
            return True
        except Exception:
            return False
    
    def get_schema(self):
        return self.graph.schema

    def get_structure_schema(self):
        return self.graph.structured_schema

    def close(self):
        """Close the Neo4j connection"""
        if hasattr(self, 'graph') and hasattr(self.graph, '_driver'):
            self.graph._driver.close()

if __name__ == "__main__":
    load_dotenv()
    neo4j_client = Neo4jClient(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    print(
        neo4j_client.graph.schema
    )
    print("Neo4j Client initialized")
    print("Neo4j Client connection verified")
    print("Neo4j Client schema:", neo4j_client.get_schema())
    print("Neo4j Client structure schema:", neo4j_client.get_structure_schema())