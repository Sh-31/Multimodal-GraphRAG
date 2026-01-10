import os
import sys
import uuid
import json
import time
import base64
from google import genai
from google.genai import types
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from llm.factory import LLMFactory
from preprocessor.helper import chunk_text, get_logger
from .qdrant_db import QdrantVectorDB
from .neo4j_client import Neo4jClient
from .prompts import (Answer_Validity_Prompt, Summarize_Group_Video_Chunks,
                     CYPHER_GENERATION_RESTRICTED_PROMPT, USER_QUERY_REWRITE_PROMPT)

class KnowledgeGraph:
    def __init__(self, llm, collection_name="multimodal_rag"):
        self.llm = llm
        # Initialize Text to Graph Agent
        self.document_to_graph_agent = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=[], 
            allowed_relationships=[], 
        )
        
        # Neo4j Client
        self.neo4j = Neo4jClient(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
        )
        
        # Qdrant Client
        self.vector_db = QdrantVectorDB(collection_name=collection_name)
        self.logger = get_logger("KnowledgeGraph")


    def process_chunk(self, text: str, metadata: dict):
        """
        Main entry point: Extracts graph data from text and saves to Neo4j.
        """
        documents = [Document(page_content=text, metadata=metadata)]
        graph_documents = self.document_to_graph_agent.convert_to_graph_documents(documents)
        
        if not graph_documents:
            return

        # Add time_steps and video_name to relationships as properties
        for graph_doc in graph_documents:
            for rel in graph_doc.relationships:
                rel.properties["time_steps"] = metadata.get("time_steps", "N/A")
                rel.properties["video_name"] = metadata.get("video_name", "N/A")
                if "chunk_id" in metadata:
                    rel.properties["chunk_id"] = metadata["chunk_id"]

        self.logger.info(f"Generated {len(graph_documents)} graph documents.")
        nodes_count = sum(len(d.nodes) for d in graph_documents)
        rels_count = sum(len(d.relationships) for d in graph_documents)
        self.logger.info(f"Extracted {nodes_count} nodes and {rels_count} relationships.")

        self._save_to_neo4j(graph_documents)

    def process_chunks_video(self, video_chunk_captions: dict, group:bool = True,  summary_group: bool= True, num_of_group:int = 5):
        if not group and (summary_group or num_of_group > 1):
            raise ValueError("Grouping is required when summary_group is True or num_of_group > 1")
 
        if not group: 
           # Process each chunk independently only add caption to graph (no subtitles)
           # AND index to Qdrant
           chunks = []
           metadatas = []
           for chunk_id in video_chunk_captions:
               caption = video_chunk_captions[chunk_id]["caption"]
               start_chunk = video_chunk_captions[chunk_id]["start_chunk"]
               end_chunk = video_chunk_captions[chunk_id]["end_chunk"]
               time_steps = f"{start_chunk} - {end_chunk}"

               metadata = {
                   "video_name": video_chunk_captions[chunk_id]["episode"],
                   "chunk_id": chunk_id,
                   "time_steps": time_steps
               }

               # Chunk text before indexing
               caption_chunks = chunk_text(caption)
               for k, cap in enumerate(caption_chunks):
                   chunks.append(cap)
                   metadata["sub_chunk_index"] = k
                   self.process_chunk(cap, metadata)
                   metadatas.append(metadata)
           
           log_index_vector_db = self.vector_db.index_chunks(chunks, metadatas=metadatas)
           self.logger.info(log_index_vector_db)
           return 
           
        chunks = []
        metadatas = []

        # Merge consecutive chunks
        for chunk_id in video_chunk_captions:
            chunks.append(video_chunk_captions[chunk_id])
            
        merged_chunks = []
        for i in range(0, len(chunks), num_of_group):
            merged_chunks.append(chunks[i:i + num_of_group])
             
        merged_chunks_jsons = []
        for i, merged_chunk in enumerate(merged_chunks):
            video_name = merged_chunk[0]['episode']
            self.logger.info(f"Processing video: {video_name} - group {i+1}/{len(merged_chunks)}")
            merged_chunks_capution = f"Episode name: {video_name}\n"
            
            for j, chunk in enumerate(merged_chunk):
                merged_chunks_capution += f"Chunk {j+1}:\n, time_steps: {chunk['start_chunk']} - {chunk['end_chunk']}\n{chunk['caption']}\n"
            
            try:
                result = self._generate_summary(merged_chunks_capution)
                merged_chunks_summary = result.content
            
            except Exception as e:
                self.logger.error(f"Error generating summary: {e}, using temp_summary")
                raise Exception("Error generating summary")
          
            # Index Summary to Vector DB
            summary_chunks = chunk_text(merged_chunks_summary)
            summary_metadatas = []
            for k in range(len(summary_chunks)):
                summary_metadata={
                        "video_name": video_name, 
                        "type": "summary",
                        "time_steps": f"{merged_chunk[0]['start_chunk']} - {merged_chunk[-1]['end_chunk']}",
                        "group_index": i,
                        "sub_chunk_index": k
                }
                summary_metadatas.append(summary_metadata)
                self.process_chunk(
                    summary_chunks[k], 
                    summary_metadata
                )

            log_index_vector_db = self.vector_db.index_chunks(summary_chunks, metadatas=summary_metadatas)
            self.logger.info(log_index_vector_db)

            # Index Individual Chunks and subtitles to Vector DB
            group_chunks_captions = []
            group_chunks_subtitles = []
            group_chunks_caption_metadatas = []
            group_chunks_subtitle_metadatas = []
         
            for j, chunk in enumerate(merged_chunk):
                # Chunk captions
                caption_chunks = chunk_text(chunk['caption'])
                for k, cap in enumerate(caption_chunks):
                    group_chunks_captions.append(cap)
                    group_chunks_caption_metadatas.append({
                        "video_name": video_name, 
                        "type": "caption", 
                        "group_index": i, 
                        "chunk_index": j,
                        "sub_chunk_index": k,
                        "time_steps": f"{chunk['start_chunk']} - {chunk['end_chunk']}"
                    })
                
                # Chunk subtitles
                subtitle_chunks = chunk_text(chunk['subtitles'])
                for k, sub in enumerate(subtitle_chunks):
                    group_chunks_subtitles.append(sub)
                    group_chunks_subtitle_metadatas.append({
                        "video_name": video_name, 
                        "type": "subtitles", 
                        "group_index": i, 
                        "chunk_index": j,
                        "sub_chunk_index": k,
                        "time_steps": f"{chunk['start_chunk']} - {chunk['end_chunk']}"
                    })
            
            log_index_vector_db = self.vector_db.index_chunks(group_chunks_captions, metadatas=group_chunks_caption_metadatas)
            self.logger.info(log_index_vector_db)
            log_index_vector_db = self.vector_db.index_chunks(group_chunks_subtitles, metadatas=group_chunks_subtitle_metadatas)
            self.logger.info(log_index_vector_db)

            # dict for merged chunks with summary
            merged_chunks_json = {}
            for j, chunk in enumerate(merged_chunk):
                merged_chunks_json[f"chunk_{j+1}"] = chunk
         
            merged_chunks_json["summary"] = merged_chunks_summary
            merged_chunks_jsons.append(merged_chunks_json)
            
        return merged_chunks_jsons 

    def _generate_summary(self, text: str):
        llm = LLMFactory.get_llm()
        chain = Summarize_Group_Video_Chunks | llm
        result = chain.invoke({"text": text})
        return result

    def _save_to_neo4j(self, graph_documents):
        self.logger.info("Saving to Neo4j...")
        self.neo4j.add_graph_documents(graph_documents, include_source=True)

    def naive_query(self, query: str):
        self.logger.info(f"Querying: {query}")
        
        chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.neo4j.graph,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            verbose=True,
            cypher_prompt=CYPHER_GENERATION_RESTRICTED_PROMPT
        )

        result = chain.invoke({"query": query})
        answer = result["result"]
        
        return answer, result


    def query(self, query: str, limit: int = 3, num_candidates: int = 2):
        """
        Hybrid query: Graph -> Vector (Fallback) -> Graph Update (Learning)
        """
        self.logger.info(f"Querying: {query}")
        
        original_query = query
        schema = self.neo4j.graph.schema
        
        # Rewrite user query to multiple candidates
        rewrite_chain = USER_QUERY_REWRITE_PROMPT | self.llm | JsonOutputParser()
        rewritten_result = rewrite_chain.invoke({"question": query, "schema": schema, "num_candidates": num_candidates})

        if type(rewritten_result) == dict:
            candidate_queries = rewritten_result.get("candidate_queries", [query])
            refined_query = rewritten_result.get("refined_query", query)
        else:
            candidate_queries = [query]
            refined_query = query
         
        self.logger.info(f"Candidate Queries: {candidate_queries}")

        # Natural language query to Cypher language query
        restricted_cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.neo4j.graph,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            verbose=True,
            cypher_prompt=CYPHER_GENERATION_RESTRICTED_PROMPT
        )

        cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.neo4j.graph,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            verbose=True,
        )

        answer, result = None, None
        for i, cand_query in enumerate(candidate_queries):
            self.logger.info(f"Trying candidate {i+1}/{len(candidate_queries)}: {cand_query}")

            # Try with cypher chain first
            result = cypher_chain.invoke({"query": cand_query})
            answer = result["result"]
        
            if self._validate_answer(refined_query, answer):
                self.logger.info(f"Valid answer found with candidate {i+1}")
                return answer, result

            # Try with restricted cypher chain
            result = restricted_cypher_chain.invoke({"query": cand_query})
            answer = result["result"]
        
            if self._validate_answer(refined_query, answer):
                self.logger.info(f"Valid answer found with candidate {i+1}")
                return answer, result

        # If invalid, fall back to Vector DB        
        self.logger.info(f"Graph answer ('{answer}') insufficient/invalid, falling back to Video revisiting and Vector DB...")
        
        # Vector Search & RAG
        search_results = self.vector_db.search(refined_query, limit=limit)
        known_video_chunks = []
        context_parts = []
        for res in search_results:
            text = res.payload.get('text', '')
            time = res.payload.get('time_steps', 'Unknown Time')
            video = res.payload.get('video_name', 'Unknown Video')
            chunk_index = res.payload.get('chunk_index', 'Unknown Chunk')
            sub_chunk_index = res.payload.get('sub_chunk_index', 'Unknown Sub Chunk')
            context_parts.append(f"Video: {video}\nTime: {time}\nContent: {text}")

            if res.payload.get('video_name') and res.payload.get('chunk_index'):
                known_video_chunks.append([res.payload.get('video_name'), res.payload.get('chunk_index'),  res.payload])

        self.logger.info("Context retrieved from Vector DB:")    
        context = "\n---\n".join(context_parts)

        if not context:
            return "I don't know.", result
      
        # self.logger.info(context)
        self.logger.info("\n\n")
        self.logger.info("Revisiting video chunks...")
        for video_name, chunk_index, metadata in known_video_chunks:
            new_caption = self.revisit_video_chunk(refined_query, video_name, chunk_index)

            prompt = ChatPromptTemplate.from_template(
                "Answer the question based only on the following context. If you cannot answer the question, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}"
            )
            
            chain = prompt | self.llm
            response = chain.invoke({"context": new_caption, "question": refined_query})
            answer = response.content

            if self._validate_answer(refined_query, answer):
                result["context"] = new_caption
              
                # Extract generated Cypher
                generated_cypher = None
                if "intermediate_steps" in result:
                    for step in result["intermediate_steps"]:
                        if isinstance(step, dict) and "query" in step:
                            generated_cypher = step["query"]
                            break
        
                self._update_graph(refined_query, answer, new_caption, metadata=metadata, cypher=generated_cypher)

                return answer, result

        self.logger.info("Video chunks revisited failed. Fallback to RAG...")
        self.logger.info("\n\n")

        prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context. If you cannot answer the question, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}"
        )
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": refined_query})
        rag_answer = response.content

        # Validate RAG Answer & Update Graph
        if self._validate_answer(refined_query, rag_answer):
            self.logger.info("RAG answer is valid. Updating Knowledge Graph with new information...")
            
            # Use metadata from the first search result for backfilling
            primary_metadata = search_results[0].payload if search_results else {}
            
            # Extract generated Cypher if available
            generated_cypher = None
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if isinstance(step, dict) and "query" in step:
                        generated_cypher = step["query"]
                        break
            
            self._update_graph(refined_query, rag_answer, context, metadata=primary_metadata, cypher=generated_cypher)
            result["context"] = context
            result["result"] = rag_answer

            return rag_answer, result
        else:
            return "I don't know.", result

    def _validate_answer(self, question: str, answer: str) -> bool:
        """
        Uses an LLM agent to validate if the answer is useful.
        """
        chain = Answer_Validity_Prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "answer": answer})
        return result == "VALID"

    def _update_graph(self, question: str, answer: str, context: str, metadata: dict = None, cypher: str = None):
        """
        Extracts new knowledge from the (question, answer) pair and adds it to the graph.
        """
        # Include context in the text to extract so the transforme  r has enough info to link entities correctly
        text_to_extract = (
            f"Based on this cypher query and that user question and answer and context, extract entities and relationships to update the knowledge graph to answer this question.\n\n"
            f"Cypher Query: {cypher}\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Context: {context}"
        )

        doc_metadata = {"source": "rag"}
        if metadata:
            doc_metadata.update(metadata)
        
        documents = [Document(page_content=text_to_extract, metadata=doc_metadata)]
        graph_documents = self.document_to_graph_agent.convert_to_graph_documents(documents)
        
        if graph_documents:
            nodes_count = sum(len(d.nodes) for d in graph_documents)
            rels_count = sum(len(d.relationships) for d in graph_documents)
            self.logger.info(f"Found {nodes_count} nodes and {rels_count} relationships.")
            self.neo4j.add_graph_documents(graph_documents, include_source=True)
            
            # Refresh schema so the Cypher agent can see the new labels/properties
            self.neo4j.refresh_schema()
            self.logger.info("Graph schema refreshed.")
        else:
            self.logger.info("No graph data extracted from answer.")

    def _encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded_string}"

    def revisit_video_chunk(self, query: str, video_name: str, chunk_index: int):
        main_video_root_path = Path("/home/sh-31/repo/Multimodal-GraphRAG/data/episodes_splitted/") 
        meta_path = Path("/home/sh-31/repo/Multimodal-GraphRAG/data/meta/") 
        chunk_id = str(chunk_index + 1).zfill(5)
        video_chunk_path = Path(f"{video_name}/chunk_{chunk_id}/{chunk_id}.mp4")
        video_path = main_video_root_path / video_chunk_path
        self.logger.info(f"Revisiting video chunk {video_chunk_path}...")

        with open(video_path, "rb") as video_file:
            video_data = base64.b64encode(video_file.read()).decode("utf-8")

        
        message = HumanMessage(
            content=[
                {"type": "text", "text": (
                    "## CHARACTER REFERENCE GUIDE\n"
                    "Study these reference images carefully. You will use ONLY these names.\n"
                    "Match faces by: facial features, overall appearance."
                )},
                
                {"type": "image_url", "image_url": {"url": self._encode_image(str(meta_path / "Monica Geller.png"))}},
                {"type": "text", "text": "Reference 1: Monica Geller"},
                {"type": "image_url", "image_url": {"url": self._encode_image(str(meta_path / "Ross Geller.png"))}},
                {"type": "text", "text": "Reference 2: Ross Geller"},
                {"type": "image_url", "image_url": {"url": self._encode_image(str(meta_path / "Rachel Green.png"))}},
                {"type": "text", "text": "Reference 3: Rachel Green"},
                {"type": "image_url", "image_url": {"url": self._encode_image(str(meta_path / "Chandler Bing.png"))}},
                {"type": "text", "text": "Reference 4: Chandler Bing"},
                {"type": "image_url", "image_url": {"url": self._encode_image(str(meta_path / "Phoebe Buffay.png"))}},
                {"type": "text", "text": "Reference 5: Phoebe Buffay"},
                {"type": "image_url", "image_url": {"url": self._encode_image(str(meta_path / "Joey Tribbiani.png"))}},
                {"type": "text", "text": "Reference 6: Joey Tribbiani"},
                
                {
                    "type": "media",
                    "mime_type": "video/mp4",
                    "data": video_data
                },

                {"type": "text", "text": "Caption this video in detail to help answer this user query given the video query: {query}"},
            ]
        )
        
        response = self.llm.invoke([message])
        self.logger.info(f"Revisited video chunk {video_chunk_path} with query {query}. New caption: {response.content}")
        return response.content