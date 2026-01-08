# Multimodal-GraphRAG

Multimodal-GraphRAG is an video analysis and retrieval system that combines the utilities VLM, Graph Databases and Vector Databases. It provides a person-grounded, context-aware way to index and query long video content by transforming visual and auditory information into a structured knowledge graph with graph reconstruction machinesim.

## Installation & Setup

### Prerequisites

- Python 3.12
- Docker & Docker Compose
- FFmpeg

### Python Packages

We use `uv` for lightning-fast dependency management, but you can also use `pip`.

```bash
# Using uv
uv sync
```

### Docker Services

The system relies on Neo4j for the Knowledge Graph and Qdrant for the Vector Database.

```bash
docker-compose up -d
```

This will start:

- **Neo4j**: Accessible at `http://localhost:7474` (Bolt: `7687`)
- **Qdrant**: Accessible at `http://localhost:6333`

### Environment Variables

Create a `.env` file based on `.env.example`.

## Project Structure

```text
Multimodal-GraphRAG/
├── data/                       # Input and metadata storage
│   ├── meta/                   # Character reference images for VLM grounding
│   └── episodes_splitted/      # Processed video chunks and captions
├── graph/                      # Hybrid Graph-Vector database logic
│   ├── KnowledgeGraph.py       # Main system orchestration (Query & Indexing)
│   ├── neo4j_client.py         # Neo4j Graph DB interface
│   ├── qdrant_db.py            # Qdrant Vector DB interface
│   └── prompts.py              # LLM prompts for summary and validation
├── preprocessor/               # Video and content preparation
│   ├── video_spilter.py        # FFmpeg splitting and subtitle alignment
│   ├── qwen3_vllm_inference.py  # VLM grounded captioning logic
│   └── helper.py               # Utilities (text chunking, time conversion)
├── llm/                        # LLM Orchestration
│   └── factory.py              # Multi-provider factory (Gemini, Groq, etc.)
├── db_data/                    # Persistent storage for Docker containers
├── index_data.py               # Entry script to build the Knowledge Graph
└── query_knowlage_graph.py     # Entry script for hybrid queries
```

## System Architecture & Workflow

### 1. Video Preprocessing

- **Splitting Long Video**: Long videos are automatically split into 1-minute clips using `ffmpeg` for efficiency without re-encoding.
- **Subtitle Alignment**: The system loads existing subtitle files (VTT/SRT) and aligns them with each video chunk.
  - Note: If subtitles are missing, the system can use ASR models (like Whisper) as a fallback.

### 2. VLM Grounded Captioning

- **Person Grounding**:
  - **Current Implementation**: The system maintains a reference list of the main characters provided in the data (located in `data/meta/`). These identities are fed to the Vision-Language Model along with the video to ground specific characters and ensure accurate, person-grounded captions.
  - **Future System Goals (To-Do)**: A more general need to have face recognition and clustering module is planned to automatically identify and group unique individuals within any video. This will include:
    1. **Face Detection & Clustering**: Automatically grouping faces into unique identities.
    2. **Character Identification**: Assigning names/IDs to each unique face cluster.
    3. **Dynamic Template Replacement**: Replacing generic character templates with real names in each video chunk (if agent can identify face with name using subtitles).
- **Structured Captions**: For each 1-minute clip, the VLM generates a detailed caption covering scene setting, characters present, actions, objects, and dialogue context.

### 3. Data Indexing and Graph Construction

The system utilizes agents from LangChain to bridge the gap between natural language and graph databases:

- **LLMGraphTransformer**: This agent acts as a text-to-graph. It processes the generated captions and summaries, extracting entities and relationships to convert them into a structured format compatible with **Cypher Query Language**. This enables the seamless feeding of unstructured video descriptions into the Neo4j graph database.
- **Grouping & Summarization**: To reduce redundancy, every 5 chunks are grouped and summarized before being passed to the graph transformer.

### 4. Hybrid Query-Aware Retrieval

The retrieval process is done by a hybrid approach that combines semantic search with graph-based reasoning:

- **GraphCypherQAChain**: This agent used for the retrieval phase. It takes a natural language user query and converts it into a valid **Cypher query**. This allows the system to retrieve precise information directly from the Neo4j knowledge graph based on the structured relationships indexed earlier.
- **Hybrid Logic**:
  1. **Graph First**: The query is first run against Neo4j using the `GraphCypherQAChain`.
  2. **Vector Fallback**: If the graph result is insufficient, the system falls back to a vector search in **Qdrant** (using semantic embeddings).
  3. **Graph Learning (Backfilling)**: If a valid answer is found via RAG/Vector search, the system extracts new knowledge and updates the Neo4j Graph, reconstructing missing information and improving the system over time.

