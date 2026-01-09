from langchain_core.prompts import PromptTemplate

Summarize_Group_Video_Chunks = PromptTemplate(
    input_variables=["text"],
    template="""
You are given multiple consecutive caption chunks generated independently by a Vision-Language Model (VLM) from the same video sequence.

Your task is to merge all chunks into ONE coherent, high-level summary that captures the overall narrative and context.

### GUIDELINES:
- Treat all chunks as temporally consecutive.
- Merge overlapping or repeated information.
- Prioritize story, dialogue topics, character relationships, intentions, and emotional dynamics.
- Preserve scene transitions and meaningful actions.
- Remove or compress repetitive visual details (clothing, furniture, props) unless narratively important.

### WHAT TO IGNORE OR COMPRESS:
- Exact clothing details unless narratively important.
- Repeated mentions of the same props or furniture.
- Micro-movements unless they signal a scene transition or emotional shift.

### OUTPUT STRUCTURE (STRICT):
Produce a structured summary with the following sections:

**SCENE OVERVIEW**
Briefly describe where and when the events take place and whether the scene shifts.

**KEY CHARACTERS**
List the characters involved and their roles or emotional states if relevant.

**MERGED NARRATIVE SUMMARY**
A concise but information-dense paragraph describing what happens across all chunks, written as a continuous story.

**IMPORTANT CONTEXT OR THEMES**
Bullet points highlighting:
- Relationship developments
- Emotional tone
- Plot setup or payoff
- Notable transitions between locations or conversations

### INPUT CAPTIONS:
{text}
"""
)

Answer_Validity_Prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
You are an expert answer validator. Your task is to determine if the provided answer actually answers the question.

Question: {question}
Answer: {answer}

If the answer says "I don't know", "I lack information", "There is no information", or purely hallucinates without facts, return "INVALID".
If the answer provides relevant information that addresses the question, return "VALID".

Return ONLY "VALID" or "INVALID".
"""
)

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
Task: Generate a Cypher query to answer the user's question about video content.
You are a Neo4j expert specializing in video metadata graphs.

Schema:
{schema}

STRICT DATA RULES:
1. NO 'EPISODE' NODES: There are NO nodes labeled 'Episode'. Video names are properties of RELATIONS.
2. METADATA ON EDGES: The properties 'video_name' and 'time_steps' are on the RELATIONSHIPS.
3. FUZZY & CASE-INSENSITIVE: Always use 'toLower(node.id) CONTAINS toLower("search_term")' for filtering.
4. GENERIC RELATIONS: Use generic relationship markers `-[r]->` or specific paths if needed, but ALWAYS include Document context.
5. MULTI-ENTITY PATHS: For questions involving people, objects, and locations, chain them together or ensure they share the same Document.
6. DEDUPLICATION: Always use 'RETURN DISTINCT' to avoid duplicate answers.
7. ALWAYS LINK TO DOCUMENT: Start every query with `MATCH (d:Document)-[r:MENTIONS]->(node)`. This allows searching within `d.text` for broad context (actions, emotions, scenes) and retrieving the source text. RETURN `d.text` as `document_context`.

Example Questions & Strategic Cypher:

- "Show me all scenes where Chandler is using a laptop."
  Query: MATCH (d:Document)-[r:MENTIONS]->(p:Person), (d)-[:MENTIONS]->(o:Object)
         WHERE toLower(p.id) CONTAINS "chandler" 
         AND toLower(o.id) CONTAINS "laptop"
         RETURN DISTINCT r.video_name, r.time_steps, d.text AS document_context

- "Did Rachel change her hairstyle?"
  Query: MATCH (d:Document)-[r:MENTIONS]->(p:Person)
         WHERE toLower(p.id) CONTAINS "rachel"
         AND (
            toLower(d.text) CONTAINS "hair" 
            OR toLower(d.text) CONTAINS "style"
         )
         AND (
            toLower(d.text) CONTAINS "change"
            OR toLower(d.text) CONTAINS "new look"
            OR toLower(d.text) CONTAINS "cut"
            OR toLower(d.text) CONTAINS "different"
         )
         RETURN DISTINCT r.video_name, r.time_steps, d.text AS document_context

- "What location was Ross in when he had a dinosaur bone?"
  Query: MATCH (d:Document)-[r:MENTIONS]->(p:Person), (d)-[:MENTIONS]->(o:Object), (d)-[:MENTIONS]->(l:Location)
         WHERE toLower(p.id) CONTAINS "ross" 
         AND toLower(o.id) CONTAINS "dinosaur"
         RETURN DISTINCT l.id, r.video_name, d.text AS document_context

- "In which episodes did we see Gunther at Central Perk?"
  Query: MATCH (d:Document)-[r:MENTIONS]->(p:Person), (d)-[:MENTIONS]->(l:Location)
         WHERE toLower(p.id) CONTAINS "gunther" 
         AND toLower(l.id) CONTAINS "central perk"
         RETURN DISTINCT r.video_name, d.text AS document_context

- "Did any character interact with a pizza in a kitchen?"
  Query: MATCH (d:Document)-[r:MENTIONS]->(p:Person), (d)-[:MENTIONS]->(o:Object), (d)-[:MENTIONS]->(l:Location)
         WHERE toLower(o.id) CONTAINS "pizza"
         AND toLower(l.id) CONTAINS "kitchen"
         RETURN DISTINCT p.id, r.video_name, r.time_steps, d.text AS document_context

Question: {question}
Cypher Query:"""
)
