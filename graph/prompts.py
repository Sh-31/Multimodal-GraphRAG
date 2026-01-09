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
Task: Generate a Cypher query to answer the user's question.
You are a Neo4j expert.

Schema:
{schema}

STRICT DATA RULES:
1. NEVER use the label 'Episode'. It is inconsistent.
2. The 'video_name' and 'time_steps' are ALWAYS stored as PROPERTIES on RELATIONSHIPS (the edges).
3. To find an episode or time, you MUST return 'r.video_name' or 'r.time_steps'.
4. Node IDs use varied casing and full names (e.g., 'Joey Tribbiani', 'Cigarette').
5. ALWAYS use 'toLower(node.id) CONTAINS toLower("string")' for all name/object/location filters to ensure a match.
6. MANDATORY: Use a generic relationship `-[r]->` instead of specific types like `-[r:HOLDS]->` to avoid missing matches on synonyms.

Example Questions:
- Question: "When did Rachel appear in Central Perk?"
  Query: MATCH (p:Person)-[r]->(l:Location) WHERE toLower(p.id) CONTAINS "rachel" AND toLower(l.id) CONTAINS "central perk" RETURN r.video_name, r.time_steps

Question: {question}
Cypher Query:"""
)
