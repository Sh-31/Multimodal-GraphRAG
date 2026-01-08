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
