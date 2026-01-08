import os
import re
import json
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from pathlib import Path
from tqdm import tqdm

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def prepare_inputs_for_vllm(messages, processor):
    """Prepare inputs in the format expected by vLLM for Qwen3-VL"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_chunk_folders(episode_path):
    episode_dir = Path(episode_path)
    chunk_folders = sorted([d for d in episode_dir.iterdir() if d.is_dir() and d.name.startswith('chunk_')])
    return chunk_folders

def create_messages_for_chunk(chunk_path, meta_path, episode_name, chunk_id):

    subtitle_file = chunk_path / "subtitles.json"
    with open(subtitle_file, "r") as f:
        subtitles_data = json.load(f)
    
    subtitles = subtitles_data["subtitles_without_timestamps"]
    subtitles_text = "\n".join([subtitle for subtitle in subtitles])
    subtitles_text = subtitles_text.replace("\n", " ")
    subtitles_text = normalize_whitespace(subtitles_text)
    
    video_files = list(chunk_path.glob("*.mp4"))
    video_file = str(video_files[0])
    
    # Construct messages
    messages = [
    {
        "role": "user",
        "content": [
            # --- CHARACTER REFERENCE SECTION ---
            {"type": "text", "text": (
                "## CHARACTER REFERENCE GUIDE\n"
                "Study these reference images carefully. You will use ONLY these names.\n"
                "Match faces by: facial features, overall appearance."
            )},
            
            {"type": "image", "image": str(meta_path / "Monica Geller.png")},
            {"type": "text", "text": "Reference 1: Monica Geller (memorize face)"},
            
            {"type": "image", "image": str(meta_path / "Ross Geller.png")},
            {"type": "text", "text": "Reference 2: Ross Geller (memorize face)"},
            
            {"type": "image", "image": str(meta_path / "Rachel Green.png")},
            {"type": "text", "text": "Reference 3: Rachel Green (memorize face)"},
            
            {"type": "image", "image": str(meta_path / "Chandler Bing.png")},
            {"type": "text", "text": "Reference 4: Chandler Bing (memorize face)"},
            
            {"type": "image", "image": str(meta_path / "Phoebe Buffay.png")},
            {"type": "text", "text": "Reference 5: Phoebe Buffay (memorize face)"},
            
            {"type": "image", "image": str(meta_path / "Joey Tribbiani.png")},
            {"type": "text", "text": "Reference 6: Joey Tribbiani (memorize face)"},
            
            # --- VIDEO SECTION ---
            {"type": "text", "text": f"\n## VIDEO TO ANALYZE\nEpisode: {episode_name}"},
            {
                "type": "video",
                "video": video_file,
                "fps": 1.0,
                "resized_height": 448,
                "resized_width": 448
            },
            
            # --- MAIN INSTRUCTION BLOCK ---
            {
                "type": "text",
                "text": (
                    f"## SUBTITLES (Context Only)\n"
                    f"\"{subtitles_text}\"\n\n"
                    
                    "Your task is to analyze the video clip above and provide a concise, structured description of what you observe. "
                    "You will:\n\n"
                    
                    "1. **Identify Characters**: Match people in the video to the reference images above using facial features and appearance\n"
                    "2. **Describe the Setting**: Identify the location, time of day, and lighting conditions\n"
                    "3. **Document Actions**: Record what each character is doing and how they interact\n"
                    "4. **List Objects**: Note significant props, furniture, and items visible in the scene\n"
                    "5. **Summarize Dialogue**: Use subtitles to understand the conversation topic and tone\n"
                    "6. **Track Changes**: Note any temporal changes in position, action, or scene composition\n\n"
                   
                    "### Rule 1: UNCERTAINTY PROTOCOL\n"
                    "- If ANY detail is unclear/ambiguous, write 'unclear' or 'not visible'\n"
                    "- NEVER guess clothing colors - use terms like 'dark colored', 'light colored', or 'unclear'\n"
                    "- NEVER invent objects mentioned in subtitles but not visible in video\n"
                    "- Confidence threshold: Only report details you're 80%+ certain about\n\n"
                    
                    "### Rule 2: ZERO REDUNDANCY POLICY\n"
                    "- Describe each character ONCE in 'Characters Present' section\n"
                    "- Describe each action ONCE in 'Actions' section (even if sustained)\n"
                    "- Example of WRONG redundancy: 'Joey holds mug. Joey drinks from mug. Joey holds mug.'\n"
                    "- Example of CORRECT approach: 'Joey holds and drinks from mug throughout'\n"
                    "- If an action changes over time, describe it ONLY in 'Temporal Notes'\n"
                    "- NEVER repeat the same sentence with minor variations\n\n"
                    
                    "### Rule 3: NAME INTEGRITY\n"
                    "- Use ONLY the 6 names from reference images above\n"
                    "- If you can't match a face to references, write 'Unidentified person' instead\n"
                    "- NEVER duplicate character names (e.g., two 'Joey Tribbiani' entries)\n"
                    "- Verify face match BEFORE writing the name\n\n"
                    
                    "### Rule 4: SUBTITLE-VISUAL SEPARATION\n"
                    "- Subtitles tell what's SAID, not what's SEEN\n"
                    "- If subtitles mention 'pizza' but you see sandwiches, describe sandwiches\n"
                    "- If subtitles say someone is 'angry' but they're smiling, describe as smiling\n"
                    "- Match dialogue to visible lip movements when possible\n\n"
                    
                    "### Rule 5: TEMPORAL CLARITY\n"
                    "- Static actions (sitting, wearing outfit) go in main sections\n"
                    "- Dynamic changes (standing up, entering room) go ONLY in 'Temporal Notes'\n"
                    
                    "### **SCENE SETTING:**\n"
                    "Template: [Location name] - [Time of day IF visible] - [Lighting type]\n"
                    "- Examples:\n"
                    "  ✓ `Central Perk coffee shop — daytime, natural window light'\n"
                    "  ✓ 'Monica’s apartment living room — evening, warm lamp lighting '\n"
                    "- Keep to 1-2 sentences maximum\n"
                    "- Focus on identifiable location markers (furniture, decor, setting type)\n\n"
                    
                    "### **CHARACTERS PRESENT:**\n"
                    "Template for EACH character (list each person ONCE):\n"
                    "[Name]: [Upper clothing] - [Lower clothing] - [Hair] - [Expression/State]\n\n"
                    "Clothing Description Rules:\n"
                    "- Use general colors only if CLEARLY visible: 'red', 'blue', 'dark', 'light'\n"
                    "- If uncertain: 'dark-colored top', 'light shirt', 'unclear color'\n"
                    "- Include clothing type: 'sweater', 'shirt', 'jacket', 'dress', 'overalls' 'hats' \n"
                    "Hair Description Rules:\n"
                    "- Color (if visible): 'blonde', 'brown', 'dark'\n"
                    "- Length: 'short', 'shoulder-length', 'long'\n"
                    "- Style: 'straight', 'curly', 'ponytail', 'loose'\n"
                    "Expression Guidelines:\n"
                    "- Use simple terms: 'smiling', 'serious', 'concerned', 'animated', 'relaxed'\n"
                    "- Match to visible facial cues, not subtitle emotions\n"
                    "Example:\n"
                    "✓ Monica Geller"  
                    "- Upper: dark-colored sweater"  
                    "- Lower: jeans"  
                    "- Hair: dark, tied back"  
                    "- Expression: animated"  
                    "✓ Chandler Bing" 
                    "- Upper: light-colored shirt"  
                    "- Lower: dark pants"  
                    "- Hair: short, brown"  
                    "- Expression: smirking"  
                    "✗ Joey Tribbiani "
                    "-white shirt and denim jacket... Joey Tribbiani: white shirt.. (REDUNDANT) "
                    "### **ACTIONS AND INTERACTIONS:**\n"
                    "Write as a brief NARRATIVE (3-5 sentences), NOT a repetitive list.\n\n"
                    "What to Include:\n"
                    "- Primary actions: sitting, standing, walking, gesturing, talking\n"
                    "- Object interactions: holding, using, manipulating items\n"
                    "- Character interactions: talking to, looking at, touching, reacting to others\n"
                    "- Spatial relationships: who's near whom, positioning in frame\n\n"
                    "What to AVOID:\n"
                    "✗ Repeating same action multiple times: 'Joey holds mug. Joey holds mug. Joey holds mug.'\n"
                    "✗ Over-describing sustained states: If someone sits throughout, mention once\n"
                    "✗ Listing every micro-gesture: Focus on significant actions only\n\n"
                    "Example Format:\n"
                    "'Monica sits on the couch, gesturing expressively while speaking to Rachel, who holds a coffee mug and nods occasionally. "
                    "Chandler leans back in his chair with arms crossed, smirking at the conversation. "
                    "Joey enters from the kitchen area holding a sandwich and sits at the table.'\n\n"
                    
                    "### **OBJECTS AND PROPS:**\n"
                    "List format: [Object type]: [Color/description IF clear] ([location/context])\n\n"
                    "Priority Items:\n"
                    "- Food/drinks: 'coffee mug: white (on table)', 'sandwich (in hand)', 'pizza box (unclear)'\n"
                    "- Furniture: 'orange couch (center)', 'wooden table', 'bar stools'\n"
                    "- Props being used: 'guitar (being played)', 'book (being read)'\n\n"
                    "Guidelines:\n"
                    "- List 3-8 items maximum (most prominent/relevant)\n"
                    "- Skip background clutter unless significant\n"
                    "- Use 'unclear' for ambiguous objects\n"
                    "- Don't invent objects from subtitles\n\n"
                    "Example:\n"
                    "- Coffee mugs: various colors (on table)\n"
                    "- Orange couch (center of frame)\n"
                    "- Brick wall decor (background)\n"
                    "- Menu board (behind counter)\n\n"
                    
                    "### **DIALOGUE CONTEXT:**\n"
                    "Based on subtitles + visual cues, describe in 2-4 sentences:\n\n"
                    "Include:\n"
                    "- Topic: What are they discussing? (use subtitles)\n"
                    "- Participants: Who's speaking to whom? (use visuals)\n"
                    "- Tone: happy, serious, joking, tense, casual (match facial expressions)\n"
                    "- Reactions: Notable responses from listeners\n\n"
                    "Template:\n"
                    "'[Speaker] discusses [topic] with [listener(s)]. "
                    "The conversation has a [tone] tone, with [speaker] appearing [expression] and [listener] reacting by [reaction].'\n\n"
                    
                    "### **TEMPORAL NOTES:**\n"
                    "ONLY include if actions/positions CHANGE during the clip.\n\n"
                    "Format: [Character] [specific change]\n\n"
                    "What qualifies as a change:\n"
                    "✓ Movement: entering, exiting, standing up, sitting down\n"
                    "✓ Object interaction shifts: picks up, sets down, passes\n"
                    "✓ Major expression changes: calm → excited, serious → laughing\n"
                    "✓ Position changes: moves to different location in frame\n\n"
                    "What NOT to include:\n"
                    "✗ Sustained actions (already in main sections)\n"
                    "✗ Minor gestures (hand movements during conversation)\n"
                    "✗ Repetition of actions listed above\n\n"
                    "Examples:\n"
                    "Scene begins with all characters seated\n"
                    "Phoebe enters frame from right side\n"
                    "Chandler stands up and walks toward counter\n"
                    "Joey sets down mug\n\n"
                    
                    "Before submitting your description, verify:\n\n"
                    "Character Names:\n"
                    "  - Each name appears ONCE in 'Characters Present'\n"
                    "  - All names match reference images exactly\n"
                    "  - No duplicate names listed\n\n"
                    "Color Accuracy:\n"
                    "  - Replaced any uncertain colors with 'dark', 'light', or 'unclear'\n"
                    "  - Verified colors against visible evidence, not assumptions\n"
                    "  - Used qualifying words: 'appears to be', 'possibly'\n\n"
                    "Redundancy Elimination:\n"
                    "  - No repeated sentences or near-identical descriptions\n"
                    "  - Each action described ONCE in appropriate section\n"
                    "  - Sustained actions summarized (not repeated)\n\n"
                    "Hallucination Prevention:\n"
                    "  - Removed any objects not actually visible\n"
                    "  - Verified actions match what's shown, not just subtitles\n"
                    "  - No invented details about clothing, props, or actions\n\n"
                    "Temporal Accuracy:\n"
                    "  - Static info in main sections, changes ONLY in Temporal Notes\n"
                    "  - No overlap between sections\n\n"
                    "Format Compliance:\n"
                    "  - All section headers included\n"
                    "  - Template structures followed\n"
                    "  - Concise writing (quality > quantity)\n\n"    
                )
            }
            ]
        }
    ]

    
    return messages, subtitles_text

if __name__ == '__main__':
    # Configuration
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    base_path = Path("/teamspace/studios/this_studio/friends-dataset-clips-60-seconds")
    episodes_root = base_path / "episodes_splitted"
    meta_path = base_path / "meta"


    processor = AutoProcessor.from_pretrained(model_id)
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        max_model_len=10000,
        enforce_eager=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
        seed=0
    )

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=10000,
        top_p=0.95,
        top_k=50,
        stop_token_ids=[],
    )


    episode_dirs = sorted([d for d in episodes_root.iterdir() if d.is_dir()])
    print(f"Found {len(episode_dirs)} episodes")

    for episode_path in episode_dirs:
        episode_name = episode_path.name
        output_file = episode_path / "all_captions.json"
        
        print("\n" + "=" * 100)
        print(f"Processing episode: {episode_name}")

        chunk_folders = get_chunk_folders(episode_path)
        print(f"Found {len(chunk_folders)} chunks")

        all_captions = {}

        for chunk_folder in tqdm(chunk_folders, desc=f"{episode_name}"):
            chunk_id = chunk_folder.name

            try:
                messages, subtitles_text = create_messages_for_chunk(
                    chunk_folder, meta_path, episode_name, chunk_id
                )

                inputs = [prepare_inputs_for_vllm(messages, processor)]
                outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
                generated_text = outputs[0].outputs[0].text

                print(f"Episode Name {episode_name} - chunk {chunk_id}\n")
                print(f"Generated capution:\n\n {generated_text}\n")
                print(f"Subtitle {subtitles_text}")

                break

                all_captions[chunk_id] = {
                    "caption": generated_text,
                    "subtitles": subtitles_text,
                    "episode": episode_name,
                    "chunk": chunk_id
                }

                caption_file = chunk_folder / "structured_caption.txt"
                with open(caption_file, "w", encoding="utf-8") as f:
                    f.write(f"Episode: {episode_name}\n")
                    f.write(f"Chunk: {chunk_id}\n")
                    f.write(f"Subtitles: {subtitles_text}\n\n")
                    f.write(generated_text)

            except Exception as e:
                all_captions[chunk_id] = {
                    "caption": None,
                    "error": str(e),
                    "subtitles": None,
                    "episode": episode_name,
                    "chunk": chunk_id
                }

        # Save episode-level JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_captions, f, indent=2, ensure_ascii=False)

        successful = sum(1 for v in all_captions.values() if v.get("caption"))
        failed = len(all_captions) - successful

        print(f"Episode complete: {episode_name}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
    print("\nALL EPISODES PROCESSED ")
