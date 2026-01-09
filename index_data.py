import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm.factory import LLMFactory
from graph.KnowledgeGraph import KnowledgeGraph
from preprocessor.helper import seconds_to_time

def index_data():
    print("Building Knowledge Graph")
    llm = LLMFactory.get_llm()
    knowledge_graph = KnowledgeGraph(llm=llm)
    videos_dir = "/home/sh-31/repo/Multimodal-GraphRAG/data/episodes_splitted"
    videos_dir = Path(videos_dir)
    
    videos_names = []
    for video_dir in videos_dir.iterdir():
        if video_dir.is_dir():
            videos_names.append(video_dir.name)

    videos_names = sorted(videos_names)
    
    # for testing index one episode one chunk group
    for video_name in videos_names:
        video_path = videos_dir / video_name
        
        if video_name != "1x24 - The One Where Rachel Finds Out":
            continue
        print(f"Processing video: {video_name}")
        with open(video_path / "all_captions.json", "r") as f:
            all_captions = json.load(f)

        for i, chunk in enumerate(all_captions):
            
            with open(video_path / chunk / "subtitles.json", "r") as f:
                chunk_subtitles = json.load(f)

            chunk_subtitles = chunk_subtitles["subtitles_with_timestamps"]
            
            if len(chunk_subtitles) == 0:
                all_captions[chunk]["start_chunk"] = ""
                all_captions[chunk]["end_chunk"] = ""
                continue
            
            start_of_chunk = seconds_to_time(chunk_subtitles[0]["start"])
            end_of_chunk = seconds_to_time(chunk_subtitles[-1]["end"])

            all_captions[chunk]["start_chunk"] = start_of_chunk
            all_captions[chunk]["end_chunk"] = end_of_chunk
                    
        merged_chunks_jsons = knowledge_graph.process_chunks_video(all_captions)

        for i, merged_chunks_json in enumerate(merged_chunks_jsons):
           with open(video_path / f"merged_chunks_{i}.json", "w") as f:
                json.dump(merged_chunks_json, f, indent=2)
        
if __name__ == "__main__":
    index_data()
