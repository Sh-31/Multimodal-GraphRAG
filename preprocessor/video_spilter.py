# https://www.kaggle.com/code/sherif31/videospilter
# TODO: add subtitle generation using ASR Model like whisper
import os
import json
from pathlib import Path
import moviepy.editor as mp
import webvtt
from typing import List, Dict, Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import subprocess

# Check if ffmpeg is installed, if not install it
try:
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    print("✓ ffmpeg is already installed")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Installing ffmpeg...")
    os.system('apt-get update && apt-get install -y ffmpeg')
    print("✓ ffmpeg installed successfully")

class VideoSpilter:
    def __init__(self, chunk_duration: int = 60):
        self.chunk_duration = chunk_duration

    
    def split_video(self, input_path: str, output_folder: str) -> List[Dict]:
        os.makedirs(output_folder, exist_ok=True)
        
        video = mp.VideoFileClip(input_path)
        total_duration = video.duration
        num_clips = int(total_duration / self.chunk_duration)
        if total_duration % self.chunk_duration != 0:
            num_clips += 1
        video.close()
        
        chunks = []
        for i in tqdm(range(num_clips), desc="Splitting video"):
            start_time = i * self.chunk_duration
            end_time = min((i + 1) * self.chunk_duration, total_duration)
            save_name = f"{i + 1}".zfill(5)
            
            # Create chunk subfolder
            chunk_folder = os.path.join(output_folder, f"chunk_{save_name}")
            os.makedirs(chunk_folder, exist_ok=True)
            
            output_path = os.path.join(chunk_folder, f"{save_name}.mp4")
            
            # Use ffmpeg directly for fast stream copying (no re-encoding)
            duration = end_time - start_time
            command = f'ffmpeg -ss {start_time} -i "{input_path}" -t {duration} -c copy -avoid_negative_ts 1 "{output_path}" -y -loglevel error'
            os.system(command)
            
            chunks.append({
                'path': output_path,
                'chunk_folder': chunk_folder,
                'start_time': start_time,
                'end_time': end_time,
                'chunk_id': i
            })
        
        return chunks
    
    def _get_existing_chunks(self, output_folder: str, input_path: str) -> List[Dict]:
        """Get chunk information from existing split videos."""
        video = mp.VideoFileClip(input_path)
        total_duration = video.duration
        video.close()
        
        # Get chunk folders
        chunk_folders = sorted([f for f in os.listdir(output_folder) if f.startswith('chunk_')])
        chunks = []
        
        for i, folder_name in enumerate(chunk_folders):
            chunk_folder = os.path.join(output_folder, folder_name)
            mp4_files = [f for f in os.listdir(chunk_folder) if f.endswith('.mp4')]
            
            if mp4_files:
                start_time = i * self.chunk_duration
                end_time = min((i + 1) * self.chunk_duration, total_duration)
                chunks.append({
                    'path': os.path.join(chunk_folder, mp4_files[0]),
                    'chunk_folder': chunk_folder,
                    'start_time': start_time,
                    'end_time': end_time,
                    'chunk_id': i
                })
        
        return chunks
    
    def load_subtitles(self, subtitle_path: str):
        """
        Load subtitle file (VTT or SRT format) as dict
        """
        if not os.path.exists(subtitle_path):
            raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
        
        # webvtt can handle both .vtt and .srt files
        subs = webvtt.read(subtitle_path)
        return subs
    
    def get_subtitles_for_timerange(
        self, 
        subtitles, 
        start_time: float, 
        end_time: float
    ) -> Tuple[List[Dict], List[str]]:
    
        def time_to_seconds(time_str: str) -> float:
            """Convert webvtt time string (HH:MM:SS.mmm) to seconds."""
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        
        subs_with_time = []
        subs_without_time = []
        
        for caption in subtitles:
            sub_start = time_to_seconds(caption.start)
            sub_end = time_to_seconds(caption.end)
            
            # Check if subtitle overlaps with time range
            # start_time sub_start sub_end  end_time
            # start_time < sub_end (overlab)
            # sub_start  < end_time (overlab)
            if sub_start < end_time and sub_end > start_time:
                subs_with_time.append({
                    'start': sub_start,
                    'end': sub_end,
                    'text': caption.text
                })
                subs_without_time.append(caption.text)
        
        return subs_with_time, subs_without_time
    
    def process_video_for_inference(
        self,
        video_path: str,
        subtitle_path: str,
        output_folder: str,
        parallel: bool = False
    ) -> List[Dict]:
        """
        Complete pipeline: split video, load subtitles, and create JSON files.
        
        Args:
            video_path: Path to video file
            subtitle_path: Path to subtitle file (.srt or .vtt)
            output_folder: Directory for video chunks
            parallel: Use parallel processing for video splitting (default: False)
            
        Returns:
            List of chunk information dictionaries
        """
        # Step 1: Split video into chunks
        print(f"Splitting video into {self.chunk_duration}s chunks...")
        if parallel:
            chunks = self.split_video_parallel(video_path, output_folder)
        else:
            chunks = self.split_video(video_path, output_folder)
        print(f"Created {len(chunks)} video chunks")
        
        # Step 2: Load subtitles
        print(f"Loading subtitles from {subtitle_path}...")
        subtitles = self.load_subtitles(subtitle_path)
        print(f"Loaded {len(subtitles)} subtitle entries")
        
        # Step 3: Align subtitles with chunks and create JSON files
        for chunk in chunks:
            chunk_subs_with_time, chunk_subs_without_time = self.get_subtitles_for_timerange(
                subtitles,
                chunk['start_time'],
                chunk['end_time']
            )
            
            # Create JSON data
            json_data = {
                'subtitles_with_timestamps': chunk_subs_with_time,
                'subtitles_without_timestamps': chunk_subs_without_time
            }
            
            # Save JSON file in chunk folder
            json_path = os.path.join(chunk['chunk_folder'], 'subtitles.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"Chunk {chunk['chunk_id']}: {len(chunk_subs_with_time)} subtitle entries aligned")
        
        return chunks

def process_all_episodes(
    episodes_folder: str,
    subtitles_folder: str,
    output_base_folder: str = "episodes_splitted",
    chunk_duration: int = 60,
    parallel: bool = False
):
    """
    Process all episodes in a folder.
    
    Args:
        episodes_folder: Folder containing episode video files
        subtitles_folder: Folder containing subtitle files (.vtt or .srt)
        output_base_folder: Base output folder (default: "episodes_splitted")
        chunk_duration: Duration of each chunk in seconds
        parallel: Use parallel processing for video splitting
    """
    processor = VideoSpilter(chunk_duration=chunk_duration)
    
    # Get all video files
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    episode_files = [f for f in os.listdir(episodes_folder) 
                     if any(f.endswith(ext) for ext in video_extensions)]
    
    episode_files = sorted(episode_files)
    print(f"Found {len(episode_files)} episodes to process")
    
    for episode_file in tqdm(episode_files, desc="Processing episodes"):
        print(f"\n{'='*60}")
        print(f"Processing: {episode_file}")
        print(f"{'='*60}")
        
        episode_name = Path(episode_file).stem
        ext = Path(episode_file).suffix
        
        # Find matching subtitle file
        subtitle_file = None
        for ext in ['.vtt', '.srt']:
            potential_sub = os.path.join(subtitles_folder, f"{episode_name.strip()}{ext}")
            if os.path.exists(potential_sub):
                subtitle_file = potential_sub
                break
        
        if subtitle_file is None:
            print(f"Warning: No subtitle file found for {episode_file}, skipping...")
            continue
        
        episode_output_folder = os.path.join(output_base_folder, episode_name)
        os.makedirs(episode_output_folder, exist_ok=True)
        video_path = os.path.join(episodes_folder, episode_file)
        
        try:
            processor.process_video_for_inference(
                video_path=video_path,
                subtitle_path=subtitle_file,
                output_folder=episode_output_folder
            )
            print(f"Successfully processed {episode_file}")
        except Exception as e:
            print(f"Error processing {episode_file}: {str(e)}")
            continue

if __name__ == "__main__":
    process_all_episodes(
        episodes_folder="/kaggle/input/v1/other/default/1/episodes",
        subtitles_folder="/kaggle/input/v1/other/default/1/vtts",
        output_base_folder="/kaggle/working/episodes_splitted",
        chunk_duration=60
    )