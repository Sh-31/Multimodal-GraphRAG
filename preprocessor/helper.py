import os
import logging
from datetime import datetime
from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

def load_text(source: str | Path, is_file: bool = False) -> str:
    if is_file or (isinstance(source, (str, Path)) and os.path.exists(source) and os.path.isfile(source)):
        try:
            with open(source, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Error reading file {source}: {e}")
    return str(source)

def seconds_to_time(seconds: float) -> str:
    """Convert seconds to webvtt time string (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    # Format with zero-padding and milliseconds
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}"



def chunk_text(text: str, size: int = 512, overlap: int = 256, method: str = 'recursive') -> List[str]:
    if not text:
        return []

    if method == 'fixed':
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=size,
            chunk_overlap=overlap,
            keep_separator=False
        )
    elif method == 'recursive':
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    else:
        raise ValueError(f"Unknown chunking method: {method}. Use 'fixed' or 'recursive'.")

    return splitter.split_text(text)

def get_logger(name="MultimodalGraphRAG"):
    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    log_filename = logs_dir / f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger