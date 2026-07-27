import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def sanitize_id(self, s: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '_', s)

    def chunk_file(self, file_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        text = file_data["text"]
        
        split_texts = self.splitter.split_text(text)
        
        project_id = file_data["project_id"]
        file_name = file_data["file_name"]
        sanitized_proj = self.sanitize_id(project_id)
        sanitized_file = self.sanitize_id(file_name)
        
        current_idx = 0
        for i, chunk_text in enumerate(split_texts):
            start_idx = text.find(chunk_text, current_idx)
            if start_idx == -1:
                start_idx = text.find(chunk_text)
            if start_idx != -1:
                char_start = start_idx
                char_end = start_idx + len(chunk_text)
                current_idx = start_idx + 1
            else:
                char_start = 0
                char_end = len(chunk_text)
                
            chunk_id = f"{sanitized_proj}_{sanitized_file}_{i}"
            
            chunks.append({
                "chunk_id": chunk_id,
                "project_id": project_id,
                "file_name": file_name,
                "file_path": file_data["file_path"],
                "subfolder_path": file_data["subfolder_path"],
                "char_start": char_start,
                "char_end": char_end,
                "text": chunk_text
            })
            
        return chunks
