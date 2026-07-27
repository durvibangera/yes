import os
from pathlib import Path
from typing import List, Dict, Any

class Parser:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def discover_projects(self) -> List[str]:
        projects = []
        if not self.data_root.exists():
            return projects
        for entry in self.data_root.iterdir():
            if entry.is_dir():
                projects.append(entry.name)
        return projects

    def parse_project(self, project_id: str) -> List[Dict[str, Any]]:
        project_path = self.data_root / project_id
        files_data = []
        
        for md_file in project_path.rglob('*.md'):
            # Relative path from project root
            try:
                subfolder_path = str(md_file.parent.relative_to(project_path))
            except ValueError:
                subfolder_path = "."
                
            if subfolder_path == ".":
                subfolder_path = ""
                
            # Read content
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                try:
                    with open(md_file, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Skipping file {md_file} due to read error: {e}")
                    continue
                    
            files_data.append({
                "project_id": project_id,
                "file_name": md_file.name,
                "file_path": str(md_file.relative_to(self.data_root)),
                "subfolder_path": subfolder_path,
                "text": content
            })
            
        return files_data
