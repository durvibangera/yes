import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MEMGRAPH_HOST: str = "localhost"
    MEMGRAPH_PORT: int = 7687
    DATA_ROOT: str = "./data"
    SIMILARITY_THRESHOLD: float = 0.92
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 60
    OPENAI_API_KEY: str = None
    
    # Entity extraction settings
    ENTITY_TYPES: List[str] = [
        "MATERIAL", "STANDARD_ID", "PROCESS", "SPECIFICATION", "TOLERANCE", "GRADE",
        "CRAFT_TYPE", "REGULATORY_TERM", "CONDITION", "COMPONENT", "PARAMETER", "EQUIPMENT"
    ]

    # Phase 2 - LLM settings
    LLM_BACKEND: str = "ollama"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    OLLAMA_HOST: str = "http://localhost:11434"
    GENERATION_TEMPERATURE: float = 0.7
    MAX_CONCURRENT_REQUESTS: int = 4

    # Phase 2 - Generation counts
    # Heavily biased toward multi-hop to stress-test Graph RAG advantage in benchmark.
    SINGLE_HOP_PER_PROJECT: int = 10
    MULTIHOP_PER_PROJECT: int = 35
    METADATA_PER_PROJECT: int = 3
    NULL_PER_PROJECT: int = 3
    TOP_K_FOR_LABELING: int = 10

    # Phase 2 - Holdout folders (never used to tune router)
    HOLDOUT_FOLDERS: str = "ASME,AWS,British Standards,DIN,EN Spec,ISO,MIL Stds,ASTM & OTHER STANDARDS"

    @property
    def holdout_folder_list(self) -> List[str]:
        return [f.strip() for f in self.HOLDOUT_FOLDERS.split(',')]
    
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

settings = Settings()
