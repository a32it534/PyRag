import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    default_model: str
    embedding_model: str
    chroma_db_dir: Path
    documents_dir: Path
    chunk_size: int
    chunk_overlap: int

    def ensure_dirs(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)

def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "ollama").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1").strip(),
        default_model=os.getenv("OPENAI_MODEL", "qwen2.5:7b").strip(),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "bge-m3:latest").strip(),
        chroma_db_dir=Path(os.getenv("CHROMA_DB_DIR", "./chroma_db")),
        documents_dir=Path(os.getenv("DOCUMENTS_DIR", "./documents")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
    )
