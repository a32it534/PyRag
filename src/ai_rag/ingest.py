import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from ai_rag.config import get_settings
from ai_rag.chunking import chunk_text
from ai_rag.vectorstore import VectorStoreService

class IngestionService:
    def __init__(self, vector_store: Optional[VectorStoreService] = None):
        self.settings = get_settings()
        self.vector_store = vector_store or VectorStoreService()

    @staticmethod
    def calculate_file_hash(content: bytes) -> str:
        return hashlib.md5(content).hexdigest()

    def process_text_content(self, text: str, source_name: str, file_hash: str = "") -> int:
        """پردازش متن خام (مناسب برای متن مستقیم یا فایل آپلود شده در Streamlit)"""
        chunks = chunk_text(
            text=text,
            max_chars=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap
        )
        if not chunks:
            return 0

        now_iso = datetime.now(timezone.utc).isoformat()
        total_chunks = len(chunks)

        metadatas: List[Dict[str, Any]] = [
            {
                "source": source_name,
                "chunk_index": idx + 1,
                "total_chunks": total_chunks,
                "file_hash": file_hash,
                "char_length": len(chunk),
                "indexed_at": now_iso
            }
            for idx, chunk in enumerate(chunks)
        ]

        return self.vector_store.add_chunks_with_metadata(chunks=chunks, metadatas=metadatas)

    def process_file(self, file_path: Path) -> int:
        """پردازش یک فایل روی دیسک"""
        if not file_path.exists() or not file_path.is_file():
            return 0
        
        # پشتیبانی از فرمت‌های متنی
        if file_path.suffix.lower() not in [".txt", ".md"]:
            return 0

        raw_bytes = file_path.read_bytes()
        file_hash = self.calculate_file_hash(raw_bytes)
        text = raw_bytes.decode("utf-8", errors="ignore")

        return self.process_text_content(
            text=text,
            source_name=file_path.name,
            file_hash=file_hash
        )

    def ingest_directory(self, target_dir: Optional[Path] = None) -> Dict[str, int]:
        """پردازش تمام اسناد داخل دایرکتوری"""
        self.settings.ensure_dirs()
        directory = target_dir or self.settings.documents_dir
        results: Dict[str, int] = {}

        for file_path in directory.glob("*.*"):
            if file_path.suffix.lower() in [".txt", ".md"]:
                chunks_count = self.process_file(file_path)
                results[file_path.name] = chunks_count

        return results
