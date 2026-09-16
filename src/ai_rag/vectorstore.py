import hashlib
import logging
import random
from typing import Any, Dict, List, Optional

import chromadb

from ai_rag.config import get_settings
from ai_rag.llm import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    سرویس مدیریت ذخیره‌سازی، جست‌وجوی هوشمند، بازیابی و حذف چانک‌ها در ChromaDB.
    """

    def __init__(
        self,
        collection_name: str = "rag_knowledge",
        path: Optional[str] = None
    ):
        self.settings = get_settings()

        db_path = (
            str(self.settings.chroma_db_dir)
            if path is None
            else path
        )

        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine"
            }
        )

        self.embedder = EmbeddingService()

    # ------------------------------------------------------------------
    # افزودن یا به‌روزرسانی چانک‌ها
    # ------------------------------------------------------------------

    def add_chunks_with_metadata(
        self,
        chunks: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> int:
        """
        ذخیره یا به‌روزرسانی چانک‌ها در ChromaDB.

        شناسه‌ی هر رکورد از ترکیب موارد زیر ساخته می‌شود:
        source + chunk_index + content

        استفاده از upsert باعث می‌شود رکوردهای تکراری ایجاد نشوند.
        """
        if not chunks:
            return 0

        if len(chunks) != len(metadatas):
            raise ValueError(
                "تعداد chunks و metadatas باید برابر باشد."
            )

        ids: List[str] = []

        for index, (chunk, metadata) in enumerate(
            zip(chunks, metadatas)
        ):
            source = metadata.get("source", "")
            chunk_index = metadata.get("chunk_index", index)

            raw_id = f"{source}_{chunk_index}_{chunk}"

            record_id = hashlib.sha256(
                raw_id.encode("utf-8")
            ).hexdigest()[:16]

            ids.append(record_id)

        embeddings = self.embedder.embed_batch(chunks)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        return len(chunks)

    # ------------------------------------------------------------------
    # جست‌وجوی معنایی و هوشمند
    # ------------------------------------------------------------------

    def search_with_metadata(
        self,
        query: str,
        top_k: int = 3,
        source_filter: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        جست‌وجوی معنایی هوشمند و برگرداندن متن، شناسه، متادیتا و امتیاز شباهت.

        :param query: متن پرسش کاربر
        :param top_k: حداکثر تعداد نتایج درخواستی
        :param source_filter: فیلتر کردن جستجو بر روی یک فایل/منبع خاص (در صورت ارسال)
        :param min_score: آستانه حداقل امتیاز شباهت (از 0.0 تا 1.0) برای فیلتر نویز
        :return: لیستی از دیشکنری‌های حاوی id, content, metadata و score
        """
        if not query or not query.strip():
            return []

        if self.collection.count() == 0:
            return []

        query_vector = self.embedder.embed_text(query)

        # ساخت شرط فیلتر متادیتایی در صورت درخواست سند خاص
        where_clause = None
        if source_filter and source_filter.strip():
            where_clause = {"source": source_filter.strip()}

        query_params: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }

        if where_clause:
            query_params["where"] = where_clause

        try:
            results = self.collection.query(**query_params)
        except Exception as e:
            logger.error(f"Error executing ChromaDB query: {e}")
            return []

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        items: List[Dict[str, Any]] = []

        for index, (doc, meta, distance) in enumerate(
            zip(docs, metas, distances)
        ):
            record_id = (
                ids[index]
                if index < len(ids)
                else None
            )

            # محاسبه امتیاز شباهت بر اساس فاصله کسینوسی (Cosine Distance)
            score = (
                1.0 - distance
                if distance is not None
                else 0.0
            )

            # فیلتر آستانه شباهت برای حذف چانک‌های بی‌ربط
            if score < min_score:
                continue

            items.append({
                "id": record_id,
                "content": doc,
                "metadata": meta or {},
                "score": round(score, 4)
            })

        return items

    def search(
        self,
        query: str,
        top_k: int = 3,
        source_filter: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[str]:
        """
        جست‌وجوی ساده و برگرداندن فقط متن چانک‌های تاییدشده.
        """
        items = self.search_with_metadata(
            query=query,
            top_k=top_k,
            source_filter=source_filter,
            min_score=min_score
        )

        return [
            item["content"]
            for item in items
        ]

    # ------------------------------------------------------------------
    # دریافت چانک‌های تصادفی
    # ------------------------------------------------------------------

    def get_random_chunks(
        self,
        limit: int = 4
    ) -> List[Dict[str, Any]]:
        """
        دریافت تعدادی چانک تصادفی برای حالت Cold Start و نمایش اولیه.
        """
        try:
            if limit <= 0:
                return []

            if self.collection.count() == 0:
                return []

            results = self.collection.get(
                include=[
                    "documents",
                    "metadatas"
                ]
            )

            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])

            if not docs:
                return []

            sample_size = min(
                limit,
                len(docs)
            )

            selected_indexes = random.sample(
                range(len(docs)),
                sample_size
            )

            random_chunks: List[Dict[str, Any]] = []

            for index in selected_indexes:
                record_id = (
                    ids[index]
                    if index < len(ids)
                    else None
                )

                metadata = (
                    metas[index]
                    if index < len(metas) and metas[index]
                    else {}
                )

                random_chunks.append({
                    "id": record_id,
                    "content": docs[index],
                    "metadata": metadata
                })

            return random_chunks

        except Exception as e:
            logger.error(f"Error fetching random chunks: {e}")
            return []

    # ------------------------------------------------------------------
    # آمار و اطلاعات کلی
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        دریافت تعداد کل رکوردهای ذخیره‌شده.
        """
        return self.collection.count()

    def get_documents_summary(self) -> Dict[str, int]:
        """
        دریافت تعداد چانک‌های هر سند.

        نمونه خروجی:
        {
            "microsoft.txt": 56,
            "iot.txt": 45
        }
        """
        try:
            results = self.collection.get(
                include=["metadatas"]
            )

            metadatas = results.get(
                "metadatas",
                []
            )

            summary: Dict[str, int] = {}

            for metadata in metadatas:
                if not metadata:
                    continue

                source_name = metadata.get("source")

                if not source_name:
                    continue

                summary[source_name] = (
                    summary.get(source_name, 0) + 1
                )

            return summary

        except Exception as e:
            logger.error(f"Error getting documents summary: {e}")
            return {}

    # ------------------------------------------------------------------
    # دریافت چانک‌ها بر اساس منبع
    # ------------------------------------------------------------------

    def get_chunks_by_source(
        self,
        source_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        دریافت چانک‌های یک سند مشخص یا تمام چانک‌ها همراه با شناسه پایگاه داده.
        """
        try:
            include_fields = [
                "documents",
                "metadatas"
            ]

            if source_name:
                results = self.collection.get(
                    where={
                        "source": source_name
                    },
                    include=include_fields
                )
            else:
                results = self.collection.get(
                    include=include_fields
                )

            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])

            chunks: List[Dict[str, Any]] = []

            for index, doc in enumerate(docs):
                record_id = (
                    ids[index]
                    if index < len(ids)
                    else None
                )

                metadata = (
                    metas[index]
                    if index < len(metas) and metas[index]
                    else {}
                )

                chunks.append({
                    "id": record_id,
                    "content": doc,
                    "metadata": metadata
                })

            chunks.sort(
                key=lambda item: (
                    item.get("metadata", {})
                    .get("chunk_index", 0)
                )
            )

            return chunks

        except Exception as e:
            logger.error(f"Error fetching chunks for source '{source_name}': {e}")
            return []

    # ------------------------------------------------------------------
    # حذف چانک‌های یک سند مشخص (Single Source Deletion)
    # ------------------------------------------------------------------

    def delete_by_source(self, source_name: str) -> int:
        """
        حذف تمام چانک‌های مربوط به یک منبع/سند مشخص بر اساس متادیتای source.

        :param source_name: نام فایل یا شناسه منبع مورد نظر (مثال: 'report.txt')
        :return: تعداد رکوردهای حذف‌شده
        """
        if not source_name or not source_name.strip():
            logger.warning("delete_by_source called with empty source_name.")
            return 0

        target_source = source_name.strip()

        try:
            # ابتدا چک می‌کنیم چقدر چانک متعلق به این سند است
            existing_chunks = self.get_chunks_by_source(target_source)
            total_found = len(existing_chunks)

            if total_found == 0:
                logger.info(f"No chunks found for source: '{target_source}'")
                return 0

            # روش ۱: استفاده از شرط where در ChromaDB (سریع‌تر و استاندارد)
            try:
                self.collection.delete(where={"source": target_source})
                logger.info(
                    f"Successfully deleted {total_found} chunks "
                    f"for source: '{target_source}' via where clause."
                )
                return total_found
            except Exception as inner_e:
                logger.warning(
                    f"Direct where delete failed, falling back to ID list deletion: {inner_e}"
                )

            # روش ۲ (فال‌بک): حذف صریح بر اساس لیست شناسه‌ها (IDs)
            ids_to_delete = [
                item["id"]
                for item in existing_chunks
                if item.get("id")
            ]

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(
                    f"Successfully deleted {len(ids_to_delete)} chunks "
                    f"for source: '{target_source}' via ID list."
                )
                return len(ids_to_delete)

            return 0

        except Exception as e:
            logger.error(f"Failed to delete source '{target_source}': {e}", exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # حذف و ساخت مجدد Collection
    # ------------------------------------------------------------------

    def reset_collection(self) -> None:
        """
        حذف کامل Collection و ساخت مجدد آن.
        """
        collection_name = self.collection.name

        self.client.delete_collection(
            name=collection_name
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine"
                }
            )
        )
