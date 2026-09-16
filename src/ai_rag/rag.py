from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional, Tuple

from ai_rag.llm import LLMClient
from ai_rag.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """شما یک تحلیل‌گر و متخصص دقیق اسناد هستید.
وظیفه شما استخراج دقیق پاسخ پرسش کاربر با تکیه بر بخش «زمینه‌ی اطلاعاتی (Context)» ارائه شده است.

دستورالعمل‌ها و قواعد پاسخ‌دهی:
۱. متن اسناد را به دقت، موشکافانه و با در نظر گرفتن کلمات هم‌معنی و مفاهیم غیرمستقیم تحلیل کنید.
۲. اگر بخشی از پاسخ یا نکات مرتبط با سوال در زمینه وجود دارد، همان بخش را با شفافیت و ساختار مرتب (بولیت‌پوینت در صورت نیاز) توضیح دهید.
۳. تنها در صورتی که موضوع پرسش کاملاً بی‌ارتباط با متون ارائه شده بود و هیچ سرنخی یافت نشد، اعلام کنید: «با توجه به اسناد موجود، پاسخی برای این پرسش یافت نشد.»
۴. پاسخ باید به زبان فارسی روان، منسجم، مستند و بدون ابهام باشد.
"""

IDEATION_SYSTEM_PROMPT = """شما یک استراتژیست ارشد نوآوری، معمار راهکارهای سازمانی و مشاور توسعه محصول هستید.
وظیفه شما بررسی عمیق اسناد و شواهد ارائه‌شده و سنتز ایده‌های نوآورانه، کاربردی و واقع‌بینانه بر پایه آن مستندات است.

قواعد حیاتی:
۱. ایده‌ها باید مستقیماً از واقعیات، داده‌ها، فرآیندها یا خلأهای ذکر شده در اسناد سرچشمه بگیرند و کاملاً سفارشی‌سازی شده باشند (نه پیشنهادات کلیشه‌ای و عمومی).
۲. در هر ایده مشخص کنید دقیقاً کدام مفهوم یا نیاز مندرج در اسناد مبنای این پیشنهاد بوده است.
۳. لحن پاسخ باید مشاوره‌ای، قاطع، خلاق و به زبان فارسی استاندارد و ساختاریافته همراه با فرمت‌بندی Markdown باشد.
"""


class RAGPipeline:
    """
    خط لوله هوشمند RAG شامل بازیابی زمینه‌های مرتبط (با پشتیبانی از فیلتر سورس و آستانه امتیاز)،
    تولید پاسخ دقیق و موتور استراتژیک ایده‌پردازی بر اساس اسناد انتخابی.
    """

    IDEATION_PRESETS: Dict[str, str] = {
        "product": "طراحی محصولات و خدمات نوآورانه، تعریف فیچرهای ارزش‌آفرین و راهکارهای کاربردی",
        "swot": "تحلیل استراتژیک، کشف فرصت‌های بکر، نقاط آسیب‌پذیر، شکاف‌های عملیاتی و مزیت‌های رقابتی",
        "content": "طراحی استراتژی محتوا، روایت‌گری برند (Storytelling) و کمپین‌های بازاریابی هدفمند",
        "process": "بهینه‌سازی جریان کاری، کاهش هزینه‌ها، اتوماسیون فرآیندها و ارتقای بهره‌وری",
        "custom": "بررسی و ایده‌پردازی بر اساس دستور و هدایت سفارشی کاربر"
    }

    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.vector_store = vector_store or VectorStoreService()
        self.llm = llm_client or LLMClient()

    # =========================================================================
    # بخش بازیابی و پرسش‌وپاسخ استنادی (Standard Retrieval & QA)
    # =========================================================================

    def retrieve_context_with_sources(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        min_score: float = 0.0
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        بازیابی چانک‌های مرتبط بر اساس بردار پرسمان،
        همراه با شماره‌گذاری ساختاریافته، متادیتای منبع و فیلترهای هوشمند.
        """
        clean_query = query.strip()
        if not clean_query:
            return "", []

        items = self.vector_store.search_with_metadata(
            query=clean_query,
            top_k=top_k,
            source_filter=source_filter,
            min_score=min_score
        )

        if not items:
            return "", []

        context_blocks: List[str] = []
        for idx, item in enumerate(items, start=1):
            source_name = item.get("metadata", {}).get("source", "سند نامشخص")
            chunk_idx = item.get("metadata", {}).get("chunk_index", 0)
            score = item.get("score")
            score_str = f" | تطابق: {round(score * 100, 1)}%" if score is not None else ""

            content = item.get("content", "").strip()
            header = f"--- [بخش {idx} | منبع: {source_name} (چانک #{chunk_idx}){score_str}] ---"
            context_blocks.append(f"{header}\n{content}")

        full_context = "\n\n".join(context_blocks)
        return full_context, items

    def query_stream(
        self,
        question: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        min_score: float = 0.0
    ) -> Generator[str, None, None]:
        """اجرای پروسه RAG و استریم کردن پاسخ به صورت توکن به توکن."""
        context, sources = self.retrieve_context_with_sources(
            query=question,
            top_k=top_k,
            source_filter=source_filter,
            min_score=min_score
        )

        if not context.strip():
            yield "با توجه به اسناد موجود در پایگاه دانش، داده‌ای با ضریب اطمینان کافی برای پاسخ به این پرسش یافت نشد."
            return

        user_prompt = f"""زمینه‌ی اطلاعاتی استخراج شده از اسناد:
{context}

==============================
پرسش کاربر:
{question}
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        yield from self.llm.stream_response(messages=messages)

    def query(
        self,
        question: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        min_score: float = 0.0
    ) -> Dict[str, Any]:
        """متد غیر استریمینگ برای دریافت یکپارچه پاسخ به همراه منابع."""
        context, sources = self.retrieve_context_with_sources(
            query=question,
            top_k=top_k,
            source_filter=source_filter,
            min_score=min_score
        )

        if not context.strip():
            return {
                "answer": "با توجه به اسناد موجود در پایگاه دانش، داده‌ای با ضریب اطمینان کافی برای پاسخ به این پرسش یافت نشد.",
                "sources": []
            }

        user_prompt = f"""زمینه‌ی اطلاعاتی استخراج شده از اسناد:
{context}

==============================
پرسش کاربر:
{question}
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            answer = self.llm.generate(prompt=user_prompt, system_message=SYSTEM_PROMPT)
        except Exception:
            answer = "".join(list(self.llm.stream_response(messages=messages)))

        return {
            "answer": answer,
            "sources": sources
        }

    # =========================================================================
    # بخش موتور ایده‌پردازی بر پایه اسناد انتخابی (Smart Ideation Engine)
    # =========================================================================

    def _build_ideation_context(
        self,
        selected_sources: List[str],
        max_chunks_per_doc: int = 15
    ) -> Tuple[str, int]:
        """
        گردآوری و ساختاربندی هوشمند چانک‌های اسناد انتخاب‌شده برای ایده‌پردازی.
        """
        combined_blocks: List[str] = []
        total_used_chunks = 0

        for source_name in selected_sources:
            chunks = self.vector_store.get_chunks_by_source(source_name)
            if not chunks:
                continue

            # انتخاب تا سقف مجاز چانک‌ها جهت رعایت پنجره کانتکست
            selected_subset = chunks[:max_chunks_per_doc]
            doc_texts = []
            for item in selected_subset:
                content = (item.get("content") or "").strip()
                if content:
                    doc_texts.append(content)
                    total_used_chunks += 1

            if doc_texts:
                combined_blocks.append(
                    f"### 📂 سند مرجع: {source_name}\n" + "\n---\n".join(doc_texts)
                )

        return "\n\n====================\n\n".join(combined_blocks), total_used_chunks

    def _build_ideation_prompt(
        self,
        context: str,
        topic_or_goal: str,
        mode: str = "product"
    ) -> str:
        """ساخت پرامپت مهندسی‌شده ایده‌پردازی با رعایت خروجی ساختاریافته."""
        mode_label = self.IDEATION_PRESETS.get(mode, self.IDEATION_PRESETS["product"])
        focus_detail = topic_or_goal.strip() if topic_or_goal.strip() else "استخراج حداکثر ارزش و ایده‌های متمایز"

        return f"""شما باید بر اساس تمام داده‌ها و مستندات منبع زیر، خروجی ایده‌پردازی تحلیلی و راهبردی ارائه دهید.

🎯 ماموریت و سبک ایده: {mode_label}
🔍 حوزه تمرکز ویژه یا نیازمندی کاربر: {focus_detail}

📚 مستندات زمینه و پایگاه داده انتخابی:
{context}

==================================================
قالب خروجی الزامی:
لطفاً ۳ الی ۵ ایده شاخص و باکیفیت را با ساختار کاملاً مشخص زیر تدوین کنید:

### 💡 ایده شماره [شماره]: [عنوان رسا و حرفه‌ای]
- 🎯 **مفهوم و شرح عملیاتی:** تشریح دقیق ایده و نحوه حل مسئله.
- 📌 **استناد به اسناد:** این ایده بر پایه کدام بخش یا دیتای موجود در اسناد شکل گرفته است؟
- 💎 **ارزش پیشنهادی و مزیت متمایز (Value Proposition):** چه تفاوتی رقم می‌زند؟
- 🛠️ **گام‌های کلیدی اجرای اولیه (Roadmap / MVP):** ۲ تا ۳ اقدام مرحله اول برای پیاده‌سازی.

در پایان، یک بخش کوتاه با عنوان «🔮 افق توسعه و توصیه‌های تکمیلی» اضافه کنید.
"""

    def generate_ideas_stream(
        self,
        selected_sources: List[str],
        topic_or_goal: str = "",
        mode: str = "product",
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """
        ایجاد ایده‌های خلاقانه و راهبردی از اسناد انتخابی به صورت لحظه‌ای (Stream).
        """
        if not selected_sources:
            yield "⚠️ هیچ سندی انتخاب نشده است. لطفاً حداقل یک سند را انتخاب نمایید."
            return

        context, total_chunks = self._build_ideation_context(selected_sources)
        if not context:
            yield "⚠️ محتوایی برای اسناد انتخاب‌شده در پایگاه داده یافت نشد."
            return

        user_prompt = self._build_ideation_prompt(
            context=context,
            topic_or_goal=topic_or_goal,
            mode=mode
        )

        messages = [
            {"role": "system", "content": IDEATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        logger.info(
            f"Starting ideation stream for {len(selected_sources)} sources ({total_chunks} chunks) in mode '{mode}'."
        )
        yield from self.llm.stream_response(messages=messages, temperature=temperature)

    def generate_ideas(
        self,
        selected_sources: List[str],
        topic_or_goal: str = "",
        mode: str = "product",
        temperature: float = 0.7
    ) -> str:
        """نسخه غیر استریمینگ متد ایده‌پردازی."""
        stream = self.generate_ideas_stream(
            selected_sources=selected_sources,
            topic_or_goal=topic_or_goal,
            mode=mode,
            temperature=temperature
        )
        return "".join(list(stream))
