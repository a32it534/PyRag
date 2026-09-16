from __future__ import annotations

import re
from typing import TypedDict, Literal, Iterator, List, Optional
from openai import OpenAI, OpenAIError
from ai_rag.config import get_settings


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(Exception):
    """خطای سفارشی برای مدیریت مشکلات ارتباط با سرویس‌های هوش مصنوعی"""
    pass


class BaseOpenAIClient:
    """کلاس پایه برای مدیریت اتصال امن به کلاینت OpenAI و پرووایدرهای سازگار"""
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url

        if not self.api_key:
            raise LLMError("کلید OPENAI_API_KEY یافت نشد. لطفاً فایل .env را بررسی کنید.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )


class LLMClient(BaseOpenAIClient):
    """سرویس مدیریت چت و پاسخ‌دهی استریمینگ با بیشترین دقت و تعهد به Context"""
    def __init__(self, model: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        settings = get_settings()
        self.model = model or settings.default_model

    def generate(
        self,
        prompt: str,
        system_message: str = "شما یک دستیار هوشمند، دقیق و تحلیل‌گر اسناد هستید.",
        temperature: float = 0.0
    ) -> str:
        """پاسخ متنی یکپارچه با دمای صفر برای حذف توهم و افزایش حداکثری دقت استناد"""
        try:
            messages: List[Message] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except OpenAIError as e:
            raise LLMError(f"خطا در دریافت پاسخ از مدل: {e}") from e

    def stream_response(
        self,
        messages: List[Message],
        temperature: float = 0.0
    ) -> Iterator[str]:
        """پاسخ زنده استریم با کمترین تصادفی‌سازی جهت پایبندی کامل به متن چانک‌ها"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=temperature,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except OpenAIError as e:
            raise LLMError(f"خطا در استریمینگ پاسخ: {e}") from e


class EmbeddingService(BaseOpenAIClient):
    """
    سرویس اختصاصی برداری‌سازی با پیش‌پردازش استاندارد زبان فارسی،
    بچ‌بندی ایمن (Batching) و جلوگیری از کاهش دقت در جستجوی شباهت
    """
    MAX_CHAR_LIMIT: int = 5000 
    BATCH_SIZE: int = 32        # سایز بهینه هر دسته جهت جلوگیری از Timeout و خطای Payload

    def __init__(self, model: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        settings = get_settings()
        self.model = model or settings.embedding_model

    @staticmethod
    def normalize_persian_text(text: str) -> str:
        """یکسان‌سازی کاراکترهای عربی/فارسی و فاصله‌ها جهت افزایش تطابق سمانتیک"""
        if not text:
            return ""
        # تبدیل انواع ک و ی عربی به فارسی
        text = text.replace("ي", "ی").replace("ك", "ک")
        # حذف تشدید و حرکات اعراب
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
        # یکسان‌سازی فاصله‌ها و تب‌ها
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _sanitize_text(self, text: str) -> str:
        """
        پاکسازی، نرمال‌سازی و برش در مرز کلمات:
        مانع از شکستن ناقص کلمات یا ارسال چانک‌های خراب به مدل برداری می‌شود.
        """
        text = self.normalize_persian_text(str(text or ""))

        if len(text) > self.MAX_CHAR_LIMIT:
            truncated = text[: self.MAX_CHAR_LIMIT]
            last_space = truncated.rfind(" ")
            if last_space > int(self.MAX_CHAR_LIMIT * 0.8):
                truncated = truncated[:last_space]
            text = truncated.strip()

        # اگر بعد از نرمال‌سازی خالی شد، از کاراکتر خنثی استفاده می‌شود تا امبدینگ مختل نشود
        return text if text else "محتوای نامشخص"

    def embed_text(self, text: str) -> List[float]:
        """تولید امبدینگ برای یک متن تکی (نرمال‌سازی شده)"""
        clean_text = self._sanitize_text(text)
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[clean_text],
            )
            return response.data[0].embedding
        except OpenAIError as e:
            raise LLMError(f"خطا در تولید بردار امبدینگ: {e}") from e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        تولید همزمان امبدینگ با تقسیم‌بندی دسته‌ای (Chunked Batches)
        برای پایداری بالا، رفع خطای محدودیت Payload و حفظ ترتیبی بردارها
        """
        if not texts:
            return []

        safe_texts = [self._sanitize_text(t) for t in texts]
        all_embeddings: List[List[float]] = []

        # پردازش در دسته‌های BATCH_SIZE تایی
        for start_idx in range(0, len(safe_texts), self.BATCH_SIZE):
            batch = safe_texts[start_idx : start_idx + self.BATCH_SIZE]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                # مرتب‌سازی تضمینی داده‌ها مطابق ایندکس ورودی
                sorted_data = sorted(response.data, key=lambda x: x.index)
                all_embeddings.extend([item.embedding for item in sorted_data])

            except OpenAIError:
                # فال‌بک تک‌به‌تک اختصاصی در صورت خطای گروهی
                for sub_idx, single_text in enumerate(batch):
                    try:
                        resp = self.client.embeddings.create(
                            model=self.model,
                            input=[single_text],
                        )
                        all_embeddings.append(resp.data[0].embedding)
                    except OpenAIError as single_err:
                        actual_idx = start_idx + sub_idx
                        raise LLMError(
                            f"خطا در تولید امبدینگ برای چانک {actual_idx}: {single_err}"
                        ) from single_err

        return all_embeddings
