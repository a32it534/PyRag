from __future__ import annotations

import re
from typing import List


def _find_best_split_point(text: str, start: int, end: int, min_split_pos: int) -> int:
    """
    یافتن بهترین نقطه برش بر اساس اولویت معنایی:
    ۱. مرز پاراگراف (\n\n)
    ۲. مرز خطوط (\n)
    ۳. مرز پایان جمله (نقطه، علامت سوال، تعجب، نیم‌کالن)
    ۴. ویرگول و کاما (، ,)
    ۵. در نهایت فاصله سفید (Space)
    """
    # فهرست جداکننده‌ها بر اساس اولویت معنایی و حفظ ساختار جمله
    separators_priority = [
        ["\n\n", "\r\n\r\n"],
        ["\n", "\r\n"],
        [". ", "؟ ", "! ", "؛ ", ".\n", "؟\n", "!\n"],
        ["، ", ", "],
        [" "],
    ]

    for sep_group in separators_priority:
        best_found = -1
        best_sep_len = 0
        
        for sep in sep_group:
            found = text.rfind(sep, min_split_pos, end)
            if found > best_found:
                best_found = found
                best_sep_len = len(sep)

        if best_found != -1:
            return best_found + best_sep_len

    return -1


def chunk_text(
    text: str,
    max_chars: int = 1000,
    overlap: int = 150
) -> List[str]:
    """
    تکه‌بندی سلسله‌مراتبی (Recursive/Semantic-aware) با در نظر گرفتن ساختار متن فارسی:
    - حفظ کلمات و جملات کامل بدون قطع شدن تصادفی
    - همپوشانی هوشمند با شروع از ابتدای کلمات (Word-boundary Safe)
    - پاکسازی فاصله‌ها و کاراکترهای اضافه
    """
    if not text:
        return []

    # نرمال‌سازی اولیه خطوط و فاصله‌های زاید
    cleaned_text = re.sub(r"\r\n?", "\n", text).strip()
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    if not cleaned_text:
        return []

    # اگر کل متن از سقف مجاز کمتر بود
    if len(cleaned_text) <= max_chars:
        return [cleaned_text]

    chunks: List[str] = []
    start = 0
    text_length = len(cleaned_text)

    # تعیین حداقل بازه مجاز برای برش (حداقل ۶۰٪ طول چانک طی شده باشد)
    min_split_ratio = 0.6

    while start < text_length:
        end = start + max_chars

        if end < text_length:
            min_split_pos = start + int(max_chars * min_split_ratio)
            split_idx = _find_best_split_point(cleaned_text, start, end, min_split_pos)

            # در صورتی که مرز معنایی مناسب پیدا نشد، به اولین فاصله قبل از end بسنده می‌کنیم
            if split_idx == -1:
                last_space = cleaned_text.rfind(" ", min_split_pos, end)
                if last_space != -1:
                    split_idx = last_space + 1
                else:
                    split_idx = end

            end = split_idx

        # استخراج چانک و اعتبارسنجی
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # محاسبه نقطه شروع بعدی با در نظر گرفتن اوورلپ
        next_start = max(start + 1, end - overlap)

        # اطمینان از اینکه شروع چانک بعدی وسط یک کلمه نباشد
        if next_start < text_length and not cleaned_text[next_start].isspace():
            next_space = cleaned_text.find(" ", next_start, min(next_start + 40, end))
            if next_space != -1:
                next_start = next_space + 1

        start = next_start

    return chunks
