import os
import re
import ollama
import chromadb

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5:7b"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "iot_knowledge"

# ۱. راه‌اندازی دیتابیس پایدار روی دیسک
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def chunk_text_safely(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    """
    تکه‌بندی هوشمند متن:
    متن ابتدا بر اساس پاراگراف‌ها جدا می‌شود و سپس در تکه‌های حداکثر ۱۲۰۰ کاراکتری
    با ۲۰۰ کاراکتر هم‌پوشانی (Overlap) قرار می‌گیرد تا از حد Context مدل رد نشود.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and not p.strip().startswith("---")]
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 <= max_chars:
            current_chunk += ("\n\n" + p if current_chunk else p)
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # حفظ هم‌پوشانی برای قطعه بعدی
                current_chunk = current_chunk[-overlap:] + "\n\n" + p
            else:
                # اگر یک پاراگراف به تنهایی از max_chars بزرگ‌تر بود
                chunks.append(p[:max_chars])
                current_chunk = p[max_chars - overlap:]

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# ۲. ایندکس‌گذاری در صورت خالی بودن کالکشن
if collection.count() == 0:
    print("در حال خواندن فایل iot.txt...")
    if not os.path.exists("iot.txt"):
        print("خطا: فایل iot.txt یافت نشد!")
        exit(1)

    with open("iot.txt", "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_text_safely(content, max_chars=1000, overlap=150)
    print(f"سند به {len(chunks)} قطعه استاندارد تقسیم شد. در حال ساخت امبدینگ...")

    for i, chunk in enumerate(chunks):
        res = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[res["embedding"]],
            documents=[chunk]
        )
        # نمایش روند ذخیره‌سازی
        print(f"قطعه {i + 1}/{len(chunks)} ذخیره شد.", end="\r")

    print(f"\nتمام {len(chunks)} قطعه با موفقیت ذخیره شدند!\n")
else:
    print(f"پایگاه برداری بارگذاری شد ({collection.count()} قطعه آماده جستجو).\n")

# ۳. حلقه پرسش و پاسخ
while True:
    query = input("\nسوال خود را بپرسید (برای خروج exit بزنید): ").strip()
    if query.lower() in ["exit", "quit", "خروج"]:
        break
    if not query:
        continue

    # الف) امبدینگ سوال
    query_emb = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]

    # ب) بازیابی برترین تکه‌ها
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )

    retrieved_docs = results["documents"][0]
    context = "\n\n---\n\n".join(retrieved_docs)

    # ج) پرامپت
    system_prompt = (
        "تو یک کارشناس و دستیار پاسخ‌گو بر اساس مستندات هستی.\n"
        "فقط و فقط با تکیه بر اطلاعات بخش [زمینه] و با زبان فارسی روان پاسخ بده.\n"
        "اگر جواب در زمینه وجود ندارد، صریحاً بگو «اطلاعاتی در این زمینه در سند موجود نیست».\n\n"
        f"[زمینه]:\n{context}"
    )

    # د) تولید پاسخ
    print("در حال دریافت پاسخ از مدل...")
    chat_response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    print("\nپاسخ:")
    print(chat_response["message"]["content"])
