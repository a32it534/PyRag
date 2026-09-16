from __future__ import annotations

import sys
import os
from typing import Annotated, Optional
import typer
from rich.console import Console
from rich.panel import Panel

from .chunking import chunk_text
from .config import get_settings
from .llm import LLMClient, LLMError, Message
from .rag import RAGPipeline
from .vectorstore import VectorStoreService

console = Console()
app = typer.Typer(add_completion=False, help="ابزار پیشرفته خط فرمان RAG و مدیریت مدل زبانی")

# پرامپت پیش‌فرض سخت‌گیرانه برای جلوگیری از توهم
DEFAULT_SYSTEM_PROMPT = (
    "شما یک دستیار متخصص هستید. شما فقط و فقط مجاز هستید از متنِ بازیابی شده (Context) برای پاسخگویی استفاده کنید. "
    "اگر پاسخ در متن وجود ندارد، دقیقاً بگویید 'اطلاعاتی در این زمینه در اسناد موجود نیست' و از دانش عمومی خود استفاده نکنید."
)

@app.command()
def ask(
    prompt: Annotated[str, typer.Argument(help="سوال یا متن مورد نظر برای ارسال به مدل")],
):
    """پرسش مستقیم از مدل زبانی (بدون بازیابی از اسناد)."""
    settings = get_settings()
    llm = LLMClient()
    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    console.print(f"[bold green]پرسش از مدل ({settings.default_model}):[/bold green]\n")
    try:
        for chunk in llm.stream_response(messages):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()
    except LLMError as e:
        console.print(f"[bold red]خطا در ارتباط با مدل:[/bold red] {e}")


@app.command("chat")
def chat(
    system_prompt: Annotated[
        str,
        typer.Option(
            "--system",
            "-s",
            help="دستورالعمل سیستمی برای تعیین هویت و رفتار مدل",
        ),
    ] = DEFAULT_SYSTEM_PROMPT,
    max_history: Annotated[
        int,
        typer.Option(
            "--max-history",
            "-m",
            help="حداکثر تعداد پیام‌های نگهداری‌شده",
        ),
    ] = 10,
    temperature: Annotated[
        float,
        typer.Option(
            "--temp",
            "-t",
            help="درجه خلاقیت مدل",
        ),
    ] = 0.0,
):
    """شروع گفتگوی تعاملی و پیوسته (Multi-turn RAG Chat)."""
    settings = get_settings()
    llm = LLMClient()
    rag = RAGPipeline()

    welcome_text = (
        f"[bold green]مدل فعال:[/bold green] {settings.default_model}\n"
        f"[bold cyan]حالت:[/bold cyan] RAG-Aware (جستجو در اسناد)\n"
        f"دستورات ویژه: /reset, /save, /stats, /exit"
    )
    console.print(Panel(welcome_text, title="🤖 نشست گفتگوی تعاملی (RAG)", border_style="blue"))

    messages: list[Message] = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = console.input("\n[bold cyan]👤 شما > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input: continue
        
        # مدیریت دستورات داخلی
        if user_input.lower() in {"/exit", "/quit"}: break
        if user_input.lower() in {"/reset", "/clear"}:
            messages = [{"role": "system", "content": system_prompt}]
            console.print("[yellow]🧹 حافظه پاک شد.[/yellow]")
            continue
        if user_input.lower() == "/stats":
            console.print(f"[dim]تعداد پیام‌های حافظه: {len(messages)}[/dim]")
            continue

        # --- بخش اصلی RAG ---
        context = rag.retrieve_context(user_input, top_k=3)
        augmented_input = f"متن مرجع (Context):\n{context}\n\nسوال کاربر: {user_input}"
        
        messages.append({"role": "user", "content": augmented_input})

        # محدودسازی تاریخچه
        if len(messages) > (max_history * 2) + 1:
            messages = [messages[0]] + messages[-(max_history * 2):]

        console.print("[bold green]🤖 دستیار > [/bold green]", end="")
        full_response: list[str] = []

        try:
            for chunk in llm.stream_response(messages, temperature=temperature):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                full_response.append(chunk)
            print()
            messages.append({"role": "assistant", "content": "".join(full_response)})
        except LLMError as e:
            console.print(f"\n[bold red]❌ خطا:[/bold red] {e}")
            messages.pop() 

@app.command()
def index(
    file_path: Annotated[str, typer.Argument(help="مسیر فایل متنی برای ایندکس")],
):
    """ایندکس کردن یک فایل در دیتابیس برداری."""
    if not os.path.exists(file_path):
        console.print(f"[bold red]فایل یافت نشد:[/bold red] {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    vector_service = VectorStoreService()
    vector_service.add_chunks(chunks)
    
    console.print(f"[bold green]✅ فایل با موفقیت ایندکس شد![/bold green]")
    console.print(f"تعداد چانک‌های ایجاد شده: {len(chunks)}")

@app.command()
def doctor():
    """بررسی سلامت سیستم و وضعیت دیتابیس."""
    console.print("[bold yellow]در حال بررسی سلامت سیستم...[/bold yellow]")
    try:
        vector_service = VectorStoreService()
        count = vector_service.collection.count()
        console.print(f"[green]✅ دیتابیس در دسترس است.[/green]")
        console.print(f"[blue]تعداد قطعات موجود در ChromaDB:[/blue] [bold]{count}[/bold]")
        
        if count == 0:
            console.print("[yellow]⚠️ دیتابیس خالی است. برای شروع، فایل‌ها را با دستور index ایندکس کنید.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]❌ خطا در اتصال به دیتابیس:[/bold red] {e}")

if __name__ == "__main__":
    app()
