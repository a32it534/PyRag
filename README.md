# ⚡ PyRag — سیستم هوش مصنوعی محلی RAG

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)

**PyRag** یک سیستم سبک و ماژولار برای پیاده‌سازی **Retrieval-Augmented Generation (RAG)** با پایتون است.

این پروژه امکان دریافت اسناد، ایندکس کردن محتوا، جستجوی معنایی و پاسخ‌گویی مبتنی بر محتوای اسناد را فراهم می‌کند.

---

## ✨ امکانات

* 📄 ایندکس و پردازش اسناد
* ✂️ تقسیم‌بندی هوشمند متن
* 🔎 جستجوی معنایی با ChromaDB
* 🤖 اتصال به مدل‌های هوش مصنوعی مانند OpenAI / GapGPT
* 💬 پاسخ‌گویی مبتنی بر Context بازیابی‌شده
* 🌐 رابط کاربری وب با Streamlit
* 💻 رابط خط فرمان با Typer و Rich
* ⚡ پشتیبانی از پاسخ‌دهی Streaming
* 📊 بررسی وضعیت پایگاه داده و سیستم

---

## 🖥️ رابط‌های کاربری

### 🌐 Streamlit

رابط گرافیکی تحت وب برای گفتگو با اسناد و دریافت پاسخ‌های مبتنی بر RAG.

### 💻 CLI

رابط خط فرمان برای ایندکس اسناد، گفتگو، پرسش مستقیم و بررسی وضعیت سیستم.

---

## 🚀 نصب و اجرا

### 1. دریافت پروژه

```bash
git clone https://github.com/a32it534/PyRag.git
cd PyRag
```

### 2. ساخت محیط مجازی

```bash
python -m venv .venv
```

فعال‌سازی در ویندوز:

```bash
.venv\Scripts\activate
```

### 3. نصب وابستگی‌ها

```bash
pip install -e .
pip install chromadb streamlit
```

4. ایجاد فایل .env

در ویندوز:

copy .env.example .env

در Linux/macOS:

cp .env.example .env

### 4. تنظیم متغیرهای محیطی

فایل `.env.example` را به `.env` تبدیل کرده و اطلاعات API را وارد کنید.

نمونه:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_base_url
LLM_MODEL=your_model
EMBEDDING_MODEL=your_embedding_model
```

---

## 💻 استفاده

### اجرای رابط وب

```bash
streamlit run streamlit_app.py
```

### ایندکس کردن فایل

```bash
ai-rag index path/to/document.txt
```

### بررسی وضعیت سیستم

```bash
ai-rag doctor
```

### شروع گفتگوی RAG

```bash
ai-rag chat
```

### ارسال پرسش مستقیم

```bash
ai-rag ask "What is RAG?"
```

---

## 📂 ساختار پروژه

```text
PyRag/
├── src/
│   └── ai_rag/
│       ├── chunking.py       # تقسیم‌بندی متن
│       ├── cli.py            # رابط خط فرمان
│       ├── config.py         # تنظیمات پروژه
│       ├── ingest.py         # پردازش اسناد
│       ├── llm.py            # ارتباط با مدل هوش مصنوعی
│       ├── rag.py            # منطق RAG
│       └── vectorstore.py    # مدیریت ChromaDB
│
├── streamlit_app.py          # رابط وب
├── pyproject.toml            # تنظیمات و وابستگی‌ها
├── .env.example              # نمونه تنظیمات محیطی
└── README.md
```

---

## 🔄 روند کلی سیستم

```text
سند
 ↓
تقسیم‌بندی متن
 ↓
Embedding
 ↓
ChromaDB
 ↓
جستجوی معنایی
 ↓
بازیابی Context
 ↓
مدل هوش مصنوعی
 ↓
پاسخ
```

---

## 🛡️ مجوز

این پروژه تحت مجوز **MIT** منتشر شده است.

---
