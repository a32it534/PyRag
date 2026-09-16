import os
import re
import html
import sys
from pathlib import Path

# =========================================================
# 1. تنظیم مسیر پروژه
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
DOCUMENTS_DIR = ROOT_DIR / "documents"

for path in (str(ROOT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st

from ai_rag.ingest import IngestionService
from ai_rag.rag import RAGPipeline
from ai_rag.vectorstore import VectorStoreService


# =========================================================
# 2. تنظیمات صفحه
# =========================================================

st.set_page_config(
    page_title="دانش داده RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 3. طراحی رابط کاربری (RTL فوق‌حرفه‌ای، زیبا و پایدار)
# =========================================================

st.markdown(
    """
<style>
@import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

:root {
    --bg: #070b14;
    --panel: #0e1726;
    --panel2: #131f33;
    --border: rgba(148, 163, 184, 0.16);
    --border-hover: rgba(56, 189, 248, 0.4);
    --text: #e2e8f0;
    --muted: #94a3b8;
    --blue: #38bdf8;
    --purple: #a78bfa;
    --green: #34d399;
    --sidebar-width: 430px;
}

/* =========================================================
   پایه‌ریزی کامل راست به چپ (Full RTL System)
   ========================================================= */
html, body, .stApp {
    font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
    background-color: var(--bg);
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(59, 130, 246, 0.12), transparent 40%),
        radial-gradient(ellipse at 85% 15%, rgba(139, 92, 246, 0.12), transparent 40%),
        var(--bg);
}

/* پهنای استاندارد سایدبار */
[data-testid="stSidebar"], section[data-testid="stSidebar"] {
    width: var(--sidebar-width) !important;
    min-width: var(--sidebar-width) !important;
    max-width: var(--sidebar-width) !important;
    background: linear-gradient(180deg, #0b1322, #080d18) !important;
    border-left: 1px solid var(--border) !important;
    border-right: none !important;
    direction: rtl !important;
}

[data-testid="collapsedControl"] {
    right: 1.2rem !important;
    left: auto !important;
    direction: rtl !important;
    z-index: 1000;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
    direction: rtl !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1600px;
    direction: rtl !important;
    text-align: right !important;
}

/* =========================================================
   اصلاح تایپوگرافی، متون، لیست‌ها و خروجی مارک‌داون
   ========================================================= */
h1, h2, h3, h4, h5, h6, p, label, span, div,
.stMarkdown, .stCaption, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
}

/* اصلاح چیدمان لیست‌های بالت‌دار و شماره‌دار در فارسی */
div[data-testid="stMarkdownContainer"] ul,
div[data-testid="stMarkdownContainer"] ol {
    direction: rtl !important;
    text-align: right !important;
    padding-right: 1.8rem !important;
    padding-left: 0 !important;
    margin-right: 0 !important;
    margin-left: 0 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.8rem !important;
}

div[data-testid="stMarkdownContainer"] li {
    direction: rtl !important;
    text-align: right !important;
    margin-bottom: 0.5rem !important;
    line-height: 1.85 !important;
    color: #cbd5e1 !important;
}

div[data-testid="stMarkdownContainer"] ol > li::marker {
    color: var(--blue) !important;
    font-weight: 700 !important;
}

div[data-testid="stMarkdownContainer"] ul > li::marker {
    color: var(--purple) !important;
}

div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4 {
    direction: rtl !important;
    text-align: right !important;
    color: #f8fafc !important;
    font-weight: 750 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.6rem !important;
}

div[data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.75rem !important;
    margin-top: 0 !important;
    line-height: 1.85 !important;
    color: #e2e8f0;
}

/* اصلاح تگ‌ها و چیپ‌های مالتی‌سلکت */
[data-baseweb="tag"] {
    direction: rtl !important;
    background: rgba(56, 189, 248, 0.15) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    color: #7dd3fc !important;
    border-radius: 8px !important;
    padding: 2px 8px !important;
}

[data-baseweb="tag"] span {
    direction: rtl !important;
    font-family: 'Vazirmatn', sans-serif !important;
}

/* ورودی‌ها و انتخابگرها */
input, textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
    font-family: 'Vazirmatn', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
    border-radius: 10px !important;
}

[data-baseweb="select"] {
    direction: rtl !important;
    text-align: right !important;
}
[data-baseweb="select"] * {
    font-family: 'Vazirmatn', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] {
    direction: rtl !important;
    text-align: right !important;
}

/* اکاردئون‌ها (Expander) */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: rgba(14, 23, 38, 0.6) !important;
    overflow: hidden;
    margin-bottom: 12px;
}

[data-testid="stExpander"] summary {
    direction: rtl !important;
    text-align: right !important;
    padding: 12px 16px !important;
}

[data-testid="stExpander"] summary svg {
    margin-left: 10px !important;
    margin-right: 0 !important;
}

/* تب‌ها */
[data-baseweb="tab-list"] {
    direction: rtl !important;
    gap: 8px;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 4px;
}

[data-baseweb="tab"] {
    font-family: 'Vazirmatn', sans-serif !important;
    direction: rtl !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 10px 20px !important;
    color: var(--muted) !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    color: #fff !important;
    background: rgba(56, 189, 248, 0.08) !important;
    border-bottom: 2px solid var(--blue) !important;
}

/* دکمه‌ها */
.stButton > button, .stDownloadButton > button {
    font-family: 'Vazirmatn', sans-serif !important;
    direction: rtl !important;
    border-radius: 10px !important;
    min-height: 44px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--blue) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(56, 189, 248, 0.18);
}

button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important;
    color: #ffffff !important;
}

/* کارت‌های آماری */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #131f33, #0d1626) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 18px 20px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.88rem !important;
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 800 !important;
    direction: rtl !important;
    text-align: right !important;
}

/* چت باکس */
[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    margin-bottom: 14px !important;
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stChatMessage"] > div {
    flex-direction: row !important;
    gap: 14px !important;
}

[data-testid="stChatMessageContent"] {
    direction: rtl !important;
    text-align: right !important;
    width: 100% !important;
}

[data-testid="stChatInput"] {
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stChatInput"] textarea {
    direction: rtl !important;
    text-align: right !important;
    padding-left: 50px !important;
    padding-right: 16px !important;
}

[data-testid="stChatInput"] button {
    left: 12px !important;
    right: auto !important;
}

/* استایل‌های اختصاصی بنر و کارت‌ها */
.hero {
    position: relative;
    padding: 22px 28px;
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 20px;
    margin-bottom: 24px;
    background:
        radial-gradient(ellipse at 10% 0%, rgba(56, 189, 248, 0.16), transparent 50%),
        linear-gradient(135deg, rgba(17, 29, 49, 0.98), rgba(16, 21, 38, 0.96));
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
    direction: rtl;
    text-align: right;
}

.hero-title {
    color: #f8fafc;
    font-size: 1.85rem;
    font-weight: 800;
    margin-bottom: 0;
}

.section-heading {
    font-size: 1.22rem;
    font-weight: 800;
    color: #f1f5f9;
    margin: 12px 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.doc-card {
    background: linear-gradient(140deg, rgba(30, 41, 59, 0.65), rgba(15, 23, 42, 0.8));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    direction: rtl;
    text-align: right;
    transition: all 0.2s ease;
}

.doc-card:hover {
    border-color: var(--border-hover);
}

.doc-name {
    color: #e2e8f0;
    font-size: 0.9rem;
    font-weight: 600;
    overflow-wrap: anywhere;
    word-break: break-word;
}

.doc-meta {
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 6px;
}

.chunk-badge {
    display: inline-block;
    background: rgba(56, 189, 248, 0.12);
    color: #7dd3fc;
    border: 1px solid rgba(56, 189, 248, 0.24);
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 0.78rem;
    white-space: nowrap;
}

.source-card {
    background: rgba(11, 19, 33, 0.95);
    border: 1px solid rgba(56, 189, 248, 0.16);
    border-right: 4px solid var(--blue);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 10px 0 16px;
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.85;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    direction: rtl;
    text-align: right;
}

.chunk-card {
    background: #0b1220;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    color: #cbd5e1;
    line-height: 1.85;
    font-size: 0.92rem;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    direction: rtl;
    text-align: right;
}

.ideation-result-box {
    background: linear-gradient(145deg, rgba(17, 29, 49, 0.95), rgba(11, 19, 33, 0.98));
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-top: 20px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
    direction: rtl;
    text-align: right;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #6ee7b7;
    border-radius: 30px;
    font-size: 0.85rem;
}

.small-muted {
    color: #94a3b8;
    font-size: 0.85rem;
    direction: rtl;
    text-align: right;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}

.brand-icon {
    width: 46px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    font-size: 1.6rem;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.28);
}

.brand-name {
    color: #ffffff;
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.2px;
}

.brand-caption {
    color: #94a3b8;
    font-size: 0.74rem;
    font-weight: 500;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 4. توابع کمکی
# =========================================================

def safe_html(value):
    """جلوگیری از اجرای HTML ناخواسته در محتوای اسناد."""
    return html.escape(str(value or ""))


def safe_filename(name):
    """اعتبارسنجی نام فایل و جلوگیری از مسیرهای ناامن."""
    name = Path(str(name).strip()).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.strip(" .")

    if not name:
        raise ValueError("نام فایل معتبر نیست.")

    if len(name) > 180:
        raise ValueError("نام فایل بیش از حد طولانی است.")

    return name


def format_score(score):
    if score is None:
        return "نامشخص"

    try:
        return f"{float(score) * 100:.1f}%"
    except (ValueError, TypeError):
        return "نامشخص"


def get_source_name(source):
    metadata = source.get("metadata") or {}
    return metadata.get("source", "نامشخص")


def render_source(source, index):
    """نمایش منبع استنادی به صورت کاملاً راست‌چین و خوانا."""
    metadata = source.get("metadata") or {}
    source_name = get_source_name(source)
    chunk_index = metadata.get("chunk_index", "-")
    score = source.get("score")
    content = source.get("content", "")

    st.markdown(
        f"""
        <div class="small-muted">
            <b>📌 منبع {index}</b>
            &nbsp; | &nbsp; سند:
            <code>{safe_html(source_name)}</code>
            &nbsp; | &nbsp; چانک:
            <code>{safe_html(chunk_index)}</code>
            &nbsp; | &nbsp; شباهت معنایی:
            <b>{safe_html(format_score(score))}</b>
        </div>
        <div class="source-card">{safe_html(content)}</div>
        """,
        unsafe_allow_html=True,
    )


def get_documents_summary():
    try:
        return vector_store.get_documents_summary() or {}
    except Exception as exc:
        st.error(f"خطا در دریافت فهرست اسناد: {exc}")
        return {}


def save_and_index(text, filename):
    """ذخیره فایل و ارسال محتوا برای ایندکس."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    save_path = DOCUMENTS_DIR / filename

    if save_path.parent.resolve() != DOCUMENTS_DIR.resolve():
        raise ValueError("مسیر فایل نامعتبر است.")

    content_bytes = text.encode("utf-8")
    file_hash = ingest_service.calculate_file_hash(content_bytes)

    save_path.write_bytes(content_bytes)

    try:
        added = ingest_service.process_text_content(
            text=text,
            source_name=filename,
            file_hash=file_hash,
        )
    except Exception:
        raise

    return added


def render_sources(sources):
    if not sources:
        st.info("برای این پاسخ، منبع مرتبطی بازیابی نشده است.")
        return

    with st.expander(
        f"📚 مشاهده شواهد و منابع ({len(sources)})",
        expanded=False,
    ):
        for index, source in enumerate(sources, start=1):
            render_source(source, index)


# =========================================================
# 5. اتصال به سرویس‌های RAG
# =========================================================

@st.cache_resource
def get_services():
    vstore = VectorStoreService()
    rag = RAGPipeline(vector_store=vstore)
    ingest = IngestionService(vector_store=vstore)
    return rag, ingest, vstore


try:
    rag_pipeline, ingest_service, vector_store = get_services()
except Exception as exc:
    st.error(f"اتصال به سرویس‌های RAG با خطا مواجه شد: {exc}")
    st.stop()


# =========================================================
# 6. مدیریت Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_ideas" not in st.session_state:
    st.session_state.last_ideas = ""

if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False


# =========================================================
# 7. دریافت اطلاعات پایگاه دانش
# =========================================================

docs_summary = get_documents_summary()

total_docs = len(docs_summary)
total_chunks = (
    sum(docs_summary.values())
    if docs_summary
    else vector_store.count()
)


# =========================================================
# 8. نوار کناری مدیریتی
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">⚡</div>
            <div>
                <div class="brand-name">Rag Engine</div>
                <div class="brand-caption">سامانه هوشمند تحلیل و پردازش اسناد</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎛️ کنسول مدیریت")
    st.caption("مدیریت پایگاه دانش، اسناد و تنظیمات بهینه‌سازی مدل")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("تعداد کل اسناد", total_docs)

    with metric_col2:
        st.metric("تعداد چانک‌ها", total_chunks)

    st.markdown("---")

    # -----------------------------------------------------
    # تنظیمات بازیابی
    # -----------------------------------------------------

    with st.expander("⚙️ تنظیمات جستجوی معنایی", expanded=True):

        top_k = st.slider(
            "تعداد قطعات بازیابی (Top-K)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="تعداد چانک‌هایی که برای تولید پاسخ دقیق‌تر به مدل ارائه می‌شوند.",
        )

        min_score = st.slider(
            "حداقل آستانه شباهت",
            min_value=0.0,
            max_value=0.9,
            value=0.0,
            step=0.05,
            help="حداقل امتیاز شباهت جهت ورود قطعات به کانتکست مدل.",
        )

        doc_options = ["همه اسناد"] + sorted(docs_summary.keys())

        selected_search_doc = st.selectbox(
            "جستجو در سند مشخص",
            options=doc_options,
            index=0,
        )

        source_filter = (
            None
            if selected_search_doc == "همه اسناد"
            else selected_search_doc
        )

        st.caption(
            f"عمق بازیابی: {top_k} قطعه | "
            f"آستانه: {min_score:.0%}"
        )

    # -----------------------------------------------------
    # مدیریت اسناد
    # -----------------------------------------------------

    st.markdown("### 📚 اسناد پایگاه دانش")

    if docs_summary:

        for doc_name, chunk_count in sorted(docs_summary.items()):

            with st.container():

                st.markdown(
                    f"""
                    <div class="doc-card">
                        <div class="doc-name">
                            📄 {safe_html(doc_name)}
                        </div>
                        <div class="doc-meta">
                            <span class="chunk-badge">
                                {int(chunk_count)} چانک ثبت‌شده
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    "🗑️ حذف سند",
                    key=f"delete_{doc_name}",
                    use_container_width=True,
                ):
                    st.session_state[f"confirm_delete_{doc_name}"] = True

                if st.session_state.get(
                    f"confirm_delete_{doc_name}", False
                ):
                    st.warning(
                        f"آیا از حذف کامل «{doc_name}» اطمینان دارید؟"
                    )

                    confirm_col, cancel_col = st.columns(2)

                    with confirm_col:
                        if st.button(
                            "بله، حذف شود",
                            key=f"confirm_del_{doc_name}",
                            type="primary",
                            use_container_width=True,
                        ):
                            try:
                                with st.spinner("در حال حذف اطلاعات..."):
                                    deleted_count = (
                                        vector_store.delete_by_source(doc_name)
                                    )

                                    file_path = DOCUMENTS_DIR / safe_filename(
                                        doc_name
                                    )

                                    if file_path.exists():
                                        file_path.unlink()

                                st.session_state[
                                    f"confirm_delete_{doc_name}"
                                ] = False

                                st.toast(
                                    f"سند با موفقیت حذف شد؛ چانک‌های حذف‌شده: {deleted_count}",
                                    icon="🗑️",
                                )
                                st.rerun()

                            except Exception as exc:
                                st.error(f"خطا در فرآیند حذف: {exc}")

                    with cancel_col:
                        if st.button(
                            "انصراف",
                            key=f"cancel_del_{doc_name}",
                            use_container_width=True,
                        ):
                            st.session_state[
                                f"confirm_delete_{doc_name}"
                            ] = False
                            st.rerun()

    else:
        st.info("هیچ سندی در پایگاه دانش ثبت نشده است.")

    st.markdown("---")

    # -----------------------------------------------------
    # افزودن سند جدید
    # -----------------------------------------------------

    st.markdown("### ➕ افزودن محتوای جدید")

    upload_tab, text_tab = st.tabs(
        ["📤 آپلود فایل", "✍️ ثبت متن مستقیم"]
    )

    with upload_tab:

        uploaded_file = st.file_uploader(
            "فایل متنی را انتخاب نمایید",
            type=["txt", "md"],
            key="document_uploader",
            help="پشتیبانی از فایل‌های متنی استاندارد TXT و MD",
        )

        if uploaded_file is not None:

            st.caption(
                f"نام فایل: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)"
            )

            if st.button(
                "🚀 شروع پردازش و ایندکس",
                key="upload_index",
                type="primary",
                use_container_width=True,
            ):

                try:
                    filename = safe_filename(uploaded_file.name)

                    content_bytes = uploaded_file.getvalue()
                    content = content_bytes.decode(
                        "utf-8-sig",
                        errors="replace",
                    )

                    if not content.strip():
                        st.error("محتوای فایل خالی است یا ساختار متنی معتبری ندارد.")

                    else:
                        with st.spinner(
                            "در حال قطعه‌بندی، تولید امبدینگ و ثبت در پایگاه برداری..."
                        ):
                            added = save_and_index(content, filename)

                        if added > 0:
                            st.success(
                                f"فایل ایندکس شد؛ {added} چانک افزوده گردید."
                            )
                        else:
                            st.warning(
                                "محتوای فایل تکراری بوده یا چانک جدیدی ایجاد نشد."
                            )

                        st.rerun()

                except Exception as exc:
                    st.error(f"خطا در پردازش فایل: {exc}")

    with text_tab:

        with st.form("create_document_form"):

            new_doc_name = st.text_input(
                "عنوان سند",
                placeholder="مثال: مستندات_پروژه",
            )

            new_doc_ext = st.selectbox(
                "پسوند فایل",
                options=[".txt", ".md"],
            )

            new_doc_text = st.text_area(
                "متن محتوا",
                placeholder="متن فارسی یا انگلیسی خود را در این بخش بنویسید یا جای‌گذاری کنید...",
                height=180,
            )

            submitted = st.form_submit_button(
                "💾 ثبت و ایندکس در سیستم",
                type="primary",
                use_container_width=True,
            )

        if submitted:

            name = (new_doc_name or "").strip()
            text = (new_doc_text or "").strip()

            if not name:
                st.error("لطفاً یک عنوان برای سند تعیین کنید.")

            elif not text:
                st.error("لطفاً متن محتوا را وارد نمایید.")

            else:

                try:
                    filename = safe_filename(name)

                    if Path(filename).suffix.lower() not in (
                        ".txt",
                        ".md",
                    ):
                        filename += new_doc_ext

                    with st.spinner(
                        "در حال تولید وکتورها و ذخیره‌سازی متن..."
                    ):
                        added = save_and_index(text, filename)

                    if added > 0:
                        st.success(
                            f"سند با موفقیت ایجاد و تعداد {added} چانک ایندکس شد."
                        )
                    else:
                        st.warning(
                            "متن ثبت‌شده تکراری است یا چانک معتبری تولید نشد."
                        )

                    st.rerun()

                except Exception as exc:
                    st.error(f"خطا در ذخیره‌سازی سند: {exc}")

    st.markdown("---")

    # -----------------------------------------------------
    # ابزارهای مدیریتی
    # -----------------------------------------------------

    st.markdown("### 🛠️ ابزارهای مدیریتی")

    if st.button(
        "🧹 پاکسازی تاریخچه پیام‌ها",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    if st.button(
        "🔄 بازخوانی پایگاه دانش",
        use_container_width=True,
    ):
        st.rerun()

    if st.button(
        "⚠️ بازنشانی کامل پایگاه برداری",
        use_container_width=True,
    ):
        st.session_state.confirm_reset = True

    if st.session_state.confirm_reset:

        st.warning(
            "توجه: این عملیات تمامی بردارها و متادیتای ذخیره‌شده را برای همیشه حذف خواهد کرد."
        )

        reset_col1, reset_col2 = st.columns(2)

        with reset_col1:
            if st.button(
                "تأیید حذف کامل",
                type="primary",
                use_container_width=True,
            ):
                try:
                    vector_store.reset_collection()
                    st.session_state.messages = []
                    st.session_state.confirm_reset = False
                    st.success("مجموعه برداری با موفقیت ریست شد.")
                    st.rerun()

                except Exception as exc:
                    st.error(f"خطا در بازنشانی: {exc}")

        with reset_col2:
            if st.button(
                "انصراف",
                use_container_width=True,
            ):
                st.session_state.confirm_reset = False
                st.rerun()

    st.markdown("---")

    st.markdown(
        """
        <div class="small-muted" style="text-align: center !important;">
            <b>دانش داده RAG </b><br>
           
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 9. هدر و بنر داشبورد
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            ⚡ سامانه هوشمند مدیریت دانش و استناد (RAG)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 10. کارت‌های آماری داشبورد
# =========================================================

stats1, stats2, stats3, stats4 = st.columns(4)

with stats1:
    st.metric(
        "📚 تعداد اسناد ثبت‌شده",
        total_docs,
        help="تعداد کل فایل‌های متنی ثبت‌شده در پایگاه دانش",
    )

with stats2:
    st.metric(
        "🧩 قطعات دانش ایندکس‌شده",
        total_chunks,
        help="تعداد چانک‌های برداری آماده برای بازیابی معنایی",
    )

with stats3:
    st.metric(
        "💬 پیام‌های نشست فعلی",
        len(st.session_state.messages),
    )

with stats4:
    st.metric(
        "🔍 پارامتر Top-K",
        top_k,
        help="تعداد چانک‌های ارسال‌شده به پرامپت نهایی مدل",
    )


# =========================================================
# 11. تب‌های اصلی
# =========================================================

tab_chat, tab_ideas, tab_explorer = st.tabs(
    [
        "💬 گفتگوی هوشمند و استناد",
        "💡 آزمایشگاه ایده‌پردازی",
        "🔍 کاوشگر برداری و اسناد",
    ]
)


# =========================================================
# 12. گفتگوی هوشمند RAG
# =========================================================

with tab_chat:

    st.markdown(
        '<div class="section-heading">💬 دستیار پرسش و پاسخ اسناد</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "سؤالات تحلیلی خود را در ارتباط با اسناد مطرح کنید. مدل مستقیماً با استناد به منابع پاسخ خواهد داد."
    )

    if not docs_summary:
        st.info(
            "💡 پایگاه دانش خالی است. لطفاً ابتدا از نوار کناری، حداقل یک سند بارگذاری کنید."
        )

    if not st.session_state.messages:

        st.markdown(
            """
            <div class="doc-card" style="padding: 24px; border: 1px dashed rgba(56, 189, 248, 0.3);">
                <div style="font-size: 1.15rem; font-weight: 750; color: #f1f5f9;">
                    👋 به دستیار مدیریت دانش خوش آمدید
                </div>
                <div style="margin-top: 12px; color: #cbd5e1; line-height: 1.75;">
                    شما می‌توانید پرسش‌های خود را درباره محتوای اسناد مطرح کنید و مدل با استناد به منابع پاسخ خواهد داد.
                    <br><br>
                    برای شروع، کافیست در کادر پایین، سؤال خود را تایپ کرده و ارسال کنید.
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:

        role = message.get("role", "assistant")
        content = message.get("content", "")
        avatar = "👤" if role == "user" else "⚡"

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)

            if role == "assistant":
                render_sources(message.get("sources", []))

    user_input = st.chat_input(
        "پرسش خود را درباره محتوای اسناد بنویسید..."
    )

    if user_input:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="⚡"):

            response_placeholder = st.empty()
            full_response = ""
            sources = []

            try:

                with st.status(
                    "🔍 در حال کاوش و بازیابی بخش‌های مرتبط...",
                    expanded=False,
                ) as status:

                    context, sources = (
                        rag_pipeline.retrieve_context_with_sources(
                            query=user_input,
                            top_k=top_k,
                            source_filter=source_filter,
                            min_score=min_score,
                        )
                    )

                    if sources:
                        status.write(
                            f"تعداد {len(sources)} چانک مطابق با شرط شباهت بازیابی شد."
                        )
                    else:
                        status.write(
                            "موردی با آستانه شباهت فعلی در اسناد یافت نشد."
                        )

                    status.update(
                        label="بازیابی اطلاعات با موفقیت پایان یافت",
                        state="complete",
                        expanded=False,
                    )

                stream = rag_pipeline.query_stream(
                    question=user_input,
                    top_k=top_k,
                    source_filter=source_filter,
                    min_score=min_score,
                )

                for token in stream:
                    full_response += str(token)
                    response_placeholder.markdown(
                        full_response + "▌"
                    )

                response_placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                    }
                )

                render_sources(sources)

            except Exception as exc:

                error_message = (
                    "متأسفانه در جریان پردازش پرسش خطایی رخ داد. "
                    "وضعیت مدل، کانکشن یا تنظیمات دیتابیس را بررسی کنید."
                )

                st.error(error_message)
                st.exception(exc)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


# =========================================================
# 13. اتاق ایده‌پردازی اسناد (کاملاً راست‌چین و تفکیک‌شده)
# =========================================================

with tab_ideas:

    st.markdown(
        '<div class="section-heading">💡 آزمایشگاه ایده‌پردازی و توسعه راهکارها</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "با ترکیب چندین سند، ایده‌های جدید محصول، تحلیل و سناریوهای بازاریابی تولید کنید."
    )

    available_docs = sorted(
        get_documents_summary().keys()
    )

    if not available_docs:

        st.info(
            "برای فعال‌سازی ایده‌پردازی، ابتدا باید اسنادی در سیستم ثبت شده باشند."
        )

    else:

        with st.form("idea_generator_form"):

            selected_ideation_docs = st.multiselect(
                "📚 اسناد مبنا برای تحلیل",
                options=available_docs,
                default=(
                    available_docs[:2]
                    if len(available_docs) >= 2
                    else available_docs
                ),
                help="حداقل یک یا چند سند مرجع را انتخاب نمایید.",
            )

            idea_mode = st.selectbox(
                "🎯 مدل ذهنی / خروجی هدف",
                options=[
                    (
                        "product",
                        "🚀 ایده محصول و فیچرهای ارزش‌آفرین",
                    ),
                    (
                        "swot",
                        "📊 ماتریس نقاط قوت، ضعف، فرصت و تهدید (SWOT)",
                    ),
                    (
                        "content",
                        "📝 استراتژی محتوا و بازاریابی دانشی",
                    ),
                    (
                        "process",
                        "⚙️ بازطراحی و ارتقای فرآیندهای کاری",
                    ),
                    (
                        "custom",
                        "💬 هدف سفارشی طبق دستورالعمل",
                    ),
                ],
                format_func=lambda item: item[1],
            )

            custom_instructions = st.text_area(
                "دستورالعمل تکمیلی (پرامپت اختصاصی)",
                placeholder=(
                    "مثال: روی ایده‌هایی تمرکز کن که برای بازارهای B2B مناسب باشند و زمان پیاده‌سازی سریعی داشته باشند..."
                ),
                height=120,
            )

            generate_ideas = st.form_submit_button(
                "✨ تولید و استخراج ایده‌ها بر پایه دانش",
                type="primary",
                use_container_width=True,
            )

        if generate_ideas:

            if not selected_ideation_docs:
                st.error("لطفاً حداقل یک سند را به عنوان مرجع تعیین کنید.")

            else:

                idea_placeholder = st.empty()
                full_idea = ""

                try:

                    with st.status(
                        "🧠 در حال هم‌افزایی و ترکیب محتوای اسناد...",
                        expanded=True,
                    ) as status:

                        status.write(
                            f"تعداد اسناد مرجع در دست پردازش: {len(selected_ideation_docs)}"
                        )

                        stream = (
                            rag_pipeline.generate_ideas_stream(
                                selected_sources=selected_ideation_docs,
                                topic_or_goal=(
                                    custom_instructions.strip()
                                    or "ایده‌پردازی و تحلیل استراتژیک اسناد"
                                ),
                                mode=idea_mode[0],
                            )
                        )

                        for chunk in stream:
                            full_idea += str(chunk)

                            idea_placeholder.markdown(
                                full_idea + "▌"
                            )

                        status.update(
                            label="فرآیند استخراج ایده با موفقیت پایان یافت",
                            state="complete",
                            expanded=False,
                        )

                    idea_placeholder.markdown(full_idea)
                    st.session_state.last_ideas = full_idea

                except Exception as exc:
                    st.error("خطا در جریان تولید ایده.")
                    st.exception(exc)

    if st.session_state.last_ideas:

        st.markdown("---")
        st.markdown("### 📝 آخرین خروجی تولیدشده")

        st.download_button(
            label="📥 دریافت خروجی به صورت فایل متنی (TXT)",
            data=st.session_state.last_ideas,
            file_name="pyrag_ideas_report.txt",
            mime="text/plain;charset=utf-8",
            use_container_width=True,
        )

        with st.expander("مشاهده متن خام گزارش"):
            st.markdown(st.session_state.last_ideas)


# =========================================================
# 14. کاوشگر پایگاه دانش و چانک‌ها
# =========================================================

with tab_explorer:

    st.markdown(
        '<div class="section-heading">🔍 کاوشگر داده‌ها و بردارهای دیتابیس</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "بررسی جزئیات فنی چانک‌ها، متادیتاها، مقادیر هش و تست کیفیت قطعه‌بندی اسناد."
    )

    docs_summary = get_documents_summary()

    if not docs_summary:

        st.info("هیچ داده برداری برای نمایش در دیتابیس وجود ندارد.")

    else:

        explorer_col1, explorer_col2 = st.columns([3, 2])

        with explorer_col1:

            file_options = (
                ["همه فایل‌ها"]
                + sorted(docs_summary.keys())
            )

            selected_file = st.selectbox(
                "📂 فیلتر بر اساس نام سند",
                options=file_options,
            )

        with explorer_col2:

            search_chunk = st.text_input(
                "🔎 فیلتر زنده در متن قطعات",
                placeholder="کلمه یا عبارت مورد نظر...",
            )

        target_source = (
            None
            if selected_file == "همه فایل‌ها"
            else selected_file
        )

        try:
            chunks = vector_store.get_chunks_by_source(
                target_source
            ) or []
        except Exception as exc:
            st.error(f"خطا در واکشی چانک‌ها: {exc}")
            chunks = []

        if search_chunk.strip():
            needle = search_chunk.strip().casefold()
            chunks = [
                chunk
                for chunk in chunks
                if needle in str(chunk.get("content", "")).casefold()
            ]

        st.markdown("---")

        st.markdown(
            f"""
            <div class="status-pill">
                🧩 تعداد کل قطعات منطبق: {len(chunks)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not chunks:

            st.info("چانکی مطابق با فیلترهای اعمال‌شده پیدا نشد.")

        else:

            page_size = st.selectbox(
                "تعداد نمایش در هر صفحه",
                options=[10, 20, 50, 100],
                index=1,
            )

            total_pages = max(
                1,
                (len(chunks) + page_size - 1) // page_size,
            )

            page_number = st.number_input(
                "شماره صفحه",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
            )

            start_index = (page_number - 1) * page_size
            end_index = start_index + page_size

            visible_chunks = chunks[start_index:end_index]

            st.caption(
                f"نمایش صفحه {page_number} از {total_pages}"
            )

            for index, chunk in enumerate(
                visible_chunks,
                start=start_index + 1,
            ):

                metadata = chunk.get("metadata") or {}
                source = metadata.get("source", "نامشخص")
                chunk_index = metadata.get("chunk_index", index - 1)
                content = chunk.get("content", "") or ""
                chunk_id = chunk.get("id") or "N/A"
                file_hash = metadata.get("file_hash", "نامشخص")

                with st.expander(
                    f"🧩 چانک {chunk_index} | سند: {source} | ({len(content)} کاراکتر)",
                    expanded=False,
                ):

                    st.markdown(
                        f"""
                        <div class="doc-card">
                            <div class="doc-meta">🆔 شناسه بردار در دیتابیس</div>
                            <div style="overflow-wrap:anywhere;">
                                <code>{safe_html(chunk_id)}</code>
                            </div>

                            <div class="doc-meta" style="margin-top: 12px;">📄 نام منبع سند</div>
                            <div style="overflow-wrap:anywhere;">
                                <code>{safe_html(source)}</code>
                            </div>

                            <div class="doc-meta" style="margin-top: 12px;">🔑 هش اعتبارسنجی فایل</div>
                            <div style="overflow-wrap:anywhere;">
                                <code>{safe_html(file_hash)}</code>
                            </div>

                            <div class="doc-meta" style="margin-top: 12px;">📏 حجم قطعه</div>
                            <div>{len(content)} کاراکتر متنی</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"""
                        <div class="chunk-card">
                            {safe_html(content)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("🔬 متادیتای ساختاریافته (JSON)"):
                        st.json(metadata)

                    st.download_button(
                        label="📥 دانلود متن این چانک",
                        data=content,
                        file_name=f"chunk_{chunk_index}.txt",
                        mime="text/plain;charset=utf-8",
                        key=f"download_chunk_{index}_{chunk_id}",
                    )
