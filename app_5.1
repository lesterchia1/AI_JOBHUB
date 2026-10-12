"""
AI Job Hunt Hub — v5.1 Build
Streamlit app that tracks job applications and uses AI (Groq + Sea Lion)
to tailor resumes and generate cover letters.

Now featuring:
- Supabase REST API persistent cloud storage (bypassing TCP pooler errors)
- Human-in-the-loop editing for resume & cover letter
- PDF and DOCX export
- LLM Monitoring dashboard

Run:
    streamlit run app.py
"""

import os
import time
import io
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq
from openai import OpenAI

# Document generation
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from docx import Document

# Rich text editor
from streamlit_lexical_extended import streamlit_lexical_extended

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Job Hunt Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ============================================================
# 🚀 API KEY CONFIGURATION
# ============================================================

def _get_secret(key: str, default=None):
    try:
        from google.colab import userdata
        val = userdata.get(key)
        if val:
            return val
    except ImportError:
        pass

    val = os.environ.get(key)
    if val:
        return val

    try:
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass

    return default

GROQ_API_KEY    = _get_secret("GROQ_API_KEY")
SEALION_API_KEY = _get_secret("SEALION_API_KEY")
HF_TOKEN        = _get_secret("HF_TOKEN")

# ============================================================
# 👤 USER ID CONFIGURATION
# ============================================================

USER_ID = _get_secret("USER_ID", "adaae0b2-99ef-4524-a932-90d57b3b4945")

# ============================================================
# 🗄️ DATABASE CONFIGURATION (SUPABASE HTTP REST CLIENT)
# ============================================================

SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error(
        "❌ SUPABASE_URL or SUPABASE_KEY is missing. "
        "Please add your Supabase REST API credentials to `.streamlit/secrets.toml`."
    )
    st.stop()

@st.cache_resource
def get_supabase_client() -> Client:
    # Uses HTTPS (Port 443) web endpoints to bypass TCP socket pooler errors completely
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = get_supabase_client()

# ============================================================
# 🚀 API KEY VALIDATION
# ============================================================

if not GROQ_API_KEY and not SEALION_API_KEY:
    st.error(
        "❌ No API keys found. Set at least one of GROQ_API_KEY or SEALION_API_KEY "
        "in your `.env` file or Streamlit secrets."
    )
    st.stop()

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SEALION_BASE_URL = "https://api.sea-lion.ai/v1"
sealion_client = (
    OpenAI(api_key=SEALION_API_KEY, base_url=SEALION_BASE_URL)
    if SEALION_API_KEY else None
)

# Constants -----------------------------------------------------------------

STAGES = [
    {"id": "Applied",      "name": "Applied",      "color": "#3498db"},
    {"id": "Acknowledged", "name": "Acknowledged", "color": "#f39c12"},
    {"id": "Screening",    "name": "Screening",    "color": "#9b59b6"},
    {"id": "Interviewing", "name": "Interviewing", "color": "#e67e22"},
    {"id": "Offer",        "name": "Offer",        "color": "#27ae60"},
    {"id": "Rejected",     "name": "Rejected",     "color": "#e74c3c"},
]

# ============================================================
# 🧠 MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {
    "Sea-Lion v4 27B": {
        "provider": "sealion",
        "model_id": "aisingapore/Gemma-SEA-LION-v4-27B-IT",
        "description": "Gemma 3 27B fine-tuned for SEA languages by AI Singapore.",
    },
    "Sea-Lion v4 32B (Qwen)": {
        "provider": "sealion",
        "model_id": "aisingapore/Qwen-SEA-LION-v4-32B-IT",
        "description": "Qwen3-32B fine-tuned for SEA tasks.",
    },
    "Groq: GPT-OSS 120B": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "description": "OpenAI's open-weight MoE, 120B params, reasoning-capable.",
    },
    "Groq: GPT-OSS 20B": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "description": "Smaller, faster OpenAI open-weight model.",
    },
}

DEFAULT_MODEL_KEY = "Sea-Lion v4 27B"

# ============================================================
# 💰 MODEL PRICING (per 1M tokens, USD)
# ============================================================

MODEL_PRICING = {
    "openai/gpt-oss-120b": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60,
        "context_window": 131072,
        "max_output": 65536,
    },
    "openai/gpt-oss-20b": {
        "input": 0.075,
        "cached_input": 0.037,
        "output": 0.30,
        "context_window": 131072,
        "max_output": 65536,
    },
    "aisingapore/Gemma-SEA-LION-v4-27B-IT": {
        "input": 0.0,
        "cached_input": 0.0,
        "output": 0.0,
        "context_window": 131072,
        "max_output": 65536,
    },
    "aisingapore/Qwen-SEA-LION-v4-32B-IT": {
        "input": 0.0,
        "cached_input": 0.0,
        "output": 0.0,
        "context_window": 131072,
        "max_output": 65536,
    },
}

def get_pricing(model_id: str) -> dict:
    return MODEL_PRICING.get(
        model_id,
        {"input": 0.0, "output": 0.0, "cached_input": 0.0,
         "context_window": 0, "max_output": 0},
    )

def compute_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    p = get_pricing(model_id)
    return (
        (input_tokens / 1_000_000) * p["input"]
        + (output_tokens / 1_000_000) * p["output"]
    )

# ---------------------------------------------------------------------------
# 3. AI LAYER
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"Embedding model unavailable: {e}")
        return None

def get_active_model() -> tuple[str, str, str]:
    key = st.session_state.get("selected_model_key", DEFAULT_MODEL_KEY)
    entry = MODEL_REGISTRY.get(key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])
    return entry["provider"], entry["model_id"], key

def load_llm_for(provider: str, model_id: str):
    if provider == "groq":
        if not GROQ_API_KEY:
            return None, "Groq API key not set"
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model=model_id, temperature=0.7, api_key=GROQ_API_KEY)
            return llm, f"Groq ({model_id})"
        except Exception as e:
            return None, f"Groq init failed: {e}"

    if provider == "sealion":
        if not SEALION_API_KEY:
            return None, "Sea Lion API key not set"
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_id, temperature=0.7,
                api_key=SEALION_API_KEY, base_url=SEALION_BASE_URL,
            )
            return llm, f"Sea Lion ({model_id})"
        except Exception as e:
            return None, f"Sea Lion init failed: {e}"

    return None, f"Unknown provider: {provider}"

def _extract_usage_from_langchain(result) -> tuple[int, int]:
    um = getattr(result, "usage_metadata", None)
    if um:
        return int(um.get("input_tokens", 0)), int(um.get("output_tokens", 0))
    rm = getattr(result, "response_metadata", None) or {}
    tu = rm.get("token_usage") or rm.get("usage") or {}
    return int(tu.get("prompt_tokens", 0)), int(tu.get("completion_tokens", 0))

def svc_log_llm_usage(
    *, provider, model_key, model_id, operation,
    input_tokens, output_tokens, latency_seconds,
    success=True, error_message="",
):
    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    cost = compute_cost_usd(model_id, input_tokens, output_tokens) if success else 0.0

    payload = {
        "user_id": USER_ID,
        "called_at": datetime.now().isoformat(),
        "provider": provider,
        "model_key": model_key,
        "model_id": model_id,
        "operation": operation,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency_seconds": float(latency_seconds or 0.0),
        "cost_usd": cost,
        "success": 1 if success else 0,
        "error_message": error_message or "",
    }
    try:
        supabase.table("llm_usage").insert(payload).execute()
    except Exception:
        pass

def calculate_match_score(resume_text: str, job_description: str) -> float:
    if not resume_text or not job_description:
        return 0.0
    model = load_embedding_model()
    if model is None:
        return 0.0
    from sentence_transformers import util
    emb = model.encode([job_description, resume_text], convert_to_tensor=True)
    score = util.cos_sim(emb[0], emb[1])
    return float(score[0][0])

def call_llm(prompt: str, operation: str = "generic") -> str:
    provider, model_id, label = get_active_model()
    llm, status = load_llm_for(provider, model_id)
    if llm is None:
        svc_log_llm_usage(
            provider=provider, model_key=label, model_id=model_id,
            operation=operation, input_tokens=0, output_tokens=0,
            latency_seconds=0.0, success=False, error_message=str(status),
        )
        return f"⚠️ {status}"
    start = time.time()
    try:
        result = llm.invoke(prompt)
        latency = time.time() - start
        in_tok, out_tok = _extract_usage_from_langchain(result)
        svc_log_llm_usage(
            provider=provider, model_key=label, model_id=model_id,
            operation=operation, input_tokens=in_tok, output_tokens=out_tok,
            latency_seconds=latency, success=True,
        )
        return getattr(result, "content", str(result))
    except Exception as e:
        latency = time.time() - start
        svc_log_llm_usage(
            provider=provider, model_key=label, model_id=model_id,
            operation=operation, input_tokens=0, output_tokens=0,
            latency_seconds=latency, success=False, error_message=str(e),
        )
        return f"⚠️ LLM call failed ({status}): {e}"

def tailor_resume(resume_text: str, job_description: str) -> str:
    prompt = f"""You are an expert resume writer.
Rewrite the resume below to emphasize experience relevant to the job description.
Use quantifiable achievements and keywords from the job posting.
Keep the format clean and professional. Do NOT invent experience.

JOB DESCRIPTION:
{job_description}

CURRENT RESUME:
{resume_text}

TAILORED RESUME:"""
    return call_llm(prompt, operation="tailor_resume")

def generate_cover_letter(resume_text: str, job_description: str, company: str, role: str) -> str:
    prompt = f"""Write a professional, concise cover letter for the {role} role at {company}.
Ground it in the resume and job description below. Do NOT use placeholders like [Your Name].

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

COVER LETTER:"""
    return call_llm(prompt, operation="cover_letter")

def extract_pdf_text(uploaded_file) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        return "".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

# ---------------------------------------------------------------------------
# 3b. DOCUMENT EXPORT HELPERS
# ---------------------------------------------------------------------------

def generate_pdf_bytes(text: str, filename: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = [Paragraph(text, styles["Normal"])]
    doc.build(story)
    return buffer.getvalue()

def generate_docx_bytes(text: str) -> bytes:
    doc = Document()
    for line in text.split('\n'):
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

# ---------------------------------------------------------------------------
# 4. SERVICE LAYER (SUPABASE REST API)
# ---------------------------------------------------------------------------

def svc_list_applications():
    try:
        res = supabase.table("applications").select("*").order("last_updated", desc=True).execute()
        return res.data if res.data else []
    except Exception as e:
        st.error(f"Error loading applications: {e}")
        return []

def svc_get_application(app_id: int):
    try:
        res = supabase.table("applications").select("*").eq("id", app_id).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        st.error(f"Error fetching application: {e}")
        return None

def svc_create_application(**kwargs):
    kwargs['user_id'] = USER_ID
    if 'applied_date' in kwargs and isinstance(kwargs['applied_date'], datetime):
        kwargs['applied_date'] = kwargs['applied_date'].isoformat()
    if 'last_updated' in kwargs and isinstance(kwargs['last_updated'], datetime):
        kwargs['last_updated'] = kwargs['last_updated'].isoformat()
    
    res = supabase.table("applications").insert(kwargs).execute()
    if res.data:
        return res.data[0]['id']
    return None

def svc_update_status(app_id: int, new_status: str):
    res = supabase.table("applications").update({
        "status": new_status,
        "last_updated": datetime.now().isoformat()
    }).eq("id", app_id).execute()
    return bool(res.data)

def svc_update_application_text(app_id: int, resume_text: str, cover_letter: str):
    res = supabase.table("applications").update({
        "resume_text": resume_text,
        "cover_letter": cover_letter,
        "last_updated": datetime.now().isoformat()
    }).eq("id", app_id).execute()
    return bool(res.data)

def svc_delete_application(app_id: int):
    supabase.table("applications").delete().eq("id", app_id).execute()

def svc_list_llm_usage(days: int = 7):
    try:
        res = supabase.table("llm_usage").select("*").order("called_at", desc=True).execute()
        return res.data if res.data else []
    except Exception:
        return []

# ---------------------------------------------------------------------------
# 5. UI LAYER
# ---------------------------------------------------------------------------

def render_model_selector():
    st.sidebar.markdown("### 🧠 AI Model")
    options = list(MODEL_REGISTRY.keys())
    default_idx = options.index(DEFAULT_MODEL_KEY) if DEFAULT_MODEL_KEY in options else 0
    selected = st.sidebar.selectbox(
        "Select model", options,
        index=st.session_state.get("selected_model_index", default_idx),
        key="selected_model_widget",
        label_visibility="collapsed",
    )
    st.session_state["selected_model_key"] = selected
    st.session_state["selected_model_index"] = options.index(selected)
    entry = MODEL_REGISTRY[selected]
    provider = entry["provider"]
    st.sidebar.caption(f"Provider: **{provider}**")
    st.sidebar.caption(f"`{entry['model_id']}`")
    st.sidebar.caption(entry["description"])
    if provider == "groq" and not GROQ_API_KEY:
        st.sidebar.error("Groq key missing.")
    if provider == "sealion" and not SEALION_API_KEY:
        st.sidebar.error("Sea Lion key missing.")

def render_sidebar():
    with st.sidebar:
        st.title("💼 AI Job Hunt Hub")
        st.caption("AI-powered job search tracker")
        st.markdown("---")
        st.markdown("**👤 Lester**")
        st.caption("lesterchia1@hotmail.com")
        st.markdown("---")
        nav = st.radio(
            "Navigation",
            ["📊 Tracker", "➕ New Application", "📝 Edit & Export",
             "📥 Outbox", "📈 Analytics", "🔍 Monitoring", "⚙️ Settings"],
            index=0,
            label_visibility="collapsed",
        )
        st.markdown("---")
        search = st.text_input("🔍 Search company / role", "")
        st.markdown("---")
        render_model_selector()
        st.markdown("---")
        provider, model_id, label = get_active_model()
        llm, status = load_llm_for(provider, model_id)
        if llm is None:
            st.error(f"LLM unavailable: {status}")
        else:
            st.success(f"LLM ready: {label}")
    return nav, search

def render_kpi_banner(df: pd.DataFrame):
    total = len(df)
    applied = int((df["status"] == "Applied").sum()) if total else 0
    interviewing = int((df["status"] == "Interviewing").sum()) if total else 0
    offers = int((df["status"] == "Offer").sum()) if total else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applications", total)
    c2.metric("Applied", applied)
    c3.metric("Interviewing", interviewing)
    c4.metric("Offers", offers)

def render_kanban(df: pd.DataFrame):
    cols = st.columns(len(STAGES))
    for col, stage in zip(cols, STAGES):
        stage_df = df[df["status"] == stage["id"]]
        with col:
            st.markdown(
                f"""<div style="background:{stage['color']};color:white;
                padding:6px 10px;border-radius:6px;font-weight:600;
                font-size:13px;text-align:center;">
                {stage['name']} ({len(stage_df)})</div>""",
                unsafe_allow_html=True,
            )
            st.write("")
            if stage_df.empty:
                st.caption("—")
                continue
            for _, row in stage_df.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['role']}**")
                    st.caption(f"🏢 {row['company']}")
                    if row.get("match_score"):
                        st.caption(f"🎯 {row['match_score']}")
                    new_status = st.selectbox(
                        "Move to",
                        [s["id"] for s in STAGES],
                        index=[s["id"] for s in STAGES].index(row["status"]),
                        key=f"status_{row['id']}",
                        label_visibility="collapsed",
                    )
                    if new_status != row["status"]:
                        svc_update_status(int(row["id"]), new_status)
                        st.rerun()
                    with st.expander("Details"):
                        st.write(f"**Applied:** {row.get('applied_date', 'N/A')}")
                        if row.get("job_url"):
                            st.markdown(f"[Job posting]({row['job_url']})")
                        if st.button("🗑 Delete", key=f"del_{row['id']}"):
                            svc_delete_application(int(row["id"]))
                            st.rerun()

def view_tracker(search: str):
    st.title("📊 Tracker")
    st.caption("Track your applications from wishlist to offer.")
    apps = svc_list_applications()
    df = pd.DataFrame(apps) if apps else pd.DataFrame(
        columns=["id", "company", "role", "status", "match_score", "applied_date"]
    )
    if search and not df.empty:
        mask = (
            df["company"].str.contains(search, case=False, na=False)
            | df["role"].str.contains(search, case=False, na=False)
        )
        df = df[mask]
    render_kpi_banner(df)
    st.markdown("---")
    if df.empty:
        st.info("No applications yet. Use **➕ New Application** in the sidebar to add one.")
    else:
        render_kanban(df)

def view_new_application():
    st.title("➕ New Application")
    st.caption("Paste the job description and upload your resume. AI will score, tailor, and draft.")
    _, _, model_label = get_active_model()
    st.info(f"Using model: **{model_label}**")
    with st.form("new_app_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        company = c1.text_input("Company Name *")
        role = c2.text_input("Role Title *")
        job_url = st.text_input("Job URL (optional)")
        job_description = st.text_area("Job Description", height=180)
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        submitted = st.form_submit_button("🔍 Analyze & Add", type="primary")
    if submitted:
        if not company or not role:
            st.error("Company and Role are required.")
            return
        resume_text = extract_pdf_text(resume_file) if resume_file else ""
        match_str = ""
        if resume_text and job_description:
            with st.spinner("Scoring resume against job..."):
                score = calculate_match_score(resume_text, job_description)
                match_str = f"{score:.0%}"
                st.metric("Profile Match Score", match_str)
            with st.spinner(f"Tailoring resume with {model_label}..."):
                tailored = tailor_resume(resume_text, job_description)
            with st.spinner(f"Drafting cover letter with {model_label}..."):
                cover = generate_cover_letter(tailored, job_description, company, role)
            st.success("Documents generated.")
            with st.expander("📄 Tailored Resume", expanded=False):
                st.text_area("Tailored Resume", tailored, height=280, label_visibility="collapsed")
            with st.expander("✉️ Cover Letter", expanded=False):
                st.text_area("Cover Letter", cover, height=280, label_visibility="collapsed")
            app_id = svc_create_application(
                company=company, role=role, job_url=job_url,
                job_description=job_description,
                resume_text=tailored, cover_letter=cover,
                match_score=match_str, status="Applied",
                applied_date=datetime.now()
            )
            st.success(f"✅ Application #{app_id} saved to Tracker.")
        else:
            app_id = svc_create_application(
                company=company, role=role, job_url=job_url,
                job_description=job_description, status="Applied",
                applied_date=datetime.now()
            )
            st.info(
                f"Application #{app_id} saved without AI analysis. "
                "Upload a PDF + paste a job description to enable AI features."
            )

def view_edit_export():
    st.title("📝 Edit & Export Documents")
    st.caption("Refine your AI-generated resume and cover letter, then export as PDF or DOCX.")

    apps = svc_list_applications()
    if not apps:
        st.info("No applications found. Add one first from **➕ New Application**.")
        return

    app_options = {f"#{a['id']} · {a['company']} · {a['role']}": a['id'] for a in apps}
    selected_label = st.selectbox("Select an application to edit", list(app_options.keys()))
    app_id = app_options[selected_label]

    app = svc_get_application(app_id)
    if not app:
        st.error("Application not found.")
        return

    st.markdown(f"**{app['role']}** at **{app['company']}**")
    st.caption(f"Status: {app['status']} · Match: {app.get('match_score', 'N/A')}")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📄 Tailored Resume", "✉️ Cover Letter"])

    with tab1:
        edited_resume = streamlit_lexical_extended(
            value=app.get("resume_text", ""),
            key=f"resume_editor_{app_id}",
            height=500,
            toolbar=["undo", "redo", "block_type", "bold", "italic", "underline", "bullet_list", "numbered_list"],
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Resume", key=f"save_resume_{app_id}"):
                svc_update_application_text(app_id, edited_resume, app.get("cover_letter", ""))
                st.success("Resume saved.")
                st.rerun()
        with col2:
            pdf_bytes = generate_pdf_bytes(edited_resume, "resume.pdf")
            st.download_button(
                "📥 Download as PDF",
                data=pdf_bytes,
                file_name=f"{app['company']}_Resume.pdf",
                mime="application/pdf",
                key=f"pdf_resume_{app_id}",
            )

    with tab2:
        edited_cover = streamlit_lexical_extended(
            value=app.get("cover_letter", ""),
            key=f"cover_editor_{app_id}",
            height=500,
            toolbar=["undo", "redo", "block_type", "bold", "italic", "underline", "bullet_list", "numbered_list"],
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Cover Letter", key=f"save_cover_{app_id}"):
                svc_update_application_text(app_id, app.get("resume_text", ""), edited_cover)
                st.success("Cover letter saved.")
                st.rerun()
        with col2:
            docx_bytes = generate_docx_bytes(edited_cover)
            st.download_button(
                "📥 Download as DOCX",
                data=docx_bytes,
                file_name=f"{app['company']}_Cover_Letter.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"docx_cover_{app_id}",
            )

def view_outbox():
    st.title("📥 Outbox")
    st.info("Email sync is not yet wired up. This view will show parsed job-related emails.")

def view_analytics():
    st.title("📈 Analytics")
    apps = svc_list_applications()
    if not apps:
        st.info("No application data available yet.")
        return
    df = pd.DataFrame(apps)
    st.bar_chart(df["status"].value_counts())

def view_monitoring():
    st.title("🔍 LLM Monitoring")
    st.caption("Inspect token consumption, latencies, and API usage costs.")
    usage = svc_list_llm_usage(days=7)
    if not usage:
        st.info("No LLM usage logs recorded yet.")
        return
    df_usage = pd.DataFrame(usage)
    st.dataframe(df_usage, use_container_width=True)

def view_settings():
    st.title("⚙️ Settings")
    st.write(f"**Current User ID:** `{USER_ID}`")
    st.write(f"**Supabase REST Endpoint:** `{SUPABASE_URL}`")
    st.success("Connected via Supabase REST API (HTTPS).")

# ---------------------------------------------------------------------------
# MAIN ROUTER
# ---------------------------------------------------------------------------

nav, search = render_sidebar()

if nav == "📊 Tracker":
    view_tracker(search)
elif nav == "➕ New Application":
    view_new_application()
elif nav == "📝 Edit & Export":
    view_edit_export()
elif nav == "📥 Outbox":
    view_outbox()
elif nav == "📈 Analytics":
    view_analytics()
elif nav == "🔍 Monitoring":
    view_monitoring()
elif nav == "⚙️ Settings":
    view_settings()
