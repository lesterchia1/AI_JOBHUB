"""
AI Job Hunt OS — Test Build v0.3
Streamlit app that tracks job applications and uses AI (Groq + Sea Lion)
to tailor resumes and generate cover letters.

Run:
    streamlit run app.py
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from groq import Groq
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. CONFIGURATION (must run before any other Streamlit command)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Job Hunt OS",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()  # Load .env if present

# ============================================================
# 🚀 API KEY CONFIGURATION (Colab + local + Streamlit Cloud)
# ============================================================

def _get_secret(key: str, default=None):
    """Try Colab userdata → os.environ → st.secrets, in that order."""
    # 1. Google Colab
    try:
        from google.colab import userdata
        val = userdata.get(key)
        if val:
            return val
    except ImportError:
        pass

    # 2. Environment variables (.env or exported)
    val = os.environ.get(key)
    if val:
        return val

    # 3. Streamlit secrets (.streamlit/secrets.toml)
    try:
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass

    return default


GROQ_API_KEY    = _get_secret("GROQ_API_KEY")
SEALION_API_KEY = _get_secret("SEALION_API_KEY")   # optional fallback
HF_TOKEN        = _get_secret("HF_TOKEN")          # optional

# --- Validation: only Groq is required ---
if not GROQ_API_KEY and not SEALION_API_KEY:
    st.error(
        "❌ No API keys found. Set at least one of GROQ_API_KEY or SEALION_API_KEY "
        "in your `.env` file or Streamlit secrets."
    )
    st.stop()

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# --- Initialize clients only if keys exist ---
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

DB_PATH = "job_hunt.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ============================================================
# 🧠 MODEL REGISTRY — the single source of truth for models
# ============================================================
# Each entry: (provider, model_id, display_label)
# Add / remove entries here as providers update their catalogs.

MODEL_REGISTRY = {
    # --- AI Singapore Sea-Lion (OpenAI-compatible endpoint) ---
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

    # --- Groq (fast, free tier) ---
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
    "Groq: Llama 4 Scout": {
        "provider": "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "description": "Meta Llama 4 Scout, 17B MoE, fast general-purpose.",
    },
    "Groq: Llama 4 Maverick": {
        "provider": "groq",
        "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "description": "Meta Llama 4 Maverick, 17B MoE, higher capability.",
    },
    "Groq: Qwen 3 32B": {
        "provider": "groq",
        "model_id": "qwen/qwen3-32b",
        "description": "Qwen 3 32B, strong reasoning.",
    },
}

# Default model (change this to any key in MODEL_REGISTRY)
DEFAULT_MODEL_KEY = "Sea-Lion v4 27B"

# ---------------------------------------------------------------------------
# 2. DATABASE LAYER
# ---------------------------------------------------------------------------

Base = declarative_base()


class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True)
    company = Column(String, nullable=False)
    role = Column(String, nullable=False)
    job_url = Column(String, default="")
    status = Column(String, default="Applied")
    job_description = Column(Text, default="")
    resume_text = Column(Text, default="")
    cover_letter = Column(Text, default="")
    match_score = Column(String, default="")
    applied_date = Column(DateTime, default=datetime.now)
    last_updated = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


@st.cache_resource
def get_engine():
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


@st.cache_resource
def get_session_factory():
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


def db_session():
    return get_session_factory()()


# ---------------------------------------------------------------------------
# 3. AI LAYER
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedding_model():
    """Lazy-load sentence-transformer. Returns None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"Embedding model unavailable: {e}")
        return None


def get_active_model() -> tuple[str, str, str]:
    """
    Read the selected model from session state.
    Returns (provider, model_id, display_label).
    """
    key = st.session_state.get("selected_model_key", DEFAULT_MODEL_KEY)
    entry = MODEL_REGISTRY.get(key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])
    return entry["provider"], entry["model_id"], key


def load_llm_for(provider: str, model_id: str):
    """
    Build a LangChain LLM object for the given provider + model_id.
    Returns (llm, provider_label) or (None, error_message).
    """
    if provider == "groq":
        if not GROQ_API_KEY:
            return None, "Groq API key not set"
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=model_id,
                temperature=0.7,
                api_key=GROQ_API_KEY,
            )
            return llm, f"Groq ({model_id})"
        except Exception as e:
            return None, f"Groq init failed: {e}"

    if provider == "sealion":
        if not SEALION_API_KEY:
            return None, "Sea Lion API key not set"
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_id,
                temperature=0.7,
                api_key=SEALION_API_KEY,
                base_url=SEALION_BASE_URL,
            )
            return llm, f"Sea Lion ({model_id})"
        except Exception as e:
            return None, f"Sea Lion init failed: {e}"

    return None, f"Unknown provider: {provider}"


def calculate_match_score(resume_text: str, job_description: str) -> float:
    """Cosine similarity between resume and job description."""
    if not resume_text or not job_description:
        return 0.0
    model = load_embedding_model()
    if model is None:
        return 0.0
    from sentence_transformers import util
    emb = model.encode([job_description, resume_text], convert_to_tensor=True)
    score = util.cos_sim(emb[0], emb[1])
    return float(score[0][0])


def call_llm(prompt: str) -> str:
    """Single-shot LLM call using the currently selected model."""
    provider, model_id, label = get_active_model()
    llm, status = load_llm_for(provider, model_id)
    if llm is None:
        return f"⚠️ {status}"
    try:
        result = llm.invoke(prompt)
        return getattr(result, "content", str(result))
    except Exception as e:
        return f"⚠️ LLM call failed ({status}): {e}"


def call_groq_direct(prompt: str) -> str:
    """Direct call to Groq SDK using the currently selected Groq model."""
    if groq_client is None:
        return "⚠️ Groq client not initialized (no API key)."
    provider, model_id, _ = get_active_model()
    if provider != "groq":
        return "⚠️ Currently selected model is not a Groq model."
    try:
        resp = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq direct call failed ({model_id}): {e}"


def call_sealion_direct(prompt: str) -> str:
    """Direct call to Sea Lion using the currently selected Sea Lion model."""
    if sealion_client is None:
        return "⚠️ Sea Lion client not initialized (no API key)."
    provider, model_id, _ = get_active_model()
    if provider != "sealion":
        return "⚠️ Currently selected model is not a Sea Lion model."
    try:
        resp = sealion_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Sea Lion direct call failed ({model_id}): {e}"


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
    return call_llm(prompt)


def generate_cover_letter(resume_text: str, job_description: str, company: str, role: str) -> str:
    prompt = f"""Write a professional, concise cover letter for the {role} role at {company}.
Ground it in the resume and job description below. Do NOT use placeholders like [Your Name].

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

COVER LETTER:"""
    return call_llm(prompt)


def extract_pdf_text(uploaded_file) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        return "".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""


def classify_email_status(body: str) -> str:
    """Simple keyword classifier for email statuses."""
    b = body.lower()
    if any(k in b for k in ["unfortunately", "not moving forward", "other candidates", "rejected"]):
        return "Rejected"
    if any(k in b for k in ["interview", "schedule a call", "speak with"]):
        return "Interviewing"
    if any(k in b for k in ["offer", "pleased to offer"]):
        return "Offer"
    if any(k in b for k in ["received your application", "thank you for applying", "acknowledge"]):
        return "Acknowledged"
    return "Applied"


# ---------------------------------------------------------------------------
# 4. SERVICE LAYER
# ---------------------------------------------------------------------------

def svc_list_applications():
    with db_session() as db:
        return [a.to_dict() for a in db.query(Application).order_by(Application.last_updated.desc()).all()]


def svc_create_application(**kwargs):
    with db_session() as db:
        app = Application(**kwargs)
        db.add(app)
        db.commit()
        db.refresh(app)
        return app.id


def svc_update_status(app_id: int, new_status: str):
    with db_session() as db:
        app = db.query(Application).filter(Application.id == app_id).first()
        if app:
            app.status = new_status
            app.last_updated = datetime.now()
            db.commit()
            return True
    return False


def svc_delete_application(app_id: int):
    with db_session() as db:
        app = db.query(Application).filter(Application.id == app_id).first()
        if app:
            db.delete(app)
            db.commit()


# ---------------------------------------------------------------------------
# 5. UI LAYER
# ---------------------------------------------------------------------------

def render_model_selector():
    """Sidebar widget to pick the active model. Saves choice to session_state."""
    st.sidebar.markdown("### 🧠 AI Model")
    options = list(MODEL_REGISTRY.keys())
    default_idx = options.index(DEFAULT_MODEL_KEY) if DEFAULT_MODEL_KEY in options else 0

    selected = st.sidebar.selectbox(
        "Select model",
        options,
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

    # Warn if the selected provider has no key
    if provider == "groq" and not GROQ_API_KEY:
        st.sidebar.error("Groq key missing.")
    if provider == "sealion" and not SEALION_API_KEY:
        st.sidebar.error("Sea Lion key missing.")


def render_sidebar():
    with st.sidebar:
        st.title("💼 Job Hunt OS")
        st.caption("AI-powered job search tracker")
        st.markdown("---")

        st.markdown("**👤 Lester**")
        st.caption("lesterchia1@hotmail.com")
        st.markdown("---")

        nav = st.radio(
            "Navigation",
            ["📊 Tracker", "➕ New Application", "📥 Outbox", "📈 Analytics", "⚙️ Settings"],
            index=0,
            label_visibility="collapsed",
        )
        st.markdown("---")

        search = st.text_input("🔍 Search company / role", "")

        st.markdown("---")
        render_model_selector()
        st.markdown("---")

        # Health check on the active model
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
    """Native Streamlit Kanban: one column per stage, cards with status selector."""
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
                        st.write(f"**Applied:** {row['applied_date']}")
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
                company=company,
                role=role,
                job_url=job_url,
                job_description=job_description,
                resume_text=tailored,
                cover_letter=cover,
                match_score=match_str,
                status="Applied",
            )
            st.success(f"✅ Application #{app_id} saved to Tracker.")
        else:
            app_id = svc_create_application(
                company=company,
                role=role,
                job_url=job_url,
                job_description=job_description,
                status="Applied",
            )
            st.info(
                f"Application #{app_id} saved without AI analysis. "
                "Upload a PDF + paste a job description to enable AI features."
            )


def view_outbox():
    st.title("📥 Outbox")
    st.info("Email sync is not yet wired up. This view will show parsed job-related emails.")

    sample = pd.DataFrame({
        "Date": ["2026-10-08", "2026-10-07", "2026-10-06"],
        "Sender": ["recruiting@stripe.com", "no-reply@google.com", "hr@meta.com"],
        "Subject": ["Interview Invitation", "Application Received", "Update on your application"],
        "Detected Status": ["🟢 Interviewing", "🟡 Acknowledged", "🔴 Rejected"],
    })
    st.dataframe(sample, use_container_width=True, hide_index=True)

    st.markdown("**Test the classifier:**")
    body = st.text_area("Paste an email body:")
    if body:
        st.success(f"Detected status: **{classify_email_status(body)}**")


def view_analytics():
    st.title("📈 Analytics")
    apps = svc_list_applications()

    if not apps:
        st.info("No data yet. Add applications to see analytics.")
        return

    df = pd.DataFrame(apps)
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Applications by Stage")
        st.bar_chart(df["status"].value_counts())

    with c2:
        st.subheader("Applications Over Time")
        if "applied_date" in df:
            df["date"] = pd.to_datetime(df["applied_date"]).dt.date
            st.line_chart(df.groupby("date").size())


def view_settings():
    st.title("⚙️ Settings")

    provider, model_id, label = get_active_model()

    st.markdown("**Active Model**")
    st.code(
        f"Display key : {label}\n"
        f"Provider    : {provider}\n"
        f"Model ID    : {model_id}",
        language="bash",
    )

    st.markdown("**API Keys Detected**")
    st.write(f"- GROQ_API_KEY: {'✅ set' if GROQ_API_KEY else '❌ missing'}")
    st.write(f"- SEALION_API_KEY: {'✅ set' if SEALION_API_KEY else '❌ missing'}")
    st.write(f"- HF_TOKEN: {'✅ set' if HF_TOKEN else '⚪ not set (optional)'}")

    st.markdown("**Test Direct Calls**")
    if provider == "groq":
        if st.button("Test Groq direct call"):
            with st.spinner(f"Calling Groq ({model_id})..."):
                st.info(call_groq_direct("Reply with: 'Groq OK'"))
    elif provider == "sealion":
        if st.button("Test Sea Lion direct call"):
            with st.spinner(f"Calling Sea Lion ({model_id})..."):
                st.info(call_sealion_direct("Reply with: 'Sea Lion OK'"))

    st.markdown("---")
    st.markdown("**Available Models**")
    for key, entry in MODEL_REGISTRY.items():
        marker = " ← active" if key == label else ""
        st.markdown(f"- **{key}**{marker}  \n  `{entry['model_id']}`  \n  {entry['description']}")

    st.markdown("---")
    st.markdown("**Danger Zone**")
    if st.button("🗑 Clear all applications", type="secondary"):
        with db_session() as db:
            db.query(Application).delete()
            db.commit()
        st.success("All applications cleared.")
        st.rerun()


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    # Ensure a default model is set before the sidebar renders
    if "selected_model_key" not in st.session_state:
        st.session_state["selected_model_key"] = DEFAULT_MODEL_KEY

    nav, search = render_sidebar()

    if nav == "📊 Tracker":
        view_tracker(search)
    elif nav == "➕ New Application":
        view_new_application()
    elif nav == "📥 Outbox":
        view_outbox()
    elif nav == "📈 Analytics":
        view_analytics()
    elif nav == "⚙️ Settings":
        view_settings()


if __name__ == "__main__":
    main()
