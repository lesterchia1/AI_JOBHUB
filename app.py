"""
AI Job Hunt OS — Test Build v0.1
A Streamlit app that tracks job applications and uses AI to tailor resumes
and generate cover letters.

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
    """Create the SQLAlchemy engine once."""
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


@st.cache_resource
def get_session_factory():
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


def db_session():
    return get_session_factory()()


# ---------------------------------------------------------------------------
# 3. AI LAYER (with graceful fallback if no API key is present)
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


@st.cache_resource(show_spinner="Connecting to LLM...")
def load_llm():
    """Try Groq -> HuggingFace -> OpenAI in order. Returns (llm, provider_name)."""
    # Groq
    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7), "Groq"
        except Exception as e:
            st.warning(f"Groq init failed: {e}")

    # Hugging Face
    if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            endpoint = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.3-70B-Instruct",
                task="text-generation",
                max_new_tokens=1024,
                temperature=0.7,
            )
            return ChatHuggingFace(llm=endpoint), "Hugging Face"
        except Exception as e:
            st.warning(f"Hugging Face init failed: {e}")

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", temperature=0.7), "OpenAI"
        except Exception as e:
            st.warning(f"OpenAI init failed: {e}")

    return None, "None"


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
    """Single-shot LLM call with error handling."""
    llm, _ = load_llm()
    if llm is None:
        return "⚠️ No LLM provider configured. Add GROQ_API_KEY to your .env file."
    try:
        result = llm.invoke(prompt)
        return getattr(result, "content", str(result))
    except Exception as e:
        return f"⚠️ LLM call failed: {e}"


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

        _, provider = load_llm()
        if provider == "None":
            st.warning("No LLM key set.\nAdd `GROQ_API_KEY` to `.env`.")
        else:
            st.success(f"LLM: {provider}")

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

            with st.spinner("Tailoring resume..."):
                tailored = tailor_resume(resume_text, job_description)

            with st.spinner("Drafting cover letter..."):
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
    st.markdown("**LLM Provider Status**")
    _, provider = load_llm()
    st.write(f"Active provider: **{provider}**")

    st.markdown("**Environment Variables**")
    st.code(
        "# .env\n"
        "GROQ_API_KEY=your_key_here\n"
        "# HUGGINGFACEHUB_API_TOKEN=hf_...\n"
        "# OPENAI_API_KEY=sk-...\n",
        language="bash",
    )

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
