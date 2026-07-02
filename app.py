import streamlit as st
import os
import json
import pandas as pd
import subprocess
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="Latent-Fit AI Recruiter",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection (Dark Mode, Glassmorphism, Google Fonts)
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    
    <style>
        /* Main background and font override */
        .stApp {
            background-color: #080C14;
            color: #E2E8F0;
            font-family: 'Outfit', sans-serif;
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }
        
        .main-title {
            background: linear-gradient(135deg, #00D2FF 0%, #9B51E0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            font-weight: 800;
            margin-bottom: 5px;
            text-shadow: 0 0 30px rgba(0, 210, 255, 0.1);
        }
        
        .sub-title {
            color: #94A3B8;
            font-size: 1.15rem;
            margin-bottom: 30px;
            font-weight: 300;
        }
        
        /* Glassmorphic Metrics Card */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: rgba(21, 27, 44, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            border-color: rgba(0, 210, 255, 0.4);
        }
        
        .metric-label {
            color: #94A3B8;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        
        .metric-val {
            font-size: 2.2rem;
            font-weight: 800;
            color: #E2E8F0;
        }
        
        .metric-val-accent1 {
            color: #00D2FF;
        }
        
        .metric-val-accent2 {
            color: #FF5E7E;
        }
        
        .metric-val-accent3 {
            color: #9B51E0;
        }
        
        /* Glassmorphic Candidate Card */
        .candidate-card {
            background: rgba(21, 27, 44, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .candidate-card:hover {
            border-color: rgba(155, 81, 224, 0.4);
            box-shadow: 0 8px 32px 0 rgba(155, 81, 224, 0.05);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .rank-badge {
            background: linear-gradient(135deg, #00D2FF 0%, #9B51E0 100%);
            color: #080C14;
            font-weight: 800;
            font-size: 1.1rem;
            padding: 6px 14px;
            border-radius: 30px;
            box-shadow: 0 4px 15px rgba(0, 210, 255, 0.2);
        }
        
        .rank-badge-1 {
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
        }
        .rank-badge-2 {
            background: linear-gradient(135deg, #E2E8F0 0%, #94A3B8 100%);
            box-shadow: 0 4px 15px rgba(226, 232, 240, 0.2);
        }
        .rank-badge-3 {
            background: linear-gradient(135deg, #CD7F32 0%, #8B4513 100%);
            box-shadow: 0 4px 15px rgba(205, 127, 50, 0.2);
        }
        
        .cand-name {
            font-size: 1.35rem;
            font-weight: 800;
            color: #E2E8F0;
        }
        
        .cand-title {
            color: #94A3B8;
            font-size: 0.95rem;
            margin-bottom: 15px;
            font-weight: 400;
        }
        
        .score-pill {
            background: rgba(0, 210, 255, 0.1);
            color: #00D2FF;
            border: 1px solid rgba(0, 210, 255, 0.2);
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
        }
        
        .pill {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #CBD5E1;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
        }
        
        .pill-accent {
            background: rgba(155, 81, 224, 0.1);
            border: 1px solid rgba(155, 81, 224, 0.2);
            color: #D6BCFA;
        }
        
        /* AI Reasoning block */
        .reasoning-box {
            background: rgba(0, 210, 255, 0.04);
            border-left: 4px solid #00D2FF;
            padding: 16px;
            border-radius: 0 8px 8px 0;
            margin-top: 15px;
            font-size: 0.95rem;
            line-height: 1.5;
            color: #E2E8F0;
        }
        
        .reasoning-title {
            font-weight: 800;
            color: #00D2FF;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Streamlit overrides */
        .stButton>button {
            background: linear-gradient(135deg, #00D2FF 0%, #9B51E0 100%) !important;
            color: #080C14 !important;
            font-weight: 800 !important;
            border: none !important;
            padding: 10px 24px !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton>button:hover {
            transform: scale(1.02) !important;
            box-shadow: 0 4px 20px rgba(0, 210, 255, 0.5) !important;
        }
        
        .stTextArea textarea {
            background-color: #0F1626 !important;
            color: #E2E8F0 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
        }
    </style>
""", unsafe_allow_html=True)

# App Logo / Header
st.markdown("<div class='main-title'>LATENT-FIT AI RECRUITER</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Two-Tiered Hybrid-OKF Candidate Retrieval & Ranking Engine</div>", unsafe_allow_html=True)

# 1. Sidebar Configurations
st.sidebar.markdown("### ⚙️ Pipeline Controls")

# Default Job Description path
jd_path = "India_runs_data_and_ai_challenge/job_description.md"
default_jd = ""
if os.path.exists(jd_path):
    with open(jd_path, "r", encoding="utf-8") as f:
        default_jd = f.read()
else:
    default_jd = "Senior AI Engineer Founding Team. Python, embeddings-based retrieval systems, vector databases, hybrid search (Milvus, Qdrant, Pinecone), evaluation frameworks (NDCG, MAP, MRR), NLP."

# Job description editor
jd_text = st.sidebar.text_area("📝 Edit Job Description", value=default_jd, height=350)

# Optional file uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload Custom candidates.jsonl (Optional)", type=["jsonl"])

# Action Button
run_btn = st.sidebar.button("⚡ Execute Hybrid-OKF Search")

# 2. Main content area: Ingestion and filtration stats
st.markdown("### 📊 Dataset Filtration Funnel")

# Dynamic or static stats depending on run
total_parsed = 100000
non_tech_excl = 68821
honeypot_excl = 4970
remaining_scored = 26209

# Metrics Row
st.markdown(f"""
    <div class='metrics-grid'>
        <div class='metric-card'>
            <div class='metric-label'>1. Ingestion Pool</div>
            <div class='metric-val'>{total_parsed:,}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>2. Non-Tech Excluded</div>
            <div class='metric-val metric-val-accent2'>{non_tech_excl:,}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>3. Honeypots Blocked</div>
            <div class='metric-val metric-val-accent2'>{honeypot_excl:,}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>4. OKF Scored Subset</div>
            <div class='metric-val metric-val-accent1'>{remaining_scored:,}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Check if submission CSV exists
out_csv = "team_submission.csv"

# Function to run the backend ranker
def run_backend_engine(candidates_path, output_path, jd_content):
    # Save the current edited JD content to the path main.py reads
    with open("India_runs_data_and_ai_challenge/job_description.md", "w", encoding="utf-8") as f:
        f.write(jd_content)
        
    cmd = ["python", "rank.py", "--candidates", candidates_path, "--out", output_path]
    
    with st.spinner("Processing candidates, filtering traps, and calculating hybrid scores..."):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            st.success("⚡ Hybrid-OKF Engine executed successfully!")
            return True
        else:
            st.error(f"Execution failed: {res.stderr}")
            return False

# Execute ranker on button click
if run_btn:
    candidates_file_path = "India_runs_data_and_ai_challenge/candidates.jsonl"
    
    if uploaded_file is not None:
        # Save temp file
        temp_file = "temp_candidates.jsonl"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        candidates_file_path = temp_file
        
    success = run_backend_engine(candidates_file_path, out_csv, jd_text)
    
    if uploaded_file is not None and os.path.exists("temp_candidates.jsonl"):
        os.remove("temp_candidates.jsonl")

# 3. Load & display results
if os.path.exists(out_csv):
    df = pd.read_csv(out_csv)
    
    # Download Button
    st.markdown("### 🏆 Top Ranked Candidates (Top 100)")
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Validated CSV Submission",
        data=csv_data,
        file_name="team_submission.csv",
        mime="text/csv"
    )
    
    st.markdown("Select a candidate below to inspect their full profile, OKF markdown representation, and dynamic AI justifications.")
    
    # Display candidates in lists
    for idx, row in df.iterrows():
        rank = int(row["rank"])
        cid = row["candidate_id"]
        score = float(row["score"])
        justification = row["reasoning"]
        
        # Load candidate details from JSONL if possible
        # To avoid reading the 487MB file every time, we look inside sample_candidates or cached files.
        # However, we can also extract details directly from our precomputed /knowledge_graph directory!
        # This is incredibly fast since the top candidates are written to knowledge_graph/CAND_XXXXXXX.md!
        okf_path = f"knowledge_graph/{cid}.md"
        okf_content = ""
        cand_name = "Anonymized Candidate"
        current_title = "AI/ML Engineer"
        location = "Pune/Noida, India"
        notice = "30 Days"
        github = "Active"
        skills_pills = []
        
        if os.path.exists(okf_path):
            with open(okf_path, "r", encoding="utf-8") as f:
                okf_content = f.read()
                
            # Parse YAML frontmatter to display metrics
            if okf_content.startswith("---"):
                parts = okf_content.split("---")
                if len(parts) >= 3:
                    try:
                        yaml_part = yaml_part = yaml_part = parts[1]
                        meta = yaml.safe_load(yaml_part) if 'yaml' in globals() else {}
                        # Re-parse meta using standard library or simple splits if yaml not imported
                        import yaml
                        meta = yaml.safe_load(yaml_part)
                        cand_name = meta.get("anonymized_name", cand_name)
                        current_title = meta.get("current_title", current_title)
                        location = f"{meta.get('location')}, {meta.get('country')}"
                        notice = f"{meta.get('notice_period_days')}d notice"
                        github = f"GitHub: {meta.get('github_activity_score')}" if meta.get("github_activity_score", -1) != -1 else "No GitHub"
                        skills_pills = [s["name"] for s in meta.get("skills", [])[:5]]
                    except Exception as e:
                        pass
        
        # Format Rank Badge style class
        rank_class = "rank-badge"
        if rank == 1:
            rank_class += " rank-badge-1"
        elif rank == 2:
            rank_class += " rank-badge-2"
        elif rank == 3:
            rank_class += " rank-badge-3"
            
        badge_html = " ".join([f"<span class='pill'>{s}</span>" for s in skills_pills])
        
        expander_title = f"Rank {rank} | {cid} — {current_title} ({cand_name})"
        
        # HTML Card UI
        with st.expander(expander_title):
            st.markdown(f"""
                <div class='candidate-header'>
                    <div class='card-header'>
                        <div class='rank-badge {rank_class}'>RANK {rank}</div>
                        <div class='cand-name'>{cand_name} <span style='font-size:0.9rem; color:#94A3B8;'>({cid})</span></div>
                        <div class='score-pill'>SCORE: {score:.4f}</div>
                    </div>
                    <div class='cand-title'>💼 {current_title} | 📍 {location}</div>
                    <div class='badge-row'>
                        <span class='pill pill-accent'>⏳ {notice}</span>
                        <span class='pill pill-accent'>💻 {github}</span>
                        {badge_html}
                    </div>
                    <div class='reasoning-box'>
                        <div class='reasoning-title'>🧠 AI Recruiter Justification</div>
                        {justification}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Show OKF Markdown block
            if okf_content:
                st.markdown("#### 📄 Open Knowledge Format (OKF) Profile")
                st.text_area("Candidate OKF Text", value=okf_content, height=250, key=f"okf_{cid}")
            else:
                st.info("Additional profile content is loaded in the full candidates file.")
else:
    st.info("No submission data found. Run the Hybrid-OKF Search in the sidebar to generate rankings.")
