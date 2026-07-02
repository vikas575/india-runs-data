import json
import os
import yaml
from pathlib import Path
from datetime import datetime

# Initialize directories
os.makedirs("knowledge_graph", exist_ok=True)

def candidate_to_okf(c):
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    edu = c.get("education", [])
    signals = c.get("redrob_signals", {})
    
    # Convert key attributes to YAML frontmatter
    yaml_data = {
        "candidate_id": c.get("candidate_id"),
        "anonymized_name": profile.get("anonymized_name"),
        "years_of_experience": profile.get("years_of_experience"),
        "current_title": profile.get("current_title"),
        "location": profile.get("location"),
        "country": profile.get("country"),
        "notice_period_days": signals.get("notice_period_days"),
        "expected_salary_range_lpa": {
            "min": signals.get("expected_salary_range_inr_lpa", {}).get("min"),
            "max": signals.get("expected_salary_range_inr_lpa", {}).get("max")
        },
        "preferred_work_mode": signals.get("preferred_work_mode"),
        "willing_to_relocate": signals.get("willing_to_relocate"),
        "github_activity_score": signals.get("github_activity_score"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "last_active_date": signals.get("last_active_date"),
        "profile_completeness_score": signals.get("profile_completeness_score"),
        "skills": [{"name": s["name"], "proficiency": s["proficiency"], "duration_months": s.get("duration_months", 0)} for s in skills]
    }
    
    yaml_str = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    # Markdown body
    md_lines = []
    md_lines.append("---")
    md_lines.append(yaml_str.strip())
    md_lines.append("---")
    md_lines.append(f"\n# Profile Summary")
    md_lines.append(f"**Headline**: {profile.get('headline', '')}")
    md_lines.append(f"\n{profile.get('summary', '')}")
    
    md_lines.append(f"\n# Career History")
    for job in career:
        is_curr = "Present" if job.get("is_current") else job.get("end_date", "")
        md_lines.append(f"## {job.get('title')} at {job.get('company')} ({job.get('start_date')} - {is_curr})")
        md_lines.append(f"**Industry**: {job.get('industry')} | **Company Size**: {job.get('company_size')}")
        md_lines.append(f"{job.get('description', '')}\n")
        
    md_lines.append(f"\n# Education")
    for school in edu:
        md_lines.append(f"- **{school.get('degree')}** in {school.get('field_of_study')} from {school.get('institution')} ({school.get('start_year')} - {school.get('end_year')}), Grade: {school.get('grade')}, Tier: {school.get('tier')}")
        
    return "\n".join(md_lines)

def run_precompute():
    input_path = "India_runs_data_and_ai_challenge/candidates.jsonl"
    
    print("Pre-computation started...")
    
    excluded_titles = {
        'Business Analyst', 'HR Manager', 'Mechanical Engineer', 'Accountant', 
        'Project Manager', 'Customer Support', 'Operations Manager', 'Content Writer', 
        'Sales Executive', 'Civil Engineer', 'Graphic Designer', 'Marketing Manager'
    }
    
    valid_candidates = []
    candidate_ids = []
    candidate_texts = []
    
    total_processed = 0
    non_tech_count = 0
    honeypot_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            total_processed += 1
            c = json.loads(line)
            cid = c["candidate_id"]
            profile = c.get("profile", {})
            title = profile.get("current_title", "")
            
            # 1. Filter out non-technical roles
            if title in excluded_titles:
                non_tech_count += 1
                continue
                
            # 2. Check for Honeypots / anomalies
            is_honeypot = False
            
            # Education timeline check
            edu = c.get("education", [])
            years = profile.get("years_of_experience", 0)
            if edu:
                earliest_grad = min([e.get("end_year") for e in edu if e.get("end_year")], default=9999)
                if earliest_grad != 9999:
                    if earliest_grad > 2024 - (years - 4.0) and years > 5.0:
                        is_honeypot = True
            
            # Skill duration anomalies
            skills = c.get("skills", [])
            for s in skills:
                if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 0) == 0:
                    is_honeypot = True
            
            # Expected salary range anomaly: min > max
            signals = c.get("redrob_signals", {})
            sal_min = signals.get("expected_salary_range_inr_lpa", {}).get("min", 0)
            sal_max = signals.get("expected_salary_range_inr_lpa", {}).get("max", 0)
            if sal_min > sal_max:
                is_honeypot = True
                
            if is_honeypot:
                honeypot_count += 1
                continue
                
            # Valid candidate -> format to OKF and record
            okf_text = candidate_to_okf(c)
            valid_candidates.append(c)
            candidate_ids.append(cid)
            candidate_texts.append(okf_text)
            
    print(f"Total parsed: {total_processed}")
    print(f"Non-technical excluded: {non_tech_count}")
    print(f"Honeypots excluded: {honeypot_count}")
    print(f"Valid candidates remaining: {len(valid_candidates)}")
    
    # Write 3 sample OKF files to knowledge_graph
    sample_ids = ["CAND_0018499", "CAND_0046525", "CAND_0061257"]
    sample_written = 0
    for idx, cid in enumerate(candidate_ids):
        if cid in sample_ids:
            with open(f"knowledge_graph/{cid}.md", "w", encoding="utf-8") as out_f:
                out_f.write(candidate_texts[idx])
            sample_written += 1
            print(f"Wrote sample OKF file: knowledge_graph/{cid}.md")
            if sample_written >= 3:
                break
                
    # Fallback to write first 3 if sample IDs aren't found for some reason
    if sample_written < 3:
        for idx in range(min(3, len(candidate_ids))):
            cid = candidate_ids[idx]
            if not os.path.exists(f"knowledge_graph/{cid}.md"):
                with open(f"knowledge_graph/{cid}.md", "w", encoding="utf-8") as out_f:
                    out_f.write(candidate_texts[idx])
                print(f"Wrote fallback sample OKF file: knowledge_graph/{cid}.md")
    
    # Compute embeddings
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # We will embed the candidate OKF texts in batches
    print(f"Computing embeddings for {len(candidate_texts)} candidates...")
    embeddings = model.encode(candidate_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    
    # Save as compressed npz file
    np.savez_compressed("embeddings.npz", ids=np.array(candidate_ids), embeddings=embeddings)
    print("Pre-computation completed. Embeddings saved to 'embeddings.npz'.")

if __name__ == "__main__":
    run_precompute()
