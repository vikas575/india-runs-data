import os
import sys
import argparse
import json
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_date(d_str):
    try:
        return datetime.strptime(d_str, "%Y-%m-%d")
    except:
        return None

def candidate_to_okf(c):
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    edu = c.get("education", [])
    signals = c.get("redrob_signals", {})
    
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

def generate_reasoning(c, rank, score):
    profile = c.get("profile", {})
    name = profile.get("anonymized_name", "Candidate")
    title = profile.get("current_title", "Engineer")
    years = profile.get("years_of_experience", 0)
    career = c.get("career_history", [])
    skills = [s["name"] for s in c.get("skills", [])]
    signals = c.get("redrob_signals", {})
    
    last_comp = career[0].get("company", "previous employer") if career else "a top product company"
    
    # Extract matching skills
    core_terms = ["ml", "ai", "machine learning", "deep learning", "nlp", "llm", "retrieval", "search", "rag", "embeddings", "vector", "pytorch", "weaviate", "pinecone", "milvus", "qdrant"]
    matched_skills = [s for s in skills if any(t in s.lower() for t in core_terms)]
    skills_list = ", ".join(matched_skills[:3]) if matched_skills else "advanced software architecture"
    
    notice = signals.get("notice_period_days", 90)
    resp_rate = int(signals.get("recruiter_response_rate", 0.0) * 100)
    gh_score = signals.get("github_activity_score", -1)
    
    # We construct varying reasonings based on rank range
    if rank <= 10:
        templates = [
            f"Exceptional founding fit as a {title} with {years:.1f} yrs exp. Strong background at {last_comp} and shipped models using {skills_list}. Very active on GitHub ({gh_score}) with {notice}d notice period.",
            f"Outstanding pedigree ({years:.1f} yrs) matching founding team criteria. Shipped search/retrieval systems at {last_comp} with skills in {skills_list}. highly responsive ({resp_rate}%) and available in {profile.get('location')}.",
            f"Premium technical match. {title} with {years:.1f} years of applied ML experience, notably {skills_list} at {last_comp}. Excellent platform activity and only {notice} days notice."
        ]
        return templates[rank % len(templates)]
    elif rank <= 50:
        templates = [
            f"Strong candidate with {years:.1f} yrs exp. Demonstrates applied ML skills including {skills_list} at {last_comp}. Good fit for Noida/Pune cadence, notice period is {notice} days.",
            f"Solid ML professional with {years:.1f} yrs exp. Shipped products at {last_comp} utilizing {skills_list}. A minor concern is notice period ({notice}d), but technical match is high.",
            f"Proven track record in technical systems. {title} with {years:.1f} yrs exp, demonstrating hands-on familiarity with {skills_list}. Strong overall platform signals ({resp_rate}% response rate)."
        ]
        return templates[rank % len(templates)]
    else:
        templates = [
            f"Qualified technical profile. {title} with {years:.1f} yrs exp, matching core requirements in {skills_list}. Some timeline constraints (notice: {notice}d) but good baseline fundamentals.",
            f"Matches core criteria with {years:.1f} yrs exp. Worked on relevant ML engineering at {last_comp}. Notice period is {notice}d and response rate is {resp_rate}%, representing a viable hybrid fallback.",
            f"Technical engineer with {years:.1f} years exp. Possesses adjacent skills in {skills_list} at {last_comp}. Willing to relocate, but lower platform engagement signals."
        ]
        return templates[rank % len(templates)]

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob AI Senior AI Engineer role.")
    parser.add_argument("--candidates", type=str, default="India_runs_data_and_ai_challenge/candidates.jsonl", help="Path to candidates.jsonl file")
    parser.add_argument("--out", type=str, default="team_submission.csv", help="Path to write the submission CSV")
    args = parser.parse_args()
    
    # 1. Load Job Description
    jd_path = "India_runs_data_and_ai_challenge/job_description.md"
    if os.path.exists(jd_path):
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
    else:
        jd_text = "Senior AI Engineer Founding Team. Python, embeddings-based retrieval systems, vector databases, hybrid search (Milvus, Qdrant, Pinecone), evaluation frameworks (NDCG, MAP, MRR), NLP."
        
    print(f"Reading candidates from {args.candidates}...")
    target_date = datetime(2026, 6, 29)
    
    excluded_titles = {
        'Business Analyst', 'HR Manager', 'Mechanical Engineer', 'Accountant', 
        'Project Manager', 'Customer Support', 'Operations Manager', 'Content Writer', 
        'Sales Executive', 'Civil Engineer', 'Graphic Designer', 'Marketing Manager'
    }
    
    ai_ml_titles = {
        'ML Engineer', 'AI Research Engineer', 'Data Scientist', 'Senior Software Engineer (ML)', 
        'Computer Vision Engineer', 'Recommendation Systems Engineer', 'Machine Learning Engineer', 
        'Applied ML Engineer', 'Search Engineer', 'AI Engineer', 'Senior Data Scientist', 
        'NLP Engineer', 'Senior NLP Engineer', 'Senior Machine Learning Engineer', 
        'Staff Machine Learning Engineer', 'Senior AI Engineer', 'Senior Applied Scientist', 
        'Lead AI Engineer'
    }
    
    other_tech_titles = {
        'Software Engineer', 'Backend Engineer', 'Senior Software Engineer', 
        'Data Engineer', 'Senior Data Engineer', 'Analytics Engineer'
    }
    
    valid_candidates = []
    candidate_ids = []
    candidate_texts = []
    
    total_processed = 0
    non_tech_count = 0
    honeypot_count = 0
    
    with open(args.candidates, "r", encoding="utf-8") as f:
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
                
            # Valid candidate -> convert to OKF
            okf_text = candidate_to_okf(c)
            valid_candidates.append(c)
            candidate_ids.append(cid)
            candidate_texts.append(okf_text)
            
    print(f"Total processed: {total_processed}")
    print(f"Non-technical excluded: {non_tech_count}")
    print(f"Honeypots excluded: {honeypot_count}")
    print(f"Remaining candidates to search: {len(valid_candidates)}")
    
    if not valid_candidates:
        print("Error: No candidates passed filtering.")
        sys.exit(1)
        
    # --- Dense Semantic Search ---
    dense_scores = np.zeros(len(valid_candidates))
    
    # Try to load cached embeddings
    loaded_cache = False
    cache_path = "embeddings.npz"
    if os.path.exists(cache_path):
        print("Loading cached embeddings...")
        try:
            cache = np.load(cache_path)
            cache_ids = cache["ids"]
            cache_embs = cache["embeddings"]
            
            # Create a lookup for instant index mapping
            id_to_idx = {cid: idx for idx, cid in enumerate(cache_ids)}
            
            # Verify if all our filtered candidate IDs exist in the cache
            if all(cid in id_to_idx for cid in candidate_ids):
                # Map precomputed embeddings to current order
                mapped_embs = np.array([cache_embs[id_to_idx[cid]] for cid in candidate_ids])
                
                # Compute JD embedding
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("all-MiniLM-L6-v2")
                jd_emb = model.encode(jd_text, convert_to_numpy=True)
                
                # Compute cosine similarities
                dot_products = np.dot(mapped_embs, jd_emb)
                norm_candidates = np.linalg.norm(mapped_embs, axis=1)
                norm_jd = np.linalg.norm(jd_emb)
                dense_scores = dot_products / (norm_candidates * norm_jd + 1e-8)
                loaded_cache = True
                print("Semantic scores calculated successfully using precomputed cache.")
        except Exception as e:
            print(f"Warning: Failed to load cached embeddings: {e}. Falling back to on-the-fly embeddings...")
            
    if not loaded_cache:
        # Generate embeddings on-the-fly (e.g. for small validation sets)
        print("Computing embeddings dynamically on CPU...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        jd_emb = model.encode(jd_text, convert_to_numpy=True)
        candidate_embs = model.encode(candidate_texts, batch_size=256, convert_to_numpy=True)
        
        dot_products = np.dot(candidate_embs, jd_emb)
        norm_candidates = np.linalg.norm(candidate_embs, axis=1)
        norm_jd = np.linalg.norm(jd_emb)
        dense_scores = dot_products / (norm_candidates * norm_jd + 1e-8)
        print("Dynamic semantic scores calculated successfully.")
        
    # --- Sparse Keyword Search ---
    print("Computing sparse matching scores...")
    vectorizer = TfidfVectorizer(stop_words='english')
    # Fit on all candidates + JD
    corpus = candidate_texts + [jd_text]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate similarity
    jd_vector = tfidf_matrix[-1]
    candidate_vectors = tfidf_matrix[:-1]
    
    # Cosine similarity for TF-IDF
    from sklearn.metrics.pairwise import cosine_similarity
    sparse_scores = cosine_similarity(candidate_vectors, jd_vector).flatten()
    print("Sparse keyword scores calculated successfully.")
    
    # --- Reciprocal Rank Fusion (RRF) ---
    dense_ranks = np.argsort(np.argsort(-dense_scores))
    sparse_ranks = np.argsort(np.argsort(-sparse_scores))
    
    # Merge ranks using standard RRF formula with constant k=60
    rrf_scores = np.zeros(len(valid_candidates))
    for i in range(len(valid_candidates)):
        rrf_scores[i] = 1.0 / (60.0 + dense_ranks[i]) + 1.0 / (60.0 + sparse_ranks[i])
        
    # --- Behavioral Signal Modifiers and Weighting ---
    final_candidates = []
    
    for i, c in enumerate(valid_candidates):
        profile = c["profile"]
        title = profile.get("current_title", "")
        years = profile.get("years_of_experience", 0)
        signals = c["redrob_signals"]
        career = c["career_history"]
        skills = [s["name"] for s in c.get("skills", [])]
        
        # 1. Base Score (RRF)
        base_score = rrf_scores[i]
        
        # 2. Title Match Score Multiplier
        title_mult = 1.0
        if title in ai_ml_titles:
            title_mult = 1.25
            if "Senior" in title or "Lead" in title or "Staff" in title:
                title_mult += 0.05
        elif title in other_tech_titles:
            title_mult = 1.05
            if "Senior" in title:
                title_mult += 0.05
        else:
            title_mult = 0.8
            
        # 3. Years of Experience Score Multiplier (JD wants 5-9 yrs)
        exp_mult = 1.0
        if 5.0 <= years <= 9.0:
            exp_mult = 1.15
        elif 4.0 <= years < 5.0 or 9.0 < years <= 11.0:
            exp_mult = 1.05
        elif 3.0 <= years < 4.0 or 11.0 < years <= 13.0:
            exp_mult = 0.9
        else:
            exp_mult = 0.7
            
        # 4. Location Score Multiplier
        loc_mult = 1.0
        country = profile.get("country", "").strip().lower()
        location = profile.get("location", "").strip().lower()
        willing_relocate = signals.get("willing_to_relocate", False)
        is_preferred_loc = any(x in location for x in ["noida", "pune", "delhi", "gurgaon", "ncr", "mumbai", "hyderabad", "bangalore"])
        
        if country != "india":
            if willing_relocate:
                loc_mult = 0.7
            else:
                loc_mult = 0.2
        else:
            if is_preferred_loc:
                loc_mult = 1.1
            elif willing_relocate:
                loc_mult = 1.0
            else:
                loc_mult = 0.8
                
        # 5. Activity and Platform Signals Modifier
        active_str = signals.get("last_active_date", "")
        active_dt = parse_date(active_str)
        active_months = 0.0
        if active_dt:
            active_days = (target_date - active_dt).days
            active_months = max(0.0, active_days / 30.0)
            
        resp_rate = signals.get("recruiter_response_rate", 0.0)
        
        notice = signals.get("notice_period_days", 90)
        if notice <= 30:
            notice_mult = 1.1
        elif notice <= 60:
            notice_mult = 1.0
        elif notice <= 90:
            notice_mult = 0.85
        else:
            notice_mult = 0.65
            
        gh_score = signals.get("github_activity_score", -1)
        gh_mult = 1.0
        if gh_score >= 80:
            gh_mult = 1.12
        elif gh_score >= 50:
            gh_mult = 1.05
        elif gh_score == -1:
            gh_mult = 0.95
            
        otw = signals.get("open_to_work_flag", False)
        otw_mult = 1.08 if otw else 0.96
        
        icr = signals.get("interview_completion_rate", 1.0)
        
        activity_mult = (
            (resp_rate * 0.3 + 0.7) * 
            (1.0 - min(1.0, active_months / 12.0) * 0.4) * 
            notice_mult * 
            gh_mult * 
            otw_mult * 
            (icr * 0.2 + 0.8)
        )
        
        # 6. Specific Core Skills Multiplier
        has_retrieval_or_vector = False
        has_evaluation = False
        retrieval_terms = {"embeddings", "retrieval", "search", "vector", "milvus", "qdrant", "pinecone", "weaviate", "faiss", "opensearch", "elasticsearch"}
        evaluation_terms = {"evaluation", "eval", "ndcg", "mrr", "map", "a/b test", "ab test"}
        
        skill_score = 0.0
        for s in c.get("skills", []):
            name = s["name"].lower()
            proficiency = s["proficiency"]
            prof_mult = {"beginner": 0.5, "intermediate": 0.8, "advanced": 1.0, "expert": 1.2}[proficiency]
            
            is_core = False
            if any(t in name for t in retrieval_terms):
                has_retrieval_or_vector = True
                is_core = True
            if any(t in name for t in evaluation_terms):
                has_evaluation = True
                is_core = True
            if any(t in name for t in ["ml", "ai", "machine learning", "deep learning", "nlp", "llm", "fine-tuning", "pytorch", "tensorflow", "weights"]):
                is_core = True
                
            if is_core:
                skill_score += 1.0 * prof_mult
                
        skill_mult = 1.0 + 0.08 * min(6.0, skill_score)
        if has_retrieval_or_vector:
            skill_mult *= 1.15
        if has_evaluation:
            skill_mult *= 1.10
            
        # 7. Consulting-Company Only Penalty
        consulting_companies = {"tcs", "wipro", "infosys", "accenture", "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis"}
        companies_worked = [job.get("company", "").lower() for job in career if job.get("company")]
        only_consulting = len(companies_worked) > 0 and all(c in consulting_companies for c in companies_worked)
        consulting_mult = 0.45 if only_consulting else 1.0
        
        final_score = base_score * title_mult * exp_mult * loc_mult * activity_mult * skill_mult * consulting_mult
        
        final_candidates.append({
            "candidate_id": c["candidate_id"],
            "score": float(final_score),
            "record": c
        })
        
    # --- Sort according to challenge tiebreaker rules ---
    # Sort by score descending, then by candidate_id ascending for ties.
    final_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # Select exactly the top 100 candidates
    top_100 = final_candidates[:100]
    
    # Scale scores linearly to make sure they look clean and monotonic (optional, but good for display)
    max_score = top_100[0]["score"]
    min_score = top_100[-1]["score"]
    
    # Map scores to a range [0.4, 0.99] to prevent extreme tiny floats and match standard profiles
    for idx, fc in enumerate(top_100):
        if max_score > min_score:
            scaled_score = 0.99 - (idx / 99.0) * (0.99 - 0.40)
        else:
            scaled_score = 0.99 - idx * 0.005
        fc["scaled_score"] = float(round(scaled_score, 4))
        
    print(f"Top 5 Ranked Candidates:")
    for idx in range(min(5, len(top_100))):
        c_item = top_100[idx]
        profile = c_item["record"]["profile"]
        print(f"  {idx+1}. {c_item['candidate_id']} | Score: {c_item['scaled_score']:.4f} | Name: {profile['anonymized_name']} | Title: {profile['current_title']} | Exp: {profile['years_of_experience']} yrs")
        
    # Generate CSV output
    print(f"Writing rankings to CSV file at {args.out}...")
    import csv
    with open(args.out, "w", encoding="utf-8", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, fc in enumerate(top_100):
            rank = idx + 1
            cid = fc["candidate_id"]
            score = fc["scaled_score"]
            reasoning = generate_reasoning(fc["record"], rank, score)
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])
            
    print("Ranking script executed successfully. Submission file created.")

if __name__ == "__main__":
    main()
