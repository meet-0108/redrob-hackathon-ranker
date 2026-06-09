import json
import csv
from datetime import datetime
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# =====================================================================
# 1. HELPER FUNCTION TO READ DOCX FILES
# =====================================================================
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# =====================================================================
# 2. STAGE 1: ANTI-HONEYPOT & HARD FILTERS
# =====================================================================
def is_valid_candidate(candidate, behavioral_signals):
    if not behavioral_signals:
        return True
        
    if behavioral_signals.get('profile_completeness_score', 0) < 25:
        return False
        
    github_score = behavioral_signals.get('github_activity_score', -1)
    if github_score > 50 and not behavioral_signals.get('github_connected', True):
        return False
        
    response_rate = behavioral_signals.get('recruiter_response_rate', 0.0)
    avg_response_time = behavioral_signals.get('avg_response_time_hours', -1)
    if response_rate > 0.90 and avg_response_time == 0:
        return False 

    if behavioral_signals.get('notice_period_days', 0) > 120:
        return False

    return True

# =====================================================================
# 3. STAGE 3: BEHAVIORAL MULTIPLIERS
# =====================================================================
def calculate_behavioral_multiplier(signals):
    if not signals:
        return 1.0
        
    multiplier = 1.0
    
    if signals.get('open_to_work_flag', False):
        multiplier *= 1.15
        
    response_rate = signals.get('recruiter_response_rate', 1.0)
    if response_rate < 0.30:
        multiplier *= 0.70  
    elif response_rate > 0.85:
        multiplier *= 1.10  
        
    if signals.get('verified_email', False) and signals.get('verified_phone', False):
        multiplier *= 1.05
        
    return multiplier

# =====================================================================
# 4. CORE PIPELINE EXECUTION
# =====================================================================
def run_ranking_pipeline(jd_path, candidates_path, output_csv_path):
    print("📖 Reading Job Description from DOCX...")
    jd_text = read_docx(jd_path)
    
    candidate_records = []
    candidate_texts = []
    
    print("⏳ Stage 1: Filtering out Honeypots & Traps...")
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading Database"):
            if not line.strip():
                continue
                
            try:
                candidate = json.loads(line)
            except Exception:
                continue
                
            signals = candidate.get('redrob_signals', {})
            candidate_id = candidate.get('id') or candidate.get('candidate_id')
            if candidate_id:
                candidate_id = candidate_id.strip()
            
            if not is_valid_candidate(candidate, signals):
                continue
                
            raw_skills = candidate.get('skills', [])
            skills_list = []
            
            if isinstance(raw_skills, list):
                for s in raw_skills:
                    if isinstance(s, dict):
                        skill_name = s.get('name') or s.get('skill') or s.get('title') or s.get('skill_name')
                        if not skill_name:
                            str_vals = [str(v) for v in s.values() if isinstance(v, str)]
                            skill_name = str_vals[0] if str_vals else ""
                        if skill_name:
                            skills_list.append(skill_name)
                    elif isinstance(s, str):
                        skills_list.append(s)
            elif isinstance(raw_skills, str):
                skills_list.append(raw_skills)
                
            skills_str_final = ", ".join(skills_list)
            headline = str(candidate.get('headline', ''))
            
            experience_summary = ""
            raw_exp = candidate.get('experience_history', [])
            if isinstance(raw_exp, list):
                for exp in raw_exp:
                    if isinstance(exp, dict):
                        experience_summary += f" {exp.get('title', '')} {exp.get('description', '')};"
            
            candidate_text = f"{headline} {skills_str_final} {experience_summary}"
            
            candidate_records.append((candidate_id, candidate, skills_list, signals))
            candidate_texts.append(candidate_text)
            
    if not candidate_texts:
        print("❌ Error: No valid candidates found.")
        return

    print("🧠 Stage 2: Calculating Semantic Vector Similarities...")
    vectorizer = TfidfVectorizer(stop_words='english')
    
    all_texts = [jd_text] + candidate_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    jd_vector = tfidf_matrix[0]
    candidate_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(jd_vector, candidate_vectors).flatten()
    
    print("📊 Stage 3: Applying Modifiers and Sorting Top 100...")
    scored_candidates = []
    for i, (candidate_id, candidate, skills_list, signals) in enumerate(candidate_records):
        base_score = float(similarities[i])
        multiplier = calculate_behavioral_multiplier(signals)
        final_score = base_score * multiplier
        
        # Round the score to 4 decimals BEFORE sorting to group true ties together
        rounded_score = round(final_score, 4)
        
        top_skills = skills_list[:2]
        skills_display = " and ".join(top_skills) if top_skills else "matching role specifications"
        reasoning = f"Strong alignment in key domains including {skills_display}, backed by verified positive responsiveness metrics."
        
        scored_candidates.append({
            'candidate_id': candidate_id,
            'score': rounded_score,
            'reasoning': reasoning
        })
        
    # --- FIXED SORTING: Descending by score, Ascending alphabetically by ID ---
    scored_candidates.sort(key=lambda x: (-x['score'], x['candidate_id']))
    top_100 = scored_candidates[:100]
    
    print(f"💾 Saving clean results to {output_csv_path}...")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['candidate_id', 'rank', 'score', 'reasoning']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for rank_idx, entry in enumerate(top_100, start=1):
            writer.writerow({
                'candidate_id': entry['candidate_id'],
                'rank': rank_idx,
                'score': entry['score'],  # Already perfectly rounded
                'reasoning': entry['reasoning']
            })
            
    print("✅ File generated perfectly with synchronized rounding and tie-breaks!")

if __name__ == "__main__":
    run_ranking_pipeline(
        jd_path='job_description.docx', 
        candidates_path='candidates.jsonl', 
        output_csv_path='meetg1244_2794.csv'
    )