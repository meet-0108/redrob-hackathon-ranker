import json
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define your static submission targets
INPUT_JSONL = "candidates.jsonl"
OUTPUT_CSV = "meetg1244_2794.csv"

# The baseline Job Description context evaluated by the platform
JOB_DESCRIPTION = "Looking for a Software Engineer with deep Python, React, and Full-Stack lifecycle architecture expertise."

def advanced_behavioral_scoring(base_tfidf, signals):
    """
    Applies non-linear multiplier logic directly derived from the 
    Official Redrob Behavioral Signals Specifications.
    """
    score = base_tfidf
    
    # 1. ANTI-HONEYPOT TRAP: Drastic penalization for unverified accounts
    is_verified = signals.get('verified_email', True) and signals.get('verified_phone', True)
    if not is_verified:
        score *= 0.35  # 65% drop to force fake profiles out of the top tier
        
    # 2. KEYWORD STUFFER COUNTER-MEASURE: Profile Completeness Validation
    completeness = signals.get('profile_completeness_score', 100)
    score *= (0.6 + 0.4 * (completeness / 100.0))
    
    # 3. AVAILABILITY ACCELERATOR: High Intent Status Flags
    if signals.get('open_to_work_flag', False):
        score *= 1.25  # 25% amplification for immediate hire clarity
        
    # 4. RESPONSIVENESS MULTIPLIER: Linear scaling of recruiter engagement
    response_rate = signals.get('recruiter_response_rate', 0.0)
    score *= (1.0 + (response_rate * 0.20))  # Up to 20% performance bonus
    
    # 5. TECHNICAL EXECUTION BOOST: Open-Source Contribution Weighting
    github_score = signals.get('github_activity_score', -1)
    if github_score > 0:
        score *= (1.0 + (github_score / 100.0) * 0.10)  # Up to 10% bonus for real builders
        
    # 6. PIPELINE RELIABILITY MODIFIER: Interview Completion Consistency
    interview_rate = signals.get('interview_completion_rate', 0.0)
    score *= (1.0 + (interview_rate * 0.15))  # Up to 15% bonus for historical consistency
    
    return round(score, 6)

def run_elite_pipeline():
    print("🚀 Initializing Redrob Elite Ranking Engine Pipeline...")
    
    candidate_records = []
    corpus_text = [JOB_DESCRIPTION]
    
    if not os.path.exists(INPUT_JSONL):
        print(f"❌ Error: {INPUT_JSONL} not found in current working directory.")
        return
    
    # Fast, memory-isolated line-by-line streaming extraction
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                print(f"⚠️ Skipping malformed JSON line {line_num}: {e}")
                continue
            
            # Extract identifiers and metadata safely
            c_id = data.get('candidate_id') or data.get('id')
            signals = data.get('redrob_signals', {})
            
            # Normalize complex multi-type skills array data structures safely
            skills_raw = data.get('skills', [])
            skills_processed = []
            
            if isinstance(skills_raw, list):
                for item in skills_raw:
                    if isinstance(item, dict):
                        extracted_skill = item.get('name') or item.get('skill') or (list(item.values())[0] if item.values() else "")
                        skills_processed.append(str(extracted_skill))
                    else:
                        skills_processed.append(str(item))
                skills_str = ", ".join(skills_processed)
            else:
                skills_str = str(skills_raw)
                
            headline = str(data.get('headline', ''))
            profile_text = f"{headline} {skills_str}"
            corpus_text.append(profile_text)
            
            candidate_records.append({
                'id': c_id,
                'signals': signals,
                'headline': headline
            })
            
    print(f"📦 Successfully parsed {len(candidate_records)} records. Vectorizing alignment matrix...")
    
    # Compute baseline high-frequency TF-IDF matrix values
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus_text)
    
    # Extract calculated text similarities
    jd_vector = tfidf_matrix[0]
    candidate_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(jd_vector, candidate_vectors).flatten()
    
    # Apply the advanced behavioral signal functions
    scored_pool = []
    for idx, cand in enumerate(candidate_records):
        base_similarity = float(similarities[idx])
        final_calculated_score = advanced_behavioral_scoring(base_similarity, cand['signals'])
        
        scored_pool.append({
            'candidate_id': cand['id'],
            'score': final_calculated_score,
            'response_rate': cand['signals'].get('recruiter_response_rate', 0.0),
            'open_to_work': cand['signals'].get('open_to_work_flag', False),
            'github_score': cand['signals'].get('github_activity_score', 0),
            'is_verified': cand['signals'].get('verified_email', True) and cand['signals'].get('verified_phone', True)
        })
        
    print("⚖️ Sorting leaderboard via strict score descending and candidate_id ascending tie-breakers...")
    scored_pool.sort(key=lambda x: (-x['score'], str(x['candidate_id'])))
    
    print(f"💾 Slicing to the absolute top 100 entries and writing directly to {OUTPUT_CSV}...")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        
        # FIXED: Added the [:100] slice here to restrict the final CSV output to rows 2-101 exactly
        for index, item in enumerate(scored_pool[:100]):
            rank_val = index + 1
            
            # Formulate dynamic explanations narrative
            reasons = ["Strong keyword and skills alignment with baseline requirement specs."]
            if not item['is_verified']:
                reasons.append("Rank penalized due to unverified user credentials.")
            else:
                if item['open_to_work']:
                    reasons.append("High hiring velocity candidate marked as actively looking.")
                if item['response_rate'] > 0.75:
                    reasons.append("Demonstrates premier platform engagement and responsiveness metrics.")
                if item['github_score'] > 60:
                    reasons.append("Possesses an active open-source contribution record on GitHub.")
            
            reasoning_str = " ".join(reasons)
            writer.writerow([item['candidate_id'], rank_val, item['score'], reasoning_str])
            
    print("🏆 Done! Elite sorting script complete. Generated file matches the target schema flawlessly.")

if __name__ == "__main__":
    run_elite_pipeline()