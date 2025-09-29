import pandas as pd
import spacy
import re
from typing import List, Dict
from collections import Counter
import numpy as np

# CHANGE 1: Load the better model
try:
    nlp = spacy.load("en_core_web_lg")
    print("✅ Loaded enhanced NLP model for scoring: en_core_web_lg")
    HAS_VECTORS = True
except OSError:
    nlp = spacy.load("en_core_web_sm")
    print("⚠️  Using basic model for scoring: en_core_web_sm")
    HAS_VECTORS = False

def score_skills(found_skills: List[str], required_skills: List[str]) -> float:
    """Enhanced skill scoring with semantic matching"""
    if not required_skills:
        return 0.8 if found_skills else 0.0
    
    # Direct matches
    found_lower = [skill.lower() for skill in found_skills]
    required_lower = [skill.lower() for skill in required_skills]
    
    direct_matches = len(set(found_lower) & set(required_lower))
    
    # CHANGE 2: Add semantic matching if we have word vectors
    semantic_matches = 0
    if HAS_VECTORS and nlp.vocab.vectors_length > 0:
        for req_skill in required_lower:
            if req_skill in found_lower:
                continue  # Already counted in direct matches
            
            req_doc = nlp(req_skill)
            if not req_doc.has_vector:
                continue
            
            # Check semantic similarity with found skills
            for found_skill in found_lower:
                if found_skill in required_lower:
                    continue  # Skip direct matches
                
                found_doc = nlp(found_skill)
                if found_doc.has_vector:
                    similarity = req_doc.similarity(found_doc)
                    if similarity > 0.7:  # High similarity threshold
                        semantic_matches += 0.7  # Partial credit for semantic match
                        break
    
    total_matches = direct_matches + semantic_matches
    match_score = min(total_matches / len(required_skills), 1.0)
    return match_score

def score_name(name: str) -> float:
    """Enhanced name scoring"""
    if not name:
        return 0.0
    
    words = name.strip().split()
    
    # Full name (2+ words)
    if len(words) >= 2:
        return 0.9
    # Single name
    elif len(words) == 1 and len(words[0]) > 2:
        return 0.5
    else:
        return 0.0

def score_contacts(contacts: dict) -> float:
    """Enhanced contact scoring"""
    if not isinstance(contacts, dict):
        return 0.0
    
    score = 0.0
    
    # Email is most important
    if contacts.get("email") and "@" in str(contacts.get("email")):
        score += 0.4
    
    # Phone number
    phone = str(contacts.get("phone", ""))
    if phone and len(re.sub(r'[^\d]', '', phone)) >= 10:
        score += 0.3
    
    # Professional profiles
    if contacts.get("linkedin"):
        score += 0.2
    
    if contacts.get("github"):
        score += 0.1
    
    return min(score, 1.0)

# CHANGE 3: Enhanced semantic similarity with better model
def semantic_similarity(text: str, keywords: List[str]) -> float:
    """Enhanced semantic similarity using better word vectors"""
    if not text or not keywords:
        return 0.0
    
    if not HAS_VECTORS or not nlp.vocab.vectors_length:
        # Fallback to simple keyword matching
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return min(matches / len(keywords), 1.0)
    
    try:
        # Process text (limit length for performance)
        doc_text = nlp(text[:3000].lower())  # First 3000 chars
        doc_keywords = nlp(" ".join(keywords).lower())
        
        if doc_text.has_vector and doc_keywords.has_vector:
            similarity = doc_text.similarity(doc_keywords)
            return max(0.0, min(similarity, 1.0))
        else:
            # Fallback if no vectors available
            return 0.0
    except Exception:
        return 0.0

def score_experience(experience_data, required_years: int = 0) -> float:
    """Enhanced experience scoring"""
    if not experience_data:
        return 0.0
    
    score = 0.0
    
    # If experience_data is a list of experiences
    if isinstance(experience_data, list):
        score += min(len(experience_data) / 3.0, 0.5)  # Number of jobs
        
        # Try to extract years from experience entries
        total_years = 0
        for exp in experience_data:
            if isinstance(exp, dict):
                duration = exp.get('duration', '')
                years = extract_years_from_text(duration)
                total_years += years
            elif isinstance(exp, str):
                years = extract_years_from_text(exp)
                total_years += years
        
        # Score based on years of experience
        if total_years > 0:
            if required_years > 0:
                year_score = min(total_years / required_years, 1.0)
                score += year_score * 0.5
            else:
                score += min(total_years / 5.0, 0.5)  # Assume 5 years is good
    
    # If experience_data is just text
    elif isinstance(experience_data, str):
        # Basic scoring based on content length and keywords
        words = len(experience_data.split())
        score += min(words / 100, 0.3)  # Content richness
        
        # Look for experience indicators
        exp_keywords = ['developed', 'managed', 'led', 'implemented', 'designed', 'built']
        keyword_matches = sum(1 for kw in exp_keywords if kw.lower() in experience_data.lower())
        score += min(keyword_matches / len(exp_keywords), 0.3)
        
        # Extract years from text
        years = extract_years_from_text(experience_data)
        if years > 0:
            if required_years > 0:
                year_score = min(years / required_years, 1.0)
                score += year_score * 0.4
            else:
                score += min(years / 5.0, 0.4)
    
    return min(score, 1.0)

def extract_years_from_text(text: str) -> float:
    """Extract years of experience from text"""
    if not text:
        return 0.0
    
    text_lower = text.lower()
    
    # Pattern 1: "X years" or "X yrs"
    years_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', text_lower)
    if years_match:
        return float(years_match.group(1))
    
    # Pattern 2: Date ranges like "2020-2024" or "Jan 2020 - Dec 2023"
    date_patterns = [
        r'(\d{4})\s*[-–]\s*(\d{4})',  # 2020-2024
        r'(\d{4})\s*[-–]\s*(?:present|current|now)',  # 2020-present
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text_lower)
        if match:
            start_year = int(match.group(1))
            if 'present' in match.group(0) or 'current' in match.group(0) or 'now' in match.group(0):
                end_year = 2024  # Current year
            else:
                end_year = int(match.group(2))
            
            years_diff = max(0, end_year - start_year)
            return float(years_diff)
    
    return 0.0

# CHANGE 4: Enhanced job description parsing
def extract_requirements_from_jd(job_description: str) -> Dict[str, any]:
    """Extract requirements from job description using enhanced NLP"""
    requirements = {
        'skills': [],
        'experience_years': 0,
        'education': [],
        'keywords': []
    }
    
    if not job_description:
        return requirements
    
    text_lower = job_description.lower()
    
    # Extract required years of experience
    exp_patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)',
        r'(?:minimum|min|at least)\s*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\s*to\s*(\d+)\s*(?:years?|yrs?)',
    ]
    
    for pattern in exp_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if len(match.groups()) == 2:  # Range like "3 to 5 years"
                requirements['experience_years'] = int(match.group(1))
            else:
                requirements['experience_years'] = int(match.group(1))
            break
    
    # Extract skills using enhanced patterns
    skill_patterns = [
        r'\b(?:python|java|javascript|js|react|angular|vue|node\.?js|django|flask)\b',
        r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins|git)\b',
        r'\b(?:sql|mysql|postgresql|mongodb|redis|oracle)\b',
        r'\b(?:html|css|bootstrap|tailwind|scss)\b',
        r'\b(?:machine learning|ml|ai|tensorflow|pytorch|pandas|numpy)\b',
    ]
    
    found_skills = set()
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        found_skills.update(matches)
    
    # Use NLP to find additional technical terms
    if HAS_VECTORS:
        doc = nlp(job_description)
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                not token.is_stop and 
                len(token.text) > 2 and
                token.text.lower() not in found_skills):
                # Check if it's a technical term by comparing with known skills
                if any(keyword in token.text.lower() for keyword in 
                      ['tech', 'dev', 'data', 'web', 'app', 'system', 'software']):
                    found_skills.add(token.text.lower())
    
    requirements['skills'] = list(found_skills)
    
    # Extract education requirements
    edu_patterns = [
        r'\b(?:bachelor|master|phd|doctorate|degree|diploma)\b',
        r'\b(?:b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|mba)\b',
    ]
    
    for pattern in edu_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        requirements['education'].extend(matches)
    
    # Extract key keywords for semantic matching
    if HAS_VECTORS:
        doc = nlp(job_description)
        important_tokens = []
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 3 and
                token.pos_ in ['NOUN', 'ADJ', 'VERB']):
                important_tokens.append(token.lemma_)
        
        # Keep most frequent/important terms
        token_freq = Counter(important_tokens)
        requirements['keywords'] = [token for token, freq in token_freq.most_common(20)]
    
    return requirements

# CHANGE 5: Enhanced overall scoring function
def overall_score(resume: dict, job_description: str) -> float:
    """Enhanced overall scoring using better NLP"""
    # Extract requirements from job description
    jd_requirements = extract_requirements_from_jd(job_description)
    
    # Calculate individual scores
    skills_score = score_skills(
        resume.get("skills", []), 
        jd_requirements.get('skills', [])
    )
    
    name_score = score_name(resume.get("name", ""))
    
    contacts_score = score_contacts(resume.get("contacts", {}))
    
    experience_score = score_experience(
        resume.get("experience", []), 
        jd_requirements.get('experience_years', 0)
    )
    
    # Enhanced semantic scoring
    semantic_score = semantic_similarity(
        resume.get("raw_text", ""), 
        jd_requirements.get('keywords', [])
    )
    
    # Education scoring
    education_score = 0.5  # Default if no specific requirements
    if jd_requirements.get('education'):
        resume_education = resume.get("education", "")
        if isinstance(resume_education, str) and resume_education:
            edu_keywords = jd_requirements['education']
            edu_matches = sum(1 for edu in edu_keywords if edu.lower() in resume_education.lower())
            education_score = min(edu_matches / len(edu_keywords), 1.0) if edu_keywords else 0.5
        elif isinstance(resume_education, list) and resume_education:
            education_score = 0.7  # Has education info
    
    # CHANGE 6: Adaptive weighting based on job requirements
    weights = {
        'skills': 0.30,
        'experience': 0.25,
        'semantic': 0.20,
        'education': 0.15,
        'name': 0.05,
        'contacts': 0.05
    }
    
    # Increase skills weight if many skills required
    if len(jd_requirements.get('skills', [])) > 8:
        weights['skills'] = 0.40
        weights['semantic'] = 0.15
    
    # Increase experience weight if high experience required
    if jd_requirements.get('experience_years', 0) >= 5:
        weights['experience'] = 0.35
        weights['skills'] = 0.25
    
    # Calculate weighted score
    total_score = (
        weights['skills'] * skills_score +
        weights['experience'] * experience_score +
        weights['semantic'] * semantic_score +
        weights['education'] * education_score +
        weights['name'] * name_score +
        weights['contacts'] * contacts_score
    )
    
    return round(total_score, 3)

def score_dataframe(df: pd.DataFrame, job_description: str) -> pd.DataFrame:
    """Enhanced dataframe scoring with detailed metrics"""
    if df.empty:
        return df
    
    scored_df = df.copy()
    jd_requirements = extract_requirements_from_jd(job_description)
    
    # Initialize new columns
    scored_df['score'] = 0.0
    scored_df['skill_match'] = 0.0
    scored_df['experience_match'] = 0.0
    scored_df['semantic_match'] = 0.0
    scored_df['matched'] = None
    scored_df['missing'] = None
    scored_df['skill_gaps'] = 0
    
    for idx, row in df.iterrows():
        try:
            # Calculate overall score
            total_score = overall_score(row.to_dict(), job_description)
            scored_df.at[idx, 'score'] = total_score * 100  # Convert to percentage
            
            # Calculate individual component scores
            resume_skills = row.get('skills', [])
            required_skills = jd_requirements.get('skills', [])
            
            skill_score = score_skills(resume_skills, required_skills)
            scored_df.at[idx, 'skill_match'] = skill_score * 100
            
            exp_score = score_experience(
                row.get('experience', []), 
                jd_requirements.get('experience_years', 0)
            )
            scored_df.at[idx, 'experience_match'] = exp_score * 100
            
            semantic_score = semantic_similarity(
                row.get('raw_text', ''), 
                jd_requirements.get('keywords', [])
            )
            scored_df.at[idx, 'semantic_match'] = semantic_score * 100
            
            # Find matched and missing skills
            if isinstance(resume_skills, list) and isinstance(required_skills, list):
                resume_skills_lower = [s.lower() for s in resume_skills]
                required_skills_lower = [s.lower() for s in required_skills]
                
                matched_skills = []
                missing_skills = []
                
                for req_skill in required_skills_lower:
                    found = False
                    for resume_skill in resume_skills_lower:
                        if req_skill == resume_skill or req_skill in resume_skill or resume_skill in req_skill:
                            matched_skills.append(req_skill)
                            found = True
                            break
                    
                    if not found:
                        # Check semantic similarity for missing skills
                        if HAS_VECTORS and nlp.vocab.vectors_length > 0:
                            req_doc = nlp(req_skill)
                            if req_doc.has_vector:
                                for resume_skill in resume_skills_lower:
                                    resume_doc = nlp(resume_skill)
                                    if resume_doc.has_vector and req_doc.similarity(resume_doc) > 0.7:
                                        matched_skills.append(f"{req_skill} (~{resume_skill})")
                                        found = True
                                        break
                    
                    if not found:
                        missing_skills.append(req_skill)
                
                scored_df.at[idx, 'matched'] = matched_skills
                scored_df.at[idx, 'missing'] = missing_skills
                scored_df.at[idx, 'skill_gaps'] = len(missing_skills)
            else:
                scored_df.at[idx, 'matched'] = []
                scored_df.at[idx, 'missing'] = required_skills
                scored_df.at[idx, 'skill_gaps'] = len(required_skills)
                
        except Exception as e:
            print(f"Error scoring resume {row.get('file', 'unknown')}: {e}")
            scored_df.at[idx, 'score'] = 0.0
            scored_df.at[idx, 'skill_match'] = 0.0
            scored_df.at[idx, 'experience_match'] = 0.0
            scored_df.at[idx, 'semantic_match'] = 0.0
            scored_df.at[idx, 'matched'] = []
            scored_df.at[idx, 'missing'] = []
            scored_df.at[idx, 'skill_gaps'] = 0
    
    # Sort by overall score (descending)
    scored_df = scored_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return scored_df

# CHANGE 7: Enhanced summarization with better model
def summarize(text: str, max_sentences: int = 3) -> str:
    """Enhanced text summarization using better NLP model"""
    if not text or len(text.strip()) < 50:
        return text
    
    try:
        # Use NLP model for better sentence segmentation
        doc = nlp(text[:2000])  # First 2000 characters
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        
        # CHANGE 8: Score sentences based on importance
        if HAS_VECTORS and nlp.vocab.vectors_length > 0:
            # Create document vector for the whole text
            doc_vector = doc.vector
            
            sentence_scores = []
            for sent in sentences[:10]:  # Limit to first 10 sentences
                sent_doc = nlp(sent)
                if sent_doc.has_vector:
                    # Score based on similarity to document and sentence length
                    similarity = np.dot(doc_vector, sent_doc.vector) / (np.linalg.norm(doc_vector) * np.linalg.norm(sent_doc.vector))
                    length_score = min(len(sent.split()) / 20, 1.0)  # Prefer medium-length sentences
                    final_score = similarity * 0.7 + length_score * 0.3
                    sentence_scores.append((sent, final_score))
                else:
                    sentence_scores.append((sent, 0.1))
            
            # Sort by score and take top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent for sent, score in sentence_scores[:max_sentences]]
            
            # Maintain original order
            result_sentences = []
            for sent in sentences:
                if sent in top_sentences:
                    result_sentences.append(sent)
                if len(result_sentences) == max_sentences:
                    break
            
            return " ".join(result_sentences)
        else:
            # Fallback: just take first N sentences
            return ". ".join(sentences[:max_sentences]) + "."
            
    except Exception as e:
        print(f"Error in summarization: {e}")
        # Simple fallback
        sentences = text.split(".")
        return ". ".join(sentences[:max_sentences]) + "."

# CHANGE 9: Add performance testing function
def test_enhanced_scoring():
    """Test enhanced scoring functionality"""
    print("🧪 Testing Enhanced Scoring System...")
    
    # Sample data
    sample_resume = {
        'name': 'John Smith',
        'skills': ['Python', 'Django', 'React', 'AWS', 'Docker'],
        'experience': [
            {'role': 'Senior Developer', 'company': 'Tech Corp', 'duration': '3 years'},
            {'role': 'Developer', 'company': 'StartupXYZ', 'duration': '2 years'}
        ],
        'contacts': {'email': 'john@example.com', 'phone': '+1-555-0123'},
        'raw_text': '''John Smith is a Senior Python Developer with 5 years of experience 
        developing web applications using Django and React. Strong experience with AWS cloud services 
        and Docker containerization. Led development teams and implemented agile methodologies.'''
    }
    
    sample_jd = '''
    Senior Python Developer position requiring 4+ years experience.
    Required skills: Python, Django, React, AWS, PostgreSQL.
    Experience with Docker and Kubernetes preferred.
    Bachelor's degree in Computer Science required.
    '''
    
    # Test individual scoring functions
    jd_req = extract_requirements_from_jd(sample_jd)
    print(f"✅ JD Requirements: {jd_req}")
    
    skill_score = score_skills(sample_resume['skills'], jd_req['skills'])
    print(f"✅ Skill Score: {skill_score:.2f}")
    
    overall = overall_score(sample_resume, sample_jd)
    print(f"✅ Overall Score: {overall:.2f}")
    
    # Test summarization
    summary = summarize(sample_resume['raw_text'])
    print(f"✅ Summary: {summary}")
    
    print("🎉 Enhanced scoring test completed!")

if __name__ == "__main__":
    test_enhanced_scoring()