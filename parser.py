import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import spacy

# PDF/DOCX loaders
import fitz  # PyMuPDF
from docx import Document

# Load the better model with proper error handling
try:
    nlp = spacy.load("en_core_web_lg")
    print("✅ Loaded enhanced NLP model: en_core_web_lg")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
        print("⚠️ Fallback to basic model: en_core_web_sm")
    except OSError:
        print("❌ No spaCy model found! Please install one:")
        print("python -m spacy download en_core_web_sm")
        raise

# Enhanced section headers
SECTION_HEADERS_EN = [
    "professional summary", "summary", "profile", "objective", "about me", "overview",
    "work experience", "professional experience", "experience", "employment history", "career history", "employment",
    "education", "academic background", "academic qualifications", "educational background", "qualifications",
    "skills", "technical skills", "core competencies", "key skills", "expertise", "technologies", "technical expertise",
    "projects", "key projects", "notable projects", "personal projects", "project experience",
    "certifications", "certificates", "professional certifications", "licenses", "credentials",
    "achievements", "accomplishments", "awards", "honors", "recognition",
    "contact information", "contact details", "personal information", "contact", "personal details",
    "languages", "language skills", "spoken languages", "linguistic skills"
]

# Kannada headers
SECTION_HEADERS_KN = [
    "ಶಿಕ್ಷಣ", "ಅನುಭವ", "ಪ್ರಾಜೆಕ್ಟ್", "ಕೌಶಲ್ಯ", "ಸಾರಾಂಶ", "ಸಂಪರ್ಕ"
]

def load_text(file_path: Path) -> str:
    """Load text from various file formats"""
    try:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            with fitz.open(file_path) as doc:
                text = []
                for page in doc:
                    text.append(page.get_text())
            return "\n".join(text)
        elif ext == ".docx":
            doc = Document(str(file_path))
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return ""

def clean_text(text: str) -> str:
    """Enhanced text cleaning"""
    if not text:
        return ""
    
    text = text.replace("\x00", " ").replace("\ufeff", "")  # Remove BOM and null bytes
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_kannada(text: str) -> bool:
    """Check if text contains Kannada characters"""
    if not text:
        return False
    return bool(re.search(r"[\u0C80-\u0CFF]", text))

def split_sections(text: str) -> Dict[str, str]:
    """Fixed section splitting - resolves the variable error"""
    if not text:
        return {"body": ""}
    
    headers = SECTION_HEADERS_EN + SECTION_HEADERS_KN
    sections = {}
    
    # Create patterns for each header
    all_matches = []
    
    for header in headers:  # FIXED: Use 'header' instead of undefined 'h'
        patterns = [
            rf"(?mi)^[\s]*{re.escape(header)}[\s]*:?[\s]*$",
            rf"(?mi)^[\s]*{re.escape(header.upper())}[\s]*:?[\s]*$",
            rf"(?mi)^[\s]*{re.escape(header)}[\s]*[-=_]+[\s]*$",
        ]
        
        for pattern in patterns:
            try:
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    all_matches.append((match, header))  # FIXED: Use 'header' instead of 'h'
            except Exception as e:
                print(f"Pattern error for header '{header}': {e}")
                continue
    
    # Sort matches by position
    all_matches.sort(key=lambda x: x[0].start())
    
    if not all_matches:
        sections["body"] = text
        return sections
    
    # Process matches
    for i, (match, header_name) in enumerate(all_matches):
        # Add content before first section as intro
        if i == 0:
            pre_content = text[:match.start()].strip()
            if pre_content:
                sections["intro"] = pre_content
        
        # Extract content for this section
        start_pos = match.end()
        if i + 1 < len(all_matches):
            end_pos = all_matches[i + 1][0].start()
        else:
            end_pos = len(text)
        
        content = text[start_pos:end_pos].strip()
        if content:
            sections[header_name.lower()] = content
    
    return sections

def extract_entities_nlp(text: str) -> Dict[str, List[str]]:
    """Extract entities using NLP model with proper error handling"""
    if not text:
        return {"PERSON": [], "ORG": [], "DATE": [], "EMAIL": [], "PHONE": []}
    
    try:
        # Process text in chunks to avoid memory issues
        text_chunk = text[:5000]  # First 5000 chars
        doc = nlp(text_chunk)
        
        entities = {
            "PERSON": [], "ORG": [], "DATE": [], "EMAIL": [], "PHONE": []
        }
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in entities:
                entity_text = ent.text.strip()
                if len(entity_text) > 1 and entity_text not in entities[ent.label_]:
                    entities[ent.label_].append(entity_text)
        
        # Enhanced regex patterns for contact info
        # Phone patterns
        phone_patterns = [
            r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # US format
            r"(?:\+\d{1,3}[-.\s]?)?\d{10,15}",  # International
            r"\+91[-.\s]?[6-9]\d{9}",  # Indian format
        ]
        
        for pattern in phone_patterns:
            try:
                phones = re.findall(pattern, text)
                for phone in phones:
                    phone_clean = re.sub(r'[^\d+]', '', phone)
                    if 10 <= len(phone_clean) <= 15:
                        entities["PHONE"].append(phone.strip())
            except Exception:
                continue
        
        # Email pattern
        try:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            entities["EMAIL"].extend(emails)
        except Exception:
            pass
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
        
    except Exception as e:
        print(f"Error in NLP entity extraction: {e}")
        return {"PERSON": [], "ORG": [], "DATE": [], "EMAIL": [], "PHONE": []}

def load_skills_dict(skills_path: Path) -> List[str]:
    """Load skills dictionary with error handling"""
    try:
        if skills_path.exists():
            with open(skills_path, 'r', encoding='utf-8') as f:
                skills = json.load(f)
                return skills if isinstance(skills, list) else []
        else:
            print(f"Skills file not found: {skills_path}")
            return []
    except Exception as e:
        print(f"Error loading skills: {e}")
        return []

def extract_skills(text: str, skills_list: List[str]) -> Tuple[List[str], float]:
    """Enhanced skills extraction with proper error handling"""
    if not text or not skills_list:
        return [], 0.1
    
    try:
        found_skills = set()
        text_lower = text.lower()
        
        # Method 1: Exact matching with word boundaries
        for skill in skills_list:
            if not skill:  # Skip empty skills
                continue
            
            skill_lower = skill.lower()
            try:
                pattern = rf"\b{re.escape(skill_lower)}\b"
                if re.search(pattern, text_lower):
                    found_skills.add(skill)
            except Exception:
                # Fallback to simple substring matching
                if skill_lower in text_lower:
                    found_skills.add(skill)
        
        # Method 2: Technology stack detection
        tech_patterns = {
            'python': r'\b(?:python|py|django|flask|fastapi|pandas|numpy)\b',
            'javascript': r'\b(?:javascript|js|node\.?js|react|angular|vue)\b',
            'java': r'\b(?:java|spring|hibernate|maven|gradle)\b',
            'database': r'\b(?:sql|mysql|postgresql|mongodb|redis|oracle)\b',
            'cloud': r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins)\b',
        }
        
        for base_skill, pattern in tech_patterns.items():
            try:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    related_skills = [s for s in skills_list if s and base_skill in s.lower()]
                    found_skills.update(related_skills[:3])  # Limit to avoid over-matching
            except Exception:
                continue
        
        # Calculate confidence score
        skill_count = len(found_skills)
        if skill_count == 0:
            confidence = 0.05
        elif skill_count <= 5:
            confidence = 0.3 + (skill_count * 0.1)
        else:
            max_expected = len(skills_list) // 3
            confidence = min(0.4 + (skill_count / max_expected) * 0.5, 0.95)
        
        return sorted(list(found_skills)), confidence
        
    except Exception as e:
        print(f"Error in skills extraction: {e}")
        return [], 0.1

def extract_name_advanced(entities: Dict[str, List[str]], text: str) -> str:
    """Enhanced name extraction with better error handling"""
    if not text:
        return ""
    
    try:
        # Method 1: Use NLP detected PERSON entities
        if entities.get("PERSON"):
            names = entities["PERSON"]
            for name in names:
                if not name:
                    continue
                words = name.split()
                if 2 <= len(words) <= 4 and all(w and w[0].isupper() and w.isalpha() for w in words):
                    return name
            # Return first valid name if no perfect match
            return names[0] if names[0] else ""
        
        # Method 2: Pattern-based extraction from first few lines
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if not line or '@' in line or re.search(r'\d{3,}', line):
                continue
            
            # Look for lines that could be names
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(word and word.replace('.', '').replace(',', '').isalpha() and 
                      word[0].isupper() for word in words):
                    # Additional validation - avoid common non-name patterns
                    non_name_patterns = ['resume', 'curriculum', 'vitae', 'cv', 'profile']
                    if not any(pattern.lower() in line.lower() for pattern in non_name_patterns):
                        return line
        
        return ""
        
    except Exception as e:
        print(f"Error in name extraction: {e}")
        return ""

def section_confidence_map(conf_dict: Dict[str, float]) -> Dict[str, int]:
    """Convert confidence to percentage"""
    try:
        return {k: int(round(v * 100)) for k, v in conf_dict.items() if isinstance(v, (int, float))}
    except Exception:
        return {k: 0 for k in conf_dict.keys()}

def parse_file(path: Path, skills_path: Path) -> Dict:
    """Enhanced file parsing with comprehensive error handling"""
    try:
        print(f"Parsing file: {path.name}")
        
        # Load and clean text
        raw = load_text(path)
        if not raw:
            return {"file": path.name, "error": "Could not extract text from file"}
        
        text = clean_text(raw)
        if not text:
            return {"file": path.name, "error": "File appears to be empty"}
        
        print(f"Text length for {path.name}: {len(text)}")
        
        # Split into sections
        sections = split_sections(text)
        print(f"Sections found for {path.name}: {list(sections.keys())}")
        
        # Load skills dictionary
        skills_list = load_skills_dict(skills_path)
        
        # Extract entities using NLP
        nlp_entities = extract_entities_nlp(text)
        
        # Extract name
        name = extract_name_advanced(nlp_entities, text)
        
        # Extract contact information
        contacts = {
            "email": nlp_entities.get("EMAIL", [""])[0] if nlp_entities.get("EMAIL") else "",
            "phone": nlp_entities.get("PHONE", [""])[0] if nlp_entities.get("PHONE") else "",
        }
        
        # Look for additional contact info
        try:
            linkedin_match = re.search(r'linkedin\.com/in/([A-Za-z0-9\-]+)', text, re.IGNORECASE)
            if linkedin_match:
                contacts["linkedin"] = f"linkedin.com/in/{linkedin_match.group(1)}"
            
            github_match = re.search(r'github\.com/([A-Za-z0-9\-]+)', text, re.IGNORECASE)
            if github_match:
                contacts["github"] = f"github.com/{github_match.group(1)}"
        except Exception:
            pass
        
        # Extract skills
        skills, skills_confidence = extract_skills(text, skills_list)
        
        # Calculate confidence scores
        name_conf = 0.9 if name and len(name.split()) >= 2 else (0.4 if name else 0.0)
        contact_conf = 0.8 if any(contacts.values()) else 0.0
        
        # Extract section content safely
        education_content = (sections.get("education", "") or 
                           sections.get("academic background", "") or
                           sections.get("qualifications", ""))
        
        projects_content = (sections.get("projects", "") or 
                          sections.get("key projects", "") or
                          sections.get("project experience", ""))
        
        experience_content = (sections.get("experience", "") or 
                            sections.get("work experience", "") or 
                            sections.get("professional experience", "") or
                            sections.get("employment history", ""))
        
        education_conf = 0.7 if education_content else 0.0
        projects_conf = 0.6 if projects_content else 0.0
        experience_conf = 0.8 if experience_content else 0.0
        
        confidence = section_confidence_map({
            "name": name_conf,
            "contact": contact_conf,
            "skills": skills_confidence,
            "education": education_conf,
            "projects": projects_conf,
            "experience": experience_conf
        })
        
        result = {
            "file": path.name,
            "name": name,
            "contacts": contacts,
            "education": education_content,
            "skills": skills,
            "projects": projects_content,
            "experience": experience_content,
            "confidence": confidence,
            "language": "Kannada" if is_kannada(text) else "English",
            "raw_text": text[:2000]  # Limit raw text for memory
        }
        
        print(f"Successfully parsed {path.name}: name={name}, skills_count={len(skills)}")
        return result
        
    except Exception as e:
        error_msg = f"Error parsing file: {str(e)}"
        print(f"Error parsing {path.name}: {e}")
        return {"file": path.name, "error": error_msg}

def parse_folder(folder: Path, skills_path: Path) -> pd.DataFrame:
    """Parse folder of resumes with better error handling"""
    if not folder.exists():
        print(f"Folder does not exist: {folder}")
        return pd.DataFrame()
    
    records = []
    supported_extensions = {".pdf", ".docx", ".txt"}
    
    try:
        files = list(folder.glob("*"))
        print(f"Found {len(files)} files in {folder}")
        
        processed_count = 0
        for p in files:
            if p.suffix.lower() not in supported_extensions:
                print(f"Skipping unsupported file: {p.name}")
                continue
            
            try:
                print(f"Processing: {p.name}")
                rec = parse_file(p, skills_path)
                records.append(rec)
                processed_count += 1
            except Exception as e:
                print(f"Error parsing {p.name}: {e}")
                records.append({"file": p.name, "error": str(e)})
        
        if not records:
            print("No files were processed successfully")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        print(f"Created DataFrame with {len(df)} records")
        print(f"DataFrame columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"Error parsing folder {folder}: {e}")
        return pd.DataFrame([{"error": f"Folder parsing failed: {str(e)}"}])

# Test function
def test_parser():
    """Test the parser functionality"""
    print("🧪 Testing Enhanced Parser...")
    
    sample_text = """
    John Michael Smith
    Senior Software Engineer
    Email: john.smith@techcorp.com | Phone: +1-555-123-4567
    
    EXPERIENCE
    Senior Python Developer at Google (2020-2024)
    - Developed machine learning models
    - Built REST APIs using Django
    
    SKILLS
    Python, JavaScript, React, AWS, Docker
    """
    
    try:
        # Test entity extraction
        entities = extract_entities_nlp(sample_text)
        print(f"✅ Entities: {entities}")
        
        # Test name extraction
        name = extract_name_advanced(entities, sample_text)
        print(f"✅ Name: {name}")
        
        # Test sections
        sections = split_sections(sample_text)
        print(f"✅ Sections: {list(sections.keys())}")
        
        print("🎉 Parser test completed successfully!")
        
    except Exception as e:
        print(f"❌ Parser test failed: {e}")

if __name__ == "__main__":
    test_parser()