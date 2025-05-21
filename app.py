import streamlit as st
import fitz  # PyMuPDF
import docx
import re

# Extract raw text from PDF or DOCX
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    text = ""
    if ext == 'pdf':
        doc = fitz.open(stream=file.read(), filetype='pdf')
        text = "\n".join([page.get_text() for page in doc])
    elif ext == 'docx':
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Extract major sections using known headers
def extract_sections(text):
    known_headers = [
        'EDUCATION', 'PROJECTS', 'SKILLS', 'CERTIFICATIONS',
        'WORK EXPERIENCE', 'EXPERIENCE', 'INTERNSHIPS', 'LANGUAGES',
        'ACHIEVEMENTS', 'EXTRA CURRICULAR', 'HOBBIES', 'TECHNICAL SKILLS'
    ]
    pattern = "|".join([re.escape(h) for h in known_headers])
    regex = re.compile(rf'(?P<header>{pattern})\s*\n(?P<body>.*?)(?=\n(?:{pattern})\s*\n|\Z)', re.IGNORECASE | re.DOTALL)
    matches = regex.finditer(text)
    sections = {}
    for match in matches:
        header = match.group('header').strip().upper()
        body = match.group('body').strip()
        if header in sections:
            sections[header].append(body)
        else:
            sections[header] = [body]
    return sections

# Split section content into sub-blocks
def extract_blocks(text):
    return re.split(r'\n\s*\n', text.strip())

# Search blocks for a subheading match
def find_block_with_subheading(blocks, keyword):
    keyword = keyword.lower()
    for block in blocks:
        if keyword in block.lower():
            return block.strip()
    return None

# Streamlit App
st.set_page_config(page_title="Resume Section/Subheading Search", layout="wide")
st.title("📄 Resume viewer: Search by Main Heading or Subheading")

uploaded_files = st.file_uploader("Upload resumes (.pdf or .docx)", type=["pdf", "docx"], accept_multiple_files=True)

search_key = st.text_input("🔍 Enter a keyword (e.g., PROJECTS,Skills,education):")

if uploaded_files and search_key:
    st.subheader(f"📌 Results for: `{search_key}`")

    for file in uploaded_files:
        file.seek(0)
        text = extract_text(file)
        sections = extract_sections(text)

        found = False
        st.markdown(f"## 📄 {file.name}")

        # First: check if search_key matches a main section header
        matched_section = None
        for section_name in sections:
            if search_key.lower() == section_name.lower():
                matched_section = section_name
                break

        if matched_section:
            st.markdown(f"### 🔹 Section: {matched_section}")
            for block in sections[matched_section]:
                st.text(block)
            found = True
        else:
            # Search for subheading inside section blocks
            for section_name, contents in sections.items():
                for content in contents:
                    blocks = extract_blocks(content)
                    match = find_block_with_subheading(blocks, search_key)
                    if match:
                        st.markdown(f"### 🔹 Found in section: {section_name}")
                        st.text(match)
                        found = True

        if not found:
            st.info(f"❌ No match found for `{search_key}` in this resume.")

        st.markdown("---")
