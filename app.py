import streamlit as st
import fitz  # PyMuPDF
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import base64
import sqlite3
import re

def read_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file):
    if file.name.endswith(".pdf"):
        return read_pdf(file)
    elif file.name.endswith(".docx"):
        return read_docx(file)
    else:
        return ""

def is_allowed_word(word):
    return word.isalpha() or word.lower() in {"year", "years", "yr", "yrs"}

st.title("🤖 Applicant Screening Bot")
st.write("Upload a job description and resumes to see relevance scores and matched keywords.")

job_file = st.file_uploader("📄 Upload Job Description (.txt)", type=["txt"])
resume_files = st.file_uploader("📂 Upload Multiple Resumes (.pdf, .docx)", type=["pdf", "docx"], accept_multiple_files=True)

if job_file and resume_files:
    job_text = job_file.read().decode("utf-8", errors="ignore")
    resumes = []
    resume_names = []

    for file in resume_files:
        resumes.append(extract_text(file))
        resume_names.append(file.name)

    documents = [job_text] + resumes
    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"\b(?:[a-zA-Z]{2,}|[1-9][0-9]*\s*(?:years?|yrs?|yr))\b",
        lowercase=True
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    job_tokens = set(vectorizer.get_feature_names_out())
    resume_tokens_list = [
        set(vectorizer.get_feature_names_out()[tfidf_matrix[i].nonzero()[1]])
        for i in range(1, len(documents))
    ]

    rows = []
    for name, score, tokens in zip(resume_names, scores, resume_tokens_list):
        matched = sorted([
            word for word in job_tokens.intersection(tokens)
            if is_allowed_word(word)
        ])
        rows.append({
            "Resume": name,
            "Match Score": round(score * 100, 2),
            "Matched Words": ", ".join(matched)
        })

    df_results = pd.DataFrame(rows).sort_values(by="Match Score", ascending=False).reset_index(drop=True)

    st.subheader("🎯 Filter Options")
    min_score = st.slider("Minimum Match %", 0, 100, 50)
    top_percent = st.slider("Top % of Candidates to Show", 1, 100, 100)
    search_term = st.text_input("🔍 Search by Resume Name")

    filtered = df_results[df_results["Match Score"] >= min_score]
    if top_percent < 100:
        top_n = int(len(filtered) * top_percent / 100)
        filtered = filtered.head(top_n)
    if search_term:
        filtered = filtered[filtered["Resume"].str.contains(search_term, case=False)]

    st.subheader("📊 Resume Match Rankings")
    st.dataframe(filtered)

    csv_data = filtered.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="filtered_resume_scores.csv">📥 Download Filtered Results</a>', unsafe_allow_html=True)

    conn = sqlite3.connect("screening_results.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_name TEXT,
            score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
    for _, row in df_results.iterrows():
        cursor.execute("INSERT INTO results (resume_name, score) VALUES (?, ?)", (row["Resume"], row["Match Score"]))
    conn.commit()
    conn.close()
