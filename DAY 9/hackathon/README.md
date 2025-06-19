Culture Fit Analyzer
A Streamlit-based AI tool that evaluates candidate alignment with company culture using Retrieval-Augmented Generation (RAG) and a multi-agent AI system. It analyzes URLs (LinkedIn, company websites) or manual inputs (PDFs, text) to generate compatibility scores, insights, and DOCX reports, enhancing hiring quality by focusing on non-technical fit.
🚀 Quick Start
Option 1: Quick Analysis

Go to the "⚡ Quick Analysis" tab.
Enter URLs:
Candidate:
LinkedIn: https://www.linkedin.com/in/afrin-s-snsihub-412139370/
GitHub: https://github.com/torvalds (Optional)
Portfolio: https://brendaneich.com/ (Optional)
Other: https://news.ycombinator.com/user?id=torvalds (Optional)


Company:
About: https://about.gitlab.com/
Careers: https://about.gitlab.com/jobs/
Glassdoor: https://www.glassdoor.com/Overview/Working-at-GitLab-EI_IE1296544.11,17.htm
Other: https://handbook.gitlab.com/handbook/values/




Click "🚀 Quick Analysis".
View results in the "📊 Results" tab and download the DOCX report.

Note: LinkedIn scraping uses .env credentials. If it fails (e.g., MFA), use Option 2.
Option 2: Manual Analysis

Go to the "📄 Manual Input" tab.
Upload PDFs or enter text for:
Candidate: Resume, LinkedIn bio, GitHub README, personal statement.
Company: HR handbook, About page, reviews, job description.


Click "🔍 Analyze Culture Fit".
View results and download the DOCX report.

🎯 Features

RAG Pipeline: Chunks data, generates transformer-based embeddings, stores vectors in FAISS, retrieves relevant chunks, and augments LLM prompts.
Four AI Agents: Analyze candidate behavior, extract company culture, map compatibility, and score fit with actionable outputs.
Data Ingestion: Scrapes LinkedIn (Selenium), websites (requests), and PDFs (PyPDF2); supports manual input.
Streamlit UI: Three tabs (Quick Analysis, Manual Input, Results) with bar charts (matplotlib) and DOCX reports (python-docx).
Outputs: JSON with scores, strengths, challenges, recommendations, interview questions, and onboarding plans.
Robustness: Handles MFA, scraping failures, and LLM errors with fallbacks and logging.

🛠️ How It Was Built

RAG:
Chunking: RecursiveCharacterTextSplitter (500-char chunks, 50-char overlap).
Embeddings: sentence-transformers (all-MiniLM-L6-v2, transformer-based).
Vector Store: faiss (IndexFlatL2) for similarity search.
Retrieval: Top-10 chunks for context-aware analysis.
Augmentation: Gemini (gemini-1.5-flash) generates JSON insights.


Agents:
Candidate Behavior Analyzer: Extracts traits and skills via RAG.
Company Culture Retriever: Extracts values and norms via RAG.
Compatibility Mapping: Compares profiles for alignment scores.
Culture Fit Scoring: Computes weighted scores and recommendations.


Tech Stack: streamlit, langchain, faiss-cpu, google-generativeai, selenium, PyPDF2, requests, beautifulsoup4, python-docx, pandas, numpy, matplotlib.
Security: Credentials in .env via python-dotenv.

🔄 Workflow

Ingest Data: Scrape URLs (LinkedIn via Selenium, others via requests) or process PDFs/text.
Agent 1: Chunk candidate data, retrieve top-10 chunks, generate behavioral profile.
Agent 2: Chunk company data, retrieve top-10 chunks, extract culture profile.
Agent 3: Compare outputs, compute compatibility scores.
Agent 4: Generate weighted fit score, recommendations, and onboarding plan.
Output: Display results in Streamlit with bar charts; export DOCX report.

The sequential pipeline uses JSON for agent communication, avoiding frameworks like LangGraph for simplicity.
✅ Problem-Solving
Goal: Enhance hiring quality by evaluating cultural alignment using RAG and agentic AI.

RAG: Extracts nuanced insights (e.g., personality, values) from unstructured data, using transformers and FAISS for precision.
Agents: Modular system analyzes behavior, culture, compatibility, and fit, producing HR-ready outputs (scores, questions, onboarding).
Challenges Addressed:
LinkedIn scraping failures (MFA, private profiles) via manual input.
Diverse data handled (URLs, PDFs, text).
Robust error handling for LLM, JSON, and scraping issues.


Impact: Actionable insights improve hiring beyond technical skills, aligning with the hackathon goal.

📦 Setup

Clone: git clone <repository-url>
Install: pip install -r requirements.txt
Install Chrome WebDriver (add to PATH).
Create .env:GOOGLE_API_KEY=your_key
LINKEDIN_EMAIL=afrin.s.ihub@snsgroups.com
LINKEDIN_PASSWORD=Afrin@password123


Run: streamlit run main.py

📄 Requirements
streamlit
PyPDF2
python-dotenv
requests
beautifulsoup4
sentence-transformers
faiss-cpu
google-generativeai
python-docx
pandas
numpy
matplotlib
langchain
selenium

⚠️ Notes

LinkedIn: MFA or private profiles may require manual input. Ensure compliance with terms.

