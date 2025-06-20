import os
import json
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from PyPDF2 import PdfReader
from docx import Document
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import tempfile
import shutil
from pymongo import MongoClient
from datetime import datetime
import uuid

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB setup
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable is not set")
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['resume_ranker']
candidates_collection = db['candidate_results']

# FastAPI setup
app = FastAPI()

# Skill taxonomy
SKILL_TAXONOMY = [
    {"name": "Python", "description": "Programming language for data science, web development, and automation."},
    {"name": "JavaScript", "description": "Frontend and backend web development language."},
    {"name": "Java", "description": "Enterprise programming language for backend development."},
    {"name": "AWS", "description": "Cloud computing platform for infrastructure and services."},
    {"name": "React", "description": "JavaScript library for building user interfaces."},
    {"name": "Node.js", "description": "JavaScript runtime for backend development."},
    {"name": "SQL", "description": "Database query language for data management."},
    {"name": "Docker", "description": "Containerization platform for application deployment."},
    {"name": "Machine Learning", "description": "AI/ML algorithms and frameworks."},
    {"name": "Data Analysis", "description": "Statistical analysis and data interpretation skills."}
]

# Pydantic model for response
class CandidateResult(BaseModel):
    name: str
    score: float
    matched_skills: List[str]
    gaps: Dict[str, List[str]]
    resume_data: Dict

# Utility functions
def extract_text_from_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        return text
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {str(e)}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text])
        return text
    except Exception as e:
        logger.error(f"Error reading DOCX {file_path}: {str(e)}")
        return ""

# Enhanced RAG Pipeline with FAISS
class EnhancedRAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = [f"Skill: {skill['name']}, Description: {skill['description']}" for skill in SKILL_TAXONOMY]
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
    
    def query(self, resume_text: str) -> str:
        try:
            chunks = self.text_splitter.split_text(resume_text)
            relevant_docs = []
            for chunk in chunks:
                docs = self.vectorstore.similarity_search(chunk, k=3)
                relevant_docs.extend([doc.page_content for doc in docs])
            return "\n".join(set(relevant_docs))
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return ""

# Enhanced Agents
class EnhancedResumeParserAgent:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.1,
            timeout=60
        )
        self.rag = EnhancedRAGPipeline()
    
    def parse_resume(self, file_path: str) -> Dict:
        try:
            if file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            if not text.strip():
                logger.warning(f"Empty text extracted from {file_path}")
                return {
                    "name": os.path.basename(file_path).split('.')[0],
                    "contact_info": {},
                    "skills": [],
                    "experience": [],
                    "education": [],
                    "certifications": [],
                    "projects": [],
                    "languages": [],
                    "achievements": []
                }
            
            context = self.rag.query(text)
            prompt = f"""
            Extract comprehensive structured information from resume text using context for skills.
            Extract at least 80% of the available data. Return ALL possible fields in JSON format.
            
            Resume: {text[:3000]}...
            Context: {context}
            
            Return detailed JSON with these fields (fill all possible fields):
            {{
                "name": "full name",
                "contact_info": {{
                    "email": "",
                    "phone": "",
                    "linkedin": "",
                    "github": "",
                    "location": ""
                }},
                "summary": "professional summary",
                "skills": ["detailed technical skills"],
                "experience": [{{
                    "role": "job title",
                    "company": "company name",
                    "duration": "employment period",
                    "description": "detailed responsibilities and achievements",
                    "technologies": ["used technologies"]
                }}],
                "education": [{{
                    "degree": "degree name",
                    "institution": "school/university",
                    "year": "graduation year",
                    "gpa": "GPA if available"
                }}],
                "certifications": [{{
                    "name": "certification name",
                    "issuer": "issuing organization",
                    "year": "year obtained"
                }}],
                "projects": [{{
                    "name": "project name",
                    "description": "project details",
                    "technologies": ["used technologies"],
                    "outcome": "project outcomes"
                }}],
                "languages": ["spoken languages"],
                "achievements": ["notable achievements"],
                "publications": ["publications if any"]
            }}
            """
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].strip()
            
            try:
                result = json.loads(content)
                default_result = {
                    "name": os.path.basename(file_path).split('.')[0],
                    "contact_info": {},
                    "skills": [],
                    "experience": [],
                    "education": [],
                    "certifications": [],
                    "projects": [],
                    "languages": [],
                    "achievements": []
                }
                for key in default_result:
                    if key in result:
                        if isinstance(default_result[key], dict):
                            default_result[key].update(result[key])
                        else:
                            default_result[key] = result[key]
                return default_result
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}, Content: {content}")
                return {
                    "name": os.path.basename(file_path).split('.')[0],
                    "contact_info": {},
                    "skills": [],
                    "experience": [],
                    "education": [],
                    "certifications": [],
                    "projects": [],
                    "languages": [],
                    "achievements": []
                }
        except Exception as e:
            logger.error(f"Error parsing resume {file_path}: {str(e)}")
            return {
                "name": os.path.basename(file_path).split('.')[0],
                "contact_info": {},
                "skills": [],
                "experience": [],
                "education": [],
                "certifications": [],
                "projects": [],
                "languages": [],
                "achievements": []
            }

class EnhancedJobMatchingAgent:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.1,
            timeout=60
        )
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {str(e)}")
            self.embedder = None
    
    def match_resume_to_jd(self, resume_json: Dict, jd_text: str) -> Dict:
        try:
            similarity_score = 0.0
            if self.embedder:
                try:
                    jd_embedding = self.embedder.encode(jd_text, show_progress_bar=False)
                    resume_skills = " ".join(resume_json.get("skills", []))
                    if resume_skills:
                        resume_embedding = self.embedder.encode(resume_skills, show_progress_bar=False)
                        similarity_score = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
                except Exception as e:
                    logger.error(f"Error calculating similarity: {str(e)}")
            
            prompt = f"""
            Compare resume and job description in detail:
            Resume Skills: {resume_json.get('skills', [])}
            Resume Experience: {resume_json.get('experience', [])}
            Job Description: {jd_text[:1000]}...
            
            Return comprehensive JSON analysis:
            {{
                "matched_skills": ["matching skills with proficiency if available"],
                "matched_experience": ["relevant experience matches"],
                "match_percentage": 75.5,
                "strengths": ["candidate's strongest qualifications"],
                "weaknesses": ["areas lacking compared to JD"]
            }}
            """
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].strip()
            
            try:
                result = json.loads(content)
                final_percentage = max(
                    result.get("match_percentage", 0.0),
                    similarity_score * 100
                )
                result["match_percentage"] = final_percentage
                return result
            except json.JSONDecodeError:
                return {
                    "matched_skills": [],
                    "matched_experience": [],
                    "match_percentage": similarity_score * 100,
                    "strengths": [],
                    "weaknesses": []
                }
        except Exception as e:
            logger.error(f"Error in job matching: {str(e)}")
            return {
                "matched_skills": [],
                "matched_experience": [],
                "match_percentage": 0.0,
                "strengths": [],
                "weaknesses": []
            }

class EnhancedScoringAgent:
    def __init__(self):
        # Updated weights as per request
        self.weights = {
            "skills": 0.40,       # 40%
            "experience": 0.35,   # 35%
            "education": 0.15,    # 15%
            "projects": 0.10      # 10% (includes recency)
        }
    
    def parse_duration(self, duration: str) -> float:
        if not duration:
            return 0.0
            
        duration = duration.lower()
        
        if 'present' in duration or 'current' in duration:
            return 1.0
            
        years = 0.0
        if 'year' in duration or 'yr' in duration:
            parts = duration.split()
            for i, part in enumerate(parts):
                if part.isdigit():
                    years = max(years, float(part))
                elif part in ['year', 'years', 'yr', 'yrs'] and i > 0 and parts[i-1].isdigit():
                    years = max(years, float(parts[i-1]))
        
        if years == 0 and ('month' in duration or 'mo' in duration):
            months = 0.0
            parts = duration.split()
            for i, part in enumerate(parts):
                if part.isdigit():
                    months = max(months, float(part))
            years = months / 12
            
        return max(years, 0.5)

    def calculate_score(self, match_result: Dict, resume_json: Dict) -> Dict:
        try:
            # 1. Skill Score (40% weight)
            skill_match_percentage = match_result.get("match_percentage", 0)
            skill_score = skill_match_percentage * self.weights["skills"]
            
            # 2. Experience Score (35% weight)
            exp_score = 0.0
            for exp in resume_json.get("experience", []):
                years = self.parse_duration(exp.get("duration", ""))
                position_score = min(years * 8, 30)
                
                techs = exp.get("technologies", [])
                relevant_techs = {'python', 'fastapi', 'flask', 'django', 'postgresql', 
                                 'mysql', 'sql', 'docker', 'aws', 'git'}
                tech_bonus = sum(1 for tech in techs if tech.lower() in relevant_techs) * 2
                
                # Recency bonus: Add points if experience is recent (within last 3 years)
                if 'present' in exp.get("duration", "").lower() or '202' in exp.get("duration", ""):
                    recency_bonus = 5
                else:
                    recency_bonus = 0
                
                exp_score += position_score + tech_bonus + recency_bonus
            
            experience_score = min(exp_score, 100) * self.weights["experience"]
            
            # 3. Education/Certifications Score (15% weight)
            education_score = 0.0
            for edu in resume_json.get("education", []):
                degree = edu.get("degree", "").lower()
                if 'computer' in degree or 'engineering' in degree or 'cs' in degree:
                    if 'phd' in degree:
                        education_score += 25
                    elif 'master' in degree:
                        education_score += 20
                    elif 'bachelor' in degree or 'b.e' in degree or 'b.tech' in degree:
                        education_score += 15
                    elif 'diploma' in degree:
                        education_score += 10
            
            # Add certification points
            certifications = resume_json.get("certifications", [])
            certification_score = min(len(certifications) * 5, 20)
            education_score = min(education_score + certification_score, 100) * self.weights["education"]
            
            # 4. Project/Recency Score (10% weight)
            projects = resume_json.get("projects", [])
            project_score = 0.0
            for project in projects:
                techs = project.get("technologies", [])
                relevant_techs = {'python', 'fastapi', 'flask', 'django', 'postgresql', 
                                 'mysql', 'sql', 'docker', 'aws', 'git'}
                relevance = sum(1 for tech in techs if tech.lower() in relevant_techs)
                project_score += min(relevance * 10, 20)
            
            project_score = min(project_score, 100) * self.weights["projects"]
            
            # Calculate total score (0-100)
            total_score = skill_score + experience_score + education_score + project_score
            
            return {
                "total_score": round(total_score, 1),
                "breakdown": {
                    "skills": round(skill_score, 1),
                    "experience": round(experience_score, 1),
                    "education": round(education_score, 1),
                    "projects": round(project_score, 1)
                }
            }
        except Exception as e:
            logger.error(f"Error calculating score: {str(e)}")
            return {
                "total_score": 0.0,
                "breakdown": {
                    "skills": 0.0,
                    "experience": 0.0,
                    "education": 0.0,
                    "projects": 0.0
                }
            }

class EnhancedGapAnalysisAgent:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.1,
            timeout=60
        )
    
    def analyze_gaps(self, resume_json: Dict, jd_text: str, score_result: Dict) -> Dict:
        try:
            prompt = f"""
            Analyze gaps between resume and job description in detail:
            Resume Skills: {resume_json.get('skills', [])}
            Resume Experience: {resume_json.get('experience', [])}
            Job Description: {jd_text[:1000]}...
            Current Match Score: {score_result.get('total_score', 0)}
            
            Return comprehensive JSON gap analysis:
            {{
                "critical_gaps": ["critical missing skills/qualifications"],
                "optional_gaps": ["nice-to-have missing skills"],
                "experience_gaps": ["missing years or types of experience"],
                "education_gaps": ["missing education requirements"],
                "recommendations": ["how candidate could improve"]
            }}
            """
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].strip()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "critical_gaps": [],
                    "optional_gaps": [],
                    "experience_gaps": [],
                    "education_gaps": [],
                    "recommendations": []
                }
        except Exception as e:
            logger.error(f"Error in gap analysis: {str(e)}")
            return {
                "critical_gaps": [],
                "optional_gaps": [],
                "experience_gaps": [],
                "education_gaps": [],
                "recommendations": []
            }

class BlacklistAgent:
    def __init__(self):
        # Configurable blacklist rules
        self.rules = {
            "degree_requirements": {
                "enabled": True,
                "required_degrees": ["bachelor", "b.sc", "b.tech", "bs", "be", "b.e"],
                "required_fields": ["computer science", "engineering", "information technology"]
            },
            "keyword_blacklist": {
                "enabled": True,
                "blocked_keywords": ["faked", "fabricated", "not real", "test resume"]
            },
            "location_restrictions": {
                "enabled": False,  # Disabled by default
                "blocked_locations": []
            },
            "experience_threshold": {
                "enabled": True,
                "min_years": 1
            },
            "duplicate_detection": {
                "enabled": True,
                "similarity_threshold": 0.95
            }
        }
        
        # For duplicate detection
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading sentence transformer for BlacklistAgent: {str(e)}")
            self.embedder = None
    
    def check_degree_requirements(self, education: List[Dict]) -> bool:
        """Check if candidate meets minimum degree requirements"""
        if not self.rules["degree_requirements"]["enabled"]:
            return True
            
        required_degrees = self.rules["degree_requirements"]["required_degrees"]
        required_fields = self.rules["degree_requirements"]["required_fields"]
        
        for edu in education:
            degree = edu.get("degree", "").lower()
            field = edu.get("field", "").lower()
            
            # Check if any required degree is present
            if any(req_degree in degree for req_degree in required_degrees):
                # Check if field matches (if specified)
                if not required_fields or any(req_field in field or req_field in degree for req_field in required_fields):
                    return True
        return False
    
    def check_blocked_keywords(self, text: str) -> bool:
        """Check for blocked keywords in resume text"""
        if not self.rules["keyword_blacklist"]["enabled"]:
            return True
            
        blocked_keywords = self.rules["keyword_blacklist"]["blocked_keywords"]
        text_lower = text.lower()
        return not any(keyword in text_lower for keyword in blocked_keywords)
    
    def check_location(self, location: str) -> bool:
        """Check if candidate is in blocked location"""
        if not self.rules["location_restrictions"]["enabled"]:
            return True
            
        blocked_locations = [loc.lower() for loc in self.rules["location_restrictions"]["blocked_locations"]]
        return location.lower() not in blocked_locations
    
    def check_experience(self, experience: List[Dict]) -> bool:
        """Check if candidate meets minimum experience"""
        if not self.rules["experience_threshold"]["enabled"]:
            return True
            
        min_years = self.rules["experience_threshold"]["min_years"]
        total_years = 0
        
        for exp in experience:
            duration = exp.get("duration", "")
            if 'present' in duration.lower() or 'current' in duration.lower():
                total_years += min_years  # Assume they meet requirement if currently employed
                break
                
            years = 0
            if 'year' in duration or 'yr' in duration:
                parts = duration.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        years = max(years, float(part))
            total_years += years
            
        return total_years >= min_years
    
    def check_duplicate(self, text: str, other_texts: Dict[str, str]) -> Optional[str]:
        """Check for duplicate content using embeddings"""
        if not self.rules["duplicate_detection"]["enabled"] or not self.embedder:
            return None
            
        threshold = self.rules["duplicate_detection"]["similarity_threshold"]
        current_embedding = self.embedder.encode(text, show_progress_bar=False)
        
        for name, other_text in other_texts.items():
            other_embedding = self.embedder.encode(other_text, show_progress_bar=False)
            similarity = cosine_similarity([current_embedding], [other_embedding])[0][0]
            if similarity >= threshold:
                return name
        return None
    
    def filter_candidates(self, candidates: List[Dict], jd_text: str) -> Dict:
        """
        Filter candidates based on blacklist rules
        Returns: {
            "filtered_candidates": [approved candidates],
            "rejected_candidates": [
                {"name": "John Doe", "reason": "Missing degree requirement"},
                ...
            ]
        }
        """
        approved = []
        rejected = []
        seen_texts = {}
        
        for candidate in candidates:
            resume_data = candidate.get("resume_data", {})
            name = resume_data.get("name", "Unknown")
            reject_reasons = []
            
            # 1. Check degree requirements
            if self.rules["degree_requirements"]["enabled"]:
                education = resume_data.get("education", [])
                if not self.check_degree_requirements(education):
                    reject_reasons.append("Missing required degree")
            
            # 2. Check blocked keywords
            if self.rules["keyword_blacklist"]["enabled"]:
                resume_text = json.dumps(resume_data)
                if not self.check_blocked_keywords(resume_text):
                    reject_reasons.append("Contains blocked keywords")
            
            # 3. Check location
            if self.rules["location_restrictions"]["enabled"]:
                location = resume_data.get("contact_info", {}).get("location", "")
                if location and not self.check_location(location):
                    reject_reasons.append(f"Location restricted: {location}")
            
            # 4. Check experience
            if self.rules["experience_threshold"]["enabled"]:
                experience = resume_data.get("experience", [])
                if not self.check_experience(experience):
                    reject_reasons.append(f"Insufficient experience (min {self.rules['experience_threshold']['min_years']} years required)")
            
            # 5. Check for duplicates
            if self.rules["duplicate_detection"]["enabled"]:
                resume_text = json.dumps(resume_data)
                duplicate_of = self.check_duplicate(resume_text, seen_texts)
                if duplicate_of:
                    reject_reasons.append(f"Duplicate of {duplicate_of}")
                else:
                    seen_texts[name] = resume_text
            
            if reject_reasons:
                rejected.append({
                    "name": name,
                    "reasons": reject_reasons,
                    "original_data": candidate
                })
            else:
                approved.append(candidate)
        
        return {
            "filtered_candidates": approved,
            "rejected_candidates": rejected,
            "filter_stats": {
                "total_candidates": len(candidates),
                "approved": len(approved),
                "rejected": len(rejected),
                "approval_rate": len(approved) / len(candidates) * 100 if candidates else 0
            }
        }

def process_resumes(jd_text: str, resume_files: List[str]) -> List[Dict]:
    try:
        parser = EnhancedResumeParserAgent()
        matcher = EnhancedJobMatchingAgent()
        scorer = EnhancedScoringAgent()
        gap_analyzer = EnhancedGapAnalysisAgent()
        blacklist_agent = BlacklistAgent()
        
        results = []
        
        for file_path in resume_files:
            try:
                resume_json = parser.parse_resume(file_path)
                match_result = matcher.match_resume_to_jd(resume_json, jd_text)
                score_result = scorer.calculate_score(match_result, resume_json)
                gaps = gap_analyzer.analyze_gaps(resume_json, jd_text, score_result)
                
                candidate_result = {
                    "name": resume_json.get("name", os.path.basename(file_path).split('.')[0]),
                    "score": score_result["total_score"],
                    "matched_skills": match_result.get("matched_skills", []),
                    "gaps": gaps,
                    "resume_data": resume_json,
                    "score_breakdown": score_result["breakdown"]
                }
                
                results.append(candidate_result)
                
            except Exception as e:
                logger.error(f"Error processing resume {file_path}: {str(e)}")
                continue
        
        # Apply blacklist filtering
        filtered_results = blacklist_agent.filter_candidates(results, jd_text)
        
        # Sort only approved candidates
        filtered_results["filtered_candidates"].sort(key=lambda x: x["score"], reverse=True)
        
        # Add blacklist info to results
        return {
            "approved_candidates": filtered_results["filtered_candidates"],
            "rejected_candidates": filtered_results["rejected_candidates"],
            "filter_stats": filtered_results["filter_stats"],
            "original_count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in process_resumes: {str(e)}")
        return {
            "approved_candidates": [],
            "rejected_candidates": [],
            "filter_stats": {
                "total_candidates": 0,
                "approved": 0,
                "rejected": 0,
                "approval_rate": 0
            },
            "original_count": 0
        }

@app.post("/api/rank-resumes")
async def api_rank_resumes(jd: str = Form(...), resumes: List[UploadFile] = File(...)):
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key is not configured. Please set OPENAI_API_KEY environment variable."
            )
        
        resume_files = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            for resume in resumes:
                if resume.filename and (resume.filename.endswith('.pdf') or resume.filename.endswith('.docx')):
                    file_path = os.path.join(temp_dir, resume.filename)
                    with open(file_path, "wb") as f:
                        logger.info(f"Saving resume {resume.filename} to {file_path}")
                        content = await resume.read()
                        f.write(content)
                    resume_files.append(file_path)
            
            if not resume_files:
                raise HTTPException(
                    status_code=400,
                    detail="No valid resume files uploaded. Please upload PDF or DOCX files."
                )
            
            results = process_resumes(jd, resume_files)
            
            # Store results in MongoDB (only approved candidates)
            if results["approved_candidates"]:
                batch_id = str(uuid.uuid4())
                for result in results["approved_candidates"]:
                    db_record = {
                        "batch_id": batch_id,
                        "job_description": jd,
                        "candidate": result,
                        "timestamp": datetime.utcnow(),
                        "status": "approved"
                    }
                    candidates_collection.insert_one(db_record)
                    logger.info(f"Stored approved candidate {result['name']} in MongoDB")
                
                # Also store rejected candidates with reasons
                for rejected in results["rejected_candidates"]:
                    db_record = {
                        "batch_id": batch_id,
                        "job_description": jd,
                        "candidate": rejected["original_data"],
                        "rejection_reasons": rejected["reasons"],
                        "timestamp": datetime.utcnow(),
                        "status": "rejected"
                    }
                    candidates_collection.insert_one(db_record)
                    logger.info(f"Stored rejected candidate {rejected['name']} in MongoDB")
            
            return {
                "status": "success",
                "results": {
                    "approved": results["approved_candidates"],
                    "rejected": results["rejected_candidates"],
                    "stats": results["filter_stats"]
                },
                "count": results["filter_stats"]["total_candidates"],
                "batch_id": batch_id if results["approved_candidates"] else None
            }
            
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temp directory {temp_dir}: {str(e)}")
                
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /api/rank-resumes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Resume Ranker API is running"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Resume Ranker API...")
    logger.info("Make sure to set OPENAI_API_KEY and MONGODB_URI environment variables!")
    uvicorn.run(app, host="0.0.0.0", port=8000)
