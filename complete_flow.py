#!/usr/bin/env python3
"""
Complete End-to-End Resume Processing Flow
Processes PDF resumes through parsing → standardization → summary generation + DOCX generation
Combines all three outputs: structured JSON, professional summaries, and formatted DOCX files
"""

import os
import sys
import json
import re
import requests
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from io import BytesIO
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add current directory to path for imports
sys.path.append('/home/shtlp_0046/Desktop/summary')

# Import the DOCX utility and retailor
from docx_utils import DocxUtils
from retailor import ResumeRetailorNoJD
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('complete_flow_processing.log')
    ]
)
logger = logging.getLogger(__name__)

class CompleteResumeProcessor:
    """
    Complete resume processor that generates all outputs: JSON, summaries, and DOCX files
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the complete processor with environment-based configuration"""
        
        # Load environment variables from project root .env if present
        try:
            load_dotenv(dotenv_path="/home/shtlp_0046/Desktop/summary/.env")
        except Exception:
            # Safe fallback: ignore if dotenv not available
            pass
        
        # Configuration from environment
        self.LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
        self.GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
        
        # Set up directories
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.resumes_dir = self.base_dir / "resumes"
        self.output_dir = self.base_dir / "complete_output"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="complete_processing_"))
        
        # Initialize components
        self.setup_directories()
        self.initialize_parsers()
        self.check_template()
        self.initialize_retailor()
        
        logger.info("🎯 Complete Resume Processor initialized")
        logger.info(f"📂 Base directory: {self.base_dir}")
        logger.info(f"📄 Resumes directory: {self.resumes_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def setup_directories(self):
        """Create all necessary directories"""
        directories = [
            self.resumes_dir,
            self.output_dir,
            self.output_dir / "parsed",
            self.output_dir / "standardized", 
            self.output_dir / "enhanced",
            self.output_dir / "summaries",
            self.output_dir / "docx_files",
            self.output_dir / "failed"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("📁 Complete flow directories created successfully")
    
    def initialize_parsers(self):
        """Initialize parsing components"""
        try:
            # Try to import and initialize LlamaParse
            from llama_parse import LlamaParse
            self.llama_parser = LlamaParse(
                api_key=self.LLAMA_CLOUD_API_KEY,
                result_type="markdown",
                do_not_unroll_columns=True
            )
            self.has_llama_parse = True
            logger.info("✅ LlamaParse initialized")
        except ImportError:
            logger.warning("⚠️ LlamaParse not available - install llama-parse package")
            self.has_llama_parse = False
        except Exception as e:
            logger.warning(f"⚠️ LlamaParse initialization failed: {e}")
            self.has_llama_parse = False
        
        # Initialize Google AI for summaries
        self.has_google_ai = False
        if self.GOOGLE_AI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.GOOGLE_AI_API_KEY)
                self.genai_model = genai.GenerativeModel("gemini-2.0-flash")
                self.has_google_ai = True
                logger.info("✅ Google AI initialized for summaries")
            except ImportError:
                logger.warning("⚠️ Google AI not available - install google-generativeai package")
            except Exception as e:
                logger.warning(f"⚠️ Google AI initialization failed: {e}")
        else:
            logger.warning("⚠️ GOOGLE_AI_API_KEY not provided - summary generation will be limited")
    
    def check_template(self):
        """Check if DOCX template exists"""
        template_path = Path(DocxUtils.TEMPLATE_PATH)
        if template_path.exists():
            logger.info(f"✅ DOCX template found: {template_path}")
            self.has_template = True
        else:
            logger.error(f"❌ DOCX template not found: {template_path}")
            logger.info("💡 Please ensure Resume_Final_draft.docx is in the correct location")
            self.has_template = False
    
    def initialize_retailor(self):
        """Initialize the resume retailor"""
        try:
            self.retailor = ResumeRetailorNoJD()
            self.has_retailor = True
            logger.info("✅ Resume retailor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize retailor: {e}")
            self.has_retailor = False
    
    # ===== PDF PARSING METHODS =====
    
    def extract_text_from_pdf_ocr(self, pdf_path: Path) -> str:
        """Extract text using OCR as fallback"""
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            images = convert_from_path(str(pdf_path), dpi=300)
            text = ""
            for i, image in enumerate(images):
                try:
                    text += f"\n\n--- Page {i+1} ---\n\n"
                    text += pytesseract.image_to_string(image)
                except Exception as e:
                    logger.warning(f"⚠️ OCR failed on page {i+1}: {e}")
            return text.strip()
        except ImportError:
            logger.warning("⚠️ OCR dependencies not available")
            return ""
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {e}")
            return ""
    
    def extract_links_from_pdf(self, pdf_path: Path) -> List[Dict[str, str]]:
        """Extract hyperlinks from PDF"""
        links = []
        try:
            import fitz  # PyMuPDF
            with fitz.open(str(pdf_path)) as doc:
                for page in doc:
                    for link in page.get_links():
                        if "uri" in link:
                            text = page.get_textbox(link.get("from", ())).strip()
                            links.append({"text": text, "uri": link["uri"]})
        except ImportError:
            logger.warning("⚠️ PyMuPDF not available for link extraction")
        except Exception as e:
            logger.warning(f"⚠️ Link extraction failed: {e}")
        return links
    
    def extract_links_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract URLs from text content"""
        url_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
        urls = re.findall(url_pattern, text)
        return [{"text": url, "uri": url} for url in urls]
    
    def parse_resume_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a single PDF or DOCX resume"""
        logger.info(f"📄 Parsing file: {file_path.name}")
        
        filename = file_path.name
        file_ext = file_path.suffix.lower()
        llama_content = ""
        ocr_content = ""
        llama_links = []
        ocr_links = []
        
        # Try LlamaParse first (supports both PDF and DOCX)
        if self.has_llama_parse:
            try:
                documents = self.llama_parser.load_data(str(file_path))
                llama_content = "\n".join(doc.text for doc in documents).strip()
                
                # Extract links (only for PDFs)
                if file_ext == '.pdf':
                    llama_links = self.extract_links_from_pdf(file_path)
                
                logger.info(f"✅ LlamaParse extraction successful for {filename}")
            except Exception as e:
                logger.warning(f"⚠️ LlamaParse failed for {filename}: {e}")
        
        # Try OCR as fallback (only for PDFs)
        if file_ext == '.pdf':
            ocr_content = self.extract_text_from_pdf_ocr(file_path)
            if ocr_content:
                ocr_links = self.extract_links_from_text(ocr_content)
                logger.info(f"✅ OCR extraction successful for {filename}")
        
        # For DOCX files, if LlamaParse failed, try basic text extraction
        if file_ext == '.docx' and not llama_content.strip():
            try:
                from docx import Document
                doc = Document(str(file_path))
                ocr_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                logger.info(f"✅ Basic DOCX extraction successful for {filename}")
            except ImportError:
                logger.warning("⚠️ python-docx not available for DOCX text extraction")
            except Exception as e:
                logger.warning(f"⚠️ DOCX text extraction failed for {filename}: {e}")
        
        # Merge links
        seen_uris = set()
        merged_links = []
        for link in llama_links + ocr_links:
            uri = link.get("uri")
            if uri and uri not in seen_uris:
                merged_links.append(link)
                seen_uris.add(uri)
        
        # Validate content
        if not llama_content.strip() and not ocr_content.strip():
            raise ValueError(f"❌ Failed to extract content from: {filename}")
        
        result = {
            "file": filename,
            "content": {
                "llama": llama_content,
                "ocr": ocr_content
            },
            "links": merged_links
        }
        
        # Save parsed result
        employee_id = re.sub(r'\.[^.]+$', '', filename)
        parsed_file = self.output_dir / "parsed" / f"{employee_id}.json"
        with open(parsed_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Parsed and saved: {filename}")
        return result
    
    # ===== STANDARDIZATION METHODS =====
    
    def create_standardization_prompt(self, llama_text: str, ocr_text: str, links: List[Dict]) -> str:
        """Create the standardization prompt for Azure OpenAI"""
        links_json = json.dumps(links, indent=2) if links else "[]"
        
        prompt = f"""
You are an intelligent and robust context-aware resume standardizer. Your task is to convert resume content into a clean, structured, normalized/standardized JSON format suitable for both semantic retrieval and relational database storage.

--- PARSED RESUME CONTENT ---
The following text comes from two sources:
1. *LlamaParse* - reliable markdown-style structure
2. *OCR fallback* - may contain missing structure or artifacts

Use both as needed to create the most complete structured resume.

--- LlamaParse Content ---
\"\"\"{llama_text}\"\"\"

--- OCR Fallback Content ---
\"\"\"{ocr_text}\"\"\"

--- EXTRACTED HYPERLINKS ---
{links_json}

--- STANDARDIZED STRUCTURE ---
Convert the resume to a JSON object strictly following this structure:

{{
  "employee_id": str,
  "name": str,
  "email": str,
  "phone": str,
  "location": str,
  "summary": str,
  "education": [
    {{
      "degree": str,
      "institution": str,
      "year": int or str
    }}
  ],
  "experience": [
    {{
      "title": str,
      "company": str,
      "duration": str,
      "location": str (optional),
      "description": str
    }}
  ],
  "skills": [str],
  "projects": [
    {{
      "title": str,
      "description": str,
      "link": str (optional)
    }}
  ],
  "certifications": [
    {{
      "title": str,
      "issuer": str (optional),
      "year": str or int (optional),
      "link": str (only if it can be reliably mapped from extracted hyperlinks)
    }}
  ],
  "languages": [str],
  "social_profiles": [
    {{
      "platform": str,
      "link": str
    }}
  ]
}}

--- GUIDELINES ---
- Use both content sections above to extract all information.
- Do not introduce or remove fields arbitrarily.
- Leave keys with empty arrays if data is missing.
- Ensure the output is only valid JSON. No comments or markdown.
"""
        return prompt
    
    def call_azure_openai(self, prompt: str) -> Dict[str, Any]:
        """Make API call to Azure OpenAI"""
        url = f"{self.AZURE_OPENAI_ENDPOINT}/openai/deployments/{self.AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={self.AZURE_OPENAI_API_VERSION}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.AZURE_OPENAI_API_KEY,
            "User-Agent": "Complete-ResumeProcessor/1.0"
        }
        
        request_body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that formats resumes into structured JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 4096
        }
        
        try:
            response = requests.post(url, headers=headers, json=request_body, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Azure OpenAI API request failed: {str(e)}")
            raise
    
    def standardize_resume(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize parsed resume data using Azure OpenAI"""
        filename = parsed_data["file"]
        employee_id = re.sub(r'\.[^.]+$', '', filename)
        
        logger.info(f"🔄 Standardizing: {filename}")
        
        # Extract content
        content_map = parsed_data.get('content', {})
        llama_text = content_map.get('llama', '')
        ocr_text = content_map.get('ocr', '')
        links = parsed_data.get('links', [])
        
        # Validate content
        if not llama_text.strip() and not ocr_text.strip():
            raise ValueError("Both LlamaParse and OCR content are empty")
        
        # Create prompt
        prompt = self.create_standardization_prompt(llama_text, ocr_text, links)
        
        # Call Azure OpenAI
        response_data = self.call_azure_openai(prompt)
        
        # Parse response
        try:
            raw_llm = response_data['choices'][0]['message']['content'].strip()
            cleaned = re.sub(r'(?s)```(?:json)?', '', raw_llm).strip()
            standardized = json.loads(cleaned)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"❌ Failed to parse Azure OpenAI response: {str(e)}")
            raise
        
        # Add employee_id
        standardized['employee_id'] = employee_id
        
        # Calculate usage and cost
        usage = response_data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
        total_cost = round((total_tokens / 1000.0) * 0.00025, 6)
        
        metadata = {
            "employee_id": employee_id,
            "filename": filename,
            "llm_model": response_data.get('model', self.AZURE_OPENAI_DEPLOYMENT),
            "llm_prompt_tokens": prompt_tokens,
            "llm_completion_tokens": completion_tokens,
            "llm_total_tokens": total_tokens,
            "llm_total_cost": f"${total_cost}"
        }
        
        # Save standardized result
        standardized_file = self.output_dir / "standardized" / f"{employee_id}.json"
        with open(standardized_file, 'w', encoding='utf-8') as f:
            json.dump(standardized, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Standardized: {filename} (Cost: ${total_cost}, Tokens: {total_tokens})")
        
        return {
            'standardized_resume': standardized,
            'metadata': metadata
        }
    
    # ===== ENHANCEMENT METHODS =====
    
    def enhance_resume(self, standardized_resume: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the standardized resume using retailor"""
        if not self.has_retailor:
            logger.warning("⚠️ Retailor not available - returning original resume")
            return standardized_resume
        
        employee_id = standardized_resume.get('employee_id', 'unknown')
        logger.info(f"🔧 Enhancing resume for: {employee_id}")
        
        try:
            enhanced_resume = self.retailor.retailor_resume_no_jd(standardized_resume)
            
            # Save enhanced result
            enhanced_file = self.output_dir / "enhanced" / f"{employee_id}.json"
            with open(enhanced_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_resume, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Enhanced resume saved: {employee_id}")
            return enhanced_resume
            
        except Exception as e:
            logger.error(f"❌ Failed to enhance resume for {employee_id}: {str(e)}")
            return standardized_resume
    
    # ===== SUMMARY GENERATION METHODS =====
    
    def parse_resume_for_summary(self, resume_json: Dict[str, Any]) -> str:
        """Parse resume JSON for comprehensive summary generation"""
        name = resume_json.get('name', 'N/A')
        location = resume_json.get('location', 'N/A')
        current_summary = resume_json.get('summary', '')
        
        # Extract total years of experience directly from resume summary
        current_summary = resume_json.get('summary', '')
        total_experience = 0
        
        if current_summary:
            import re
            # Look for patterns like "21+ years", "21 years", "with 21 years", "21+ years of experience"
            experience_patterns = [
                r'(\d+)\+?\s*years?\s+of\s+experience',  # "21+ years of experience"
                r'with\s+(\d+)\+?\s*years?',             # "with 21+ years"
                r'(\d+)\+?\s*years?\s+experience',       # "21+ years experience"
                r'(\d+)\+?\s*years?\s+in',               # "21+ years in"
                r'(\d+)\+?\s*years?\s+across',           # "21+ years across"
            ]
            
            for pattern in experience_patterns:
                match = re.search(pattern, current_summary.lower())
                if match:
                    total_experience = int(match.group(1))
                    break
        
        # Fallback: if no experience found in summary, use a default
        if total_experience == 0:
            experience = resume_json.get('experience', [])
            total_experience = len(experience) if experience else 0
        
        # Extract detailed education information
        education = resume_json.get('education', [])
        education_text = "Education: "
        if education:
            edu_details = []
            for edu in education:
                degree = edu.get('degree', '')
                specialization = edu.get('field', '') or edu.get('specialization', '')
                institution = edu.get('institution', '')
                year = edu.get('year', '')
                
                edu_str = degree
                if specialization:
                    edu_str += f" in {specialization}"
                if institution:
                    edu_str += f" from {institution}"
                if year:
                    edu_str += f" ({year})"
                edu_details.append(edu_str)
            education_text += "; ".join(edu_details)
        else:
            education_text += "Not specified"
        
        # Extract comprehensive work experience
        experience = resume_json.get('experience', [])
        experience_text = ""
        if experience:
            current_role = experience[0] if experience else {}
            experience_text = f"Current Role: {current_role.get('title', '')} at {current_role.get('company', '')} ({current_role.get('duration', '')})"
            experience_text += f"\nTotal Experience: Approximately {total_experience} years"
            experience_text += f"\nTotal Positions: {len(experience)}"
            
            # Add detailed work history
            work_history = []
            for i, exp in enumerate(experience[:5]):  # Include more positions
                title = exp.get('title', '')
                company = exp.get('company', '')
                duration = exp.get('duration', '')
                work_history.append(f"{i+1}. {title} at {company} ({duration})")
            
            if work_history:
                experience_text += f"\nWork History:\n" + "\n".join(work_history)
            
            # Add detailed responsibilities and achievements
            detailed_responsibilities = []
            for exp in experience[:3]:  # Include more detailed info
                if exp.get('description'):
                    detailed_responsibilities.append(f"• {exp['description']}")
            
            if detailed_responsibilities:
                experience_text += f"\nKey Responsibilities & Achievements:\n" + "\n".join(detailed_responsibilities)
        
        # Extract comprehensive technical skills
        skills = resume_json.get('skills', [])
        skills_text = ""
        if skills:
            # Group skills by categories if possible
            all_skills = skills[:25]  # Include more skills
            skills_text = f"Technical Skills & Technologies: {', '.join(all_skills)}"
        
        # Extract certifications if available
        certifications = resume_json.get('certifications', [])
        cert_text = ""
        if certifications:
            cert_list = []
            for cert in certifications:
                if isinstance(cert, dict):
                    cert_name = cert.get('name', '') or cert.get('certification', '')
                    cert_org = cert.get('organization', '') or cert.get('issuer', '')
                    if cert_org:
                        cert_list.append(f"{cert_name} ({cert_org})")
                    else:
                        cert_list.append(cert_name)
                else:
                    cert_list.append(str(cert))
            cert_text = f"Certifications: {', '.join(cert_list)}"
        
        # Extract projects if available
        projects = resume_json.get('projects', [])
        projects_text = ""
        if projects:
            project_list = []
            for proj in projects[:3]:  # Include top 3 projects
                if isinstance(proj, dict):
                    proj_name = proj.get('name', '') or proj.get('title', '')
                    proj_desc = proj.get('description', '')
                    if proj_desc:
                        project_list.append(f"• {proj_name}: {proj_desc[:150]}...")
                    else:
                        project_list.append(f"• {proj_name}")
                else:
                    project_list.append(f"• {str(proj)}")
            projects_text = f"Key Projects:\n" + "\n".join(project_list)
        
        # Extract languages
        languages = resume_json.get('languages', [])
        languages_text = f"Languages: {', '.join(languages)}" if languages else ""
        
        return f"""
Name: {name}
Location: {location}
Total Years of Experience: {total_experience}
{education_text}
{experience_text}
{skills_text}
{cert_text}
{projects_text}
{languages_text}

Current Summary from Resume:
{current_summary}
        """.strip()
    
    def generate_summary(self, resume_data: str) -> str:
        """Generate professional summary using Google AI"""
        if not self.has_google_ai:
            return "Summary generation not available - Google AI API key required"
        
        prompt = f"""
You are a professional resume writer and career consultant. Based on the following resume information, create a comprehensive and detailed professional summary that would be perfect for a LinkedIn profile or resume header.

The summary should follow this specific structure and format:

**STRUCTURE:**
1. **Opening Statement**: "[Name] is a [Senior/Lead/Principal Title] with [X years] of experience in [domain/industry expertise]..."
2. **Technical Expertise**: "He/She is skilled/proficient in [specific technologies, tools, frameworks]..."
3. **Project Impact & Achievements**: Detailed examples with specific outcomes, metrics, and technical implementations
4. **Leadership & Collaboration**: Team leadership, mentoring, cross-functional work
5. **Education & Certifications**: Complete educational background and relevant certifications

**SPECIFIC REQUIREMENTS:**
1. **Length**: 6-8 sentences, approximately 200-300 words (much more detailed than typical summaries)
2. **Title**: Use appropriate senior-level titles (Senior Solution Architect, Senior Cloud Engineer, Lead DevOps Engineer, etc.)
3. **Experience**: Always include specific number of years of experience
4. **Technical Depth**: Include comprehensive technology stacks with specific tools, frameworks, and platforms
5. **Quantified Achievements**: Include specific project outcomes, efficiency gains, cost savings, team sizes led
6. **Project Examples**: Provide concrete examples of systems built, problems solved, and business impact
7. **Education Details**: End with complete educational qualifications including degree, specialization, and institution
8. **Professional Tone**: Written in third person, results-oriented, and technically detailed
9. **Industry Context**: Mention specific industries, domains, and business contexts
10. **Leadership Elements**: Include team sizes, mentoring, governance, and organizational impact

**EXAMPLES OF DESIRED OUTPUT STYLE:**
- "Abhishek Chandel is a Senior Solution Architect with 21 years of experience driving large-scale enterprise transformations..."
- "He specializes in building AI-first platforms and strategic technology roadmaps..."
- "His work spans generative AI, RAG, vector search, and agentic workflows, with hands-on depth in Python, FastAPI/Django, LangChain..."
- "He has delivered outcomes such as a consulting automation platform that cut manual effort by 60%..."
- "He holds certifications/education in Machine Learning & Deep Learning (IIT Delhi) and a B.Tech from IIT Kanpur..."

Resume Information:
{resume_data}

Generate a comprehensive, detailed professional summary following the exact structure and style shown above:
"""

        try:
            generation_config = {
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1200,  # Increased for longer, more detailed summaries
            }
            
            response = self.genai_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            else:
                return "No content was generated. Please try again with different input."
                
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def generate_resume_summary(self, standardized_resume: Dict[str, Any]) -> str:
        """Generate summary for a standardized resume"""
        employee_id = standardized_resume.get('employee_id', 'unknown')
        logger.info(f"📝 Generating summary for: {employee_id}")
        
        # Parse resume data for summary
        formatted_data = self.parse_resume_for_summary(standardized_resume)
        
        # Generate summary
        summary = self.generate_summary(formatted_data)
        
        # Save summary to file (clean format - only summary text)
        summary_filename = f"{employee_id}_summary.txt"
        summary_path = self.output_dir / "summaries" / summary_filename
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"✅ Summary generated and saved: {summary_filename}")
        return summary
    
    # ===== DOCX GENERATION METHODS =====
    
    def generate_docx_from_resume(self, standardized_resume: Dict[str, Any]) -> Optional[Path]:
        """Generate DOCX file from standardized resume using template"""
        if not self.has_template:
            logger.error("❌ Cannot generate DOCX - template not available")
            return None
        
        employee_id = standardized_resume.get('employee_id', 'unknown')
        logger.info(f"📄 Generating DOCX for: {employee_id}")
        
        try:
            # Generate DOCX using DocxUtils
            docx_stream = DocxUtils.generate_docx(standardized_resume)
            
            # Save DOCX file
            docx_filename = f"{employee_id}_resume.docx"
            docx_path = self.output_dir / "docx_files" / docx_filename
            
            with open(docx_path, 'wb') as f:
                f.write(docx_stream.read())
            
            logger.info(f"✅ DOCX generated and saved: {docx_filename}")
            return docx_path
            
        except Exception as e:
            logger.error(f"❌ Failed to generate DOCX for {employee_id}: {str(e)}")
            return None
    
    # ===== MAIN PROCESSING METHODS =====
    
    def process_single_resume(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single resume through the complete pipeline"""
        result = {
            'success': False,
            'filename': pdf_path.name,
            'employee_id': re.sub(r'\.[^.]+$', '', pdf_path.name),
            'stages_completed': [],
            'outputs_generated': [],
            'error': None
        }
        
        try:
            # Stage 1: Parse file
            logger.info(f"📋 Processing: {pdf_path.name}")
            parsed_data = self.parse_resume_file(pdf_path)
            result['stages_completed'].append('parsing')
            result['outputs_generated'].append('parsed_json')
            
            # Stage 2: Standardize
            standardized_result = self.standardize_resume(parsed_data)
            result['stages_completed'].append('standardization')
            result['outputs_generated'].append('standardized_json')
            
            # Stage 3: Enhance using retailor
            enhanced_resume = self.enhance_resume(standardized_result['standardized_resume'])
            result['stages_completed'].append('enhancement')
            result['outputs_generated'].append('enhanced_json')
            
            # Stage 4: Generate Summary (use standardized data to preserve original experience years)
            if self.has_google_ai:
                summary = self.generate_resume_summary(standardized_result['standardized_resume'])
                result['stages_completed'].append('summary_generation')
                result['outputs_generated'].append('summary_txt')
                result['summary'] = summary
            else:
                logger.warning(f"⚠️ Skipping summary generation for {pdf_path.name} - Google AI not available")
                result['summary'] = "Summary generation skipped - Google AI API key not provided"
            
            # Stage 5: Generate DOCX
            if self.has_template:
                docx_path = self.generate_docx_from_resume(enhanced_resume)
                if docx_path:
                    result['stages_completed'].append('docx_generation')
                    result['outputs_generated'].append('docx_file')
                    result['docx_file'] = str(docx_path.name)
                else:
                    logger.warning(f"⚠️ DOCX generation failed for {pdf_path.name}")
                    result['docx_file'] = "Generation failed"
            else:
                logger.warning(f"⚠️ Skipping DOCX generation for {pdf_path.name} - template not available")
                result['docx_file'] = "Template not available"
            
            result['success'] = True
            logger.info(f"🎯 Successfully processed: {pdf_path.name} (Outputs: {', '.join(result['outputs_generated'])})")
            
        except Exception as e:
            error_msg = f"Error processing {pdf_path.name}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            result['error'] = error_msg
            
            # Save error info
            error_file = self.output_dir / "failed" / f"{result['employee_id']}_error.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        
        return result
    
    def process_all_resumes(self) -> Dict[str, Any]:
        """Process all PDF and DOCX files in the resumes directory"""
        pdf_files = list(self.resumes_dir.glob("*.pdf"))
        docx_files = list(self.resumes_dir.glob("*.docx"))
        all_files = pdf_files + docx_files
        
        if not all_files:
            logger.warning("⚠️ No PDF or DOCX files found in resumes directory")
            return {"total": 0, "successful": 0, "failed": 0, "results": []}
        
        logger.info(f"📄 Found {len(pdf_files)} PDF files and {len(docx_files)} DOCX files")
        logger.info(f"🎯 Starting complete processing of {len(all_files)} files...")
        
        results = []
        successful = 0
        failed = 0
        
        # Track output counts
        output_counts = {
            'parsed_json': 0,
            'standardized_json': 0,
            'enhanced_json': 0,
            'summary_txt': 0,
            'docx_file': 0
        }
        
        for i, file_path in enumerate(all_files, 1):
            logger.info(f"📊 Processing {i}/{len(all_files)}: {file_path.name}")
            
            result = self.process_single_resume(file_path)
            results.append(result)
            
            if result['success']:
                successful += 1
                # Count outputs
                for output_type in result.get('outputs_generated', []):
                    output_counts[output_type] = output_counts.get(output_type, 0) + 1
            else:
                failed += 1
        
        # Create final summary report
        summary_report = {
            "total": len(all_files),
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful / len(all_files) * 100):.1f}%",
            "processing_timestamp": datetime.now().isoformat(),
            "flow_type": "COMPLETE_END_TO_END",
            "stages": ["parsing", "standardization", "enhancement", "summary_generation", "docx_generation"],
            "output_counts": output_counts,
            "capabilities": {
                "llama_parse": self.has_llama_parse,
                "google_ai": self.has_google_ai,
                "docx_template": self.has_template
            },
            "results": results
        }
        
        # Save final report
        report_path = self.output_dir / "complete_processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        # Log final summary
        logger.info("=" * 80)
        logger.info("🎯 COMPLETE END-TO-END PROCESSING FINISHED")
        logger.info("=" * 80)
        logger.info(f"📊 Total files processed: {summary_report['total']}")
        logger.info(f"✅ Successful: {summary_report['successful']}")
        logger.info(f"❌ Failed: {summary_report['failed']}")
        logger.info(f"📈 Success rate: {summary_report['success_rate']}")
        logger.info("")
        logger.info("📁 Output files generated:")
        logger.info(f"   📄 Parsed JSONs: {output_counts['parsed_json']}")
        logger.info(f"   🎯 Standardized JSONs: {output_counts['standardized_json']}")
        logger.info(f"   🔧 Enhanced JSONs: {output_counts['enhanced_json']}")
        logger.info(f"   📝 Summary TXT files: {output_counts['summary_txt']}")
        logger.info(f"   📄 DOCX files: {output_counts['docx_file']}")
        logger.info(f"📁 Results saved to: {self.output_dir}")
        
        if failed > 0:
            failed_files = [r['filename'] for r in results if not r['success']]
            logger.warning(f"❌ Failed files: {', '.join(failed_files)}")
        
        logger.info("=" * 80)
        
        return summary_report
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info(f"🧹 Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup temp directory: {e}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete End-to-End Resume Processing Pipeline')
    parser.add_argument('--base-dir', default='/home/shtlp_0046/Desktop/summary',
                       help='Base directory containing resumes/ folder')
    parser.add_argument('--google-ai-api-key', 
                       help='Google AI API key for summary generation (optional)')
    
    args = parser.parse_args()
    
    # Set Google AI API key if provided
    if args.google_ai_api_key:
        os.environ['GOOGLE_AI_API_KEY'] = args.google_ai_api_key
    
    try:
        logger.info("🎯 Starting Complete End-to-End Resume Processing Pipeline")
        
        # Initialize processor
        processor = CompleteResumeProcessor(base_dir=args.base_dir)
        
        # Process all resumes
        report = processor.process_all_resumes()
        
        # Cleanup
        processor.cleanup()
        
        # Exit with appropriate code
        if report['failed'] > 0:
            logger.warning("⚠️ Some files failed to process. Check the failed directory and logs.")
            return 1
        else:
            logger.info("🎯 All files processed successfully with complete outputs!")
            return 0
        
    except Exception as e:
        logger.error(f"💥 Pipeline execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
