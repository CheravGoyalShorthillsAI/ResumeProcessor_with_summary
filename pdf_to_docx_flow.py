#!/usr/bin/env python3
"""
PDF to DOCX Flow
Processes PDF resumes through parsing → standardization → DOCX generation using template
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
        logging.FileHandler('pdf_to_docx_processing.log')
    ]
)
logger = logging.getLogger(__name__)

class PDFToDOCXProcessor:
    """
    Processes PDF resumes to generate formatted DOCX files using templates
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the PDF to DOCX processor with environment-based configuration"""
        
        # Load environment variables from project root .env if present
        try:
            load_dotenv(dotenv_path="/home/shtlp_0046/Desktop/summary/.env")
        except Exception:
            pass
        
        # Configuration from environment
        self.LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
        
        # Set up directories
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.resumes_dir = self.base_dir / "resumes"
        self.output_dir = self.base_dir / "docx_output"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="pdf_docx_processing_"))
        
        # Initialize components
        self.setup_directories()
        self.initialize_parsers()
        self.check_template()
        self.initialize_retailor()
        
        logger.info("📄 PDF to DOCX Processor initialized")
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
            self.output_dir / "docx_files",
            self.output_dir / "failed"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("📁 DOCX flow directories created successfully")
    
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
            "User-Agent": "PDFToDOCX-Processor/1.0"
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
    
    # ===== DOCX GENERATION METHODS =====
    
    def generate_docx_from_resume(self, enhanced_resume: Dict[str, Any]) -> Optional[Path]:
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
        """Process a single resume through the PDF to DOCX pipeline"""
        result = {
            'success': False,
            'filename': pdf_path.name,
            'employee_id': re.sub(r'\.[^.]+$', '', pdf_path.name),
            'stages_completed': [],
            'error': None
        }
        
        try:
            # Stage 1: Parse file
            logger.info(f"📋 Processing: {pdf_path.name}")
            parsed_data = self.parse_resume_file(pdf_path)
            result['stages_completed'].append('parsing')
            
            # Stage 2: Standardize
            standardized_result = self.standardize_resume(parsed_data)
            result['stages_completed'].append('standardization')
            
            # Stage 3: Enhance using retailor
            enhanced_resume = self.enhance_resume(standardized_result['standardized_resume'])
            result['stages_completed'].append('enhancement')
            
            # Stage 4: Generate DOCX
            if self.has_template:
                docx_path = self.generate_docx_from_resume(enhanced_resume)
                if docx_path:
                    result['stages_completed'].append('docx_generation')
                    result['docx_file'] = str(docx_path.name)
                else:
                    logger.warning(f"⚠️ DOCX generation failed for {pdf_path.name}")
                    result['docx_file'] = "Generation failed"
            else:
                logger.warning(f"⚠️ Skipping DOCX generation for {pdf_path.name} - template not available")
                result['docx_file'] = "Template not available"
            
            result['success'] = True
            logger.info(f"📄 Successfully processed: {pdf_path.name}")
            
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
        logger.info(f"📄 Starting processing of {len(all_files)} files...")
        
        results = []
        successful = 0
        failed = 0
        
        for i, file_path in enumerate(all_files, 1):
            logger.info(f"📊 Processing {i}/{len(all_files)}: {file_path.name}")
            
            result = self.process_single_resume(file_path)
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        # Create final summary report
        summary_report = {
            "total": len(all_files),
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful / len(all_files) * 100):.1f}%",
            "processing_timestamp": datetime.now().isoformat(),
            "flow_type": "PDF_TO_DOCX",
            "stages": ["parsing", "standardization", "enhancement", "docx_generation"],
            "template_available": self.has_template,
            "results": results
        }
        
        # Save final report
        report_path = self.output_dir / "pdf_to_docx_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        # Log final summary
        logger.info("=" * 80)
        logger.info("📄 PDF TO DOCX PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Total files processed: {summary_report['total']}")
        logger.info(f"✅ Successful: {summary_report['successful']}")
        logger.info(f"❌ Failed: {summary_report['failed']}")
        logger.info(f"📈 Success rate: {summary_report['success_rate']}")
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
    
    parser = argparse.ArgumentParser(description='PDF to DOCX Processing Pipeline')
    parser.add_argument('--base-dir', default='/home/shtlp_0046/Desktop/summary',
                       help='Base directory containing resumes/ folder')
    
    args = parser.parse_args()
    
    try:
        logger.info("📄 Starting PDF to DOCX Processing Pipeline")
        
        # Initialize processor
        processor = PDFToDOCXProcessor(base_dir=args.base_dir)
        
        # Process all resumes
        report = processor.process_all_resumes()
        
        # Cleanup
        processor.cleanup()
        
        # Exit with appropriate code
        if report['failed'] > 0:
            logger.warning("⚠️ Some files failed to process. Check the failed directory and logs.")
            return 1
        else:
            logger.info("📄 All DOCX files generated successfully!")
            return 0
        
    except Exception as e:
        logger.error(f"💥 Pipeline execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
