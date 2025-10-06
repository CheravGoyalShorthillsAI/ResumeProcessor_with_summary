#!/usr/bin/env python3
"""
Streamlit UI for Resume Processing System
Provides a web interface for uploading resumes and downloading enhanced versions + summaries
"""

import streamlit as st
import os
import sys
import tempfile
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from io import BytesIO
import warnings
import math
import html as html_lib
import streamlit.components.v1 as components

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import existing functionality without modification
from complete_flow import CompleteResumeProcessor

# Configure logging to capture processing info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Minimal wrapper to persist uploaded file across reruns
class _InMemoryUploadedFile:
    def __init__(self, data: bytes, name: str, file_type: str):
        self._data = data
        self.name = name
        self.type = file_type
    def read(self) -> bytes:
        return self._data

class StreamlitResumeProcessor:
    """
    Streamlit wrapper for the existing CompleteResumeProcessor
    Handles single file processing for the web interface
    """
    
    def __init__(self):
        self.temp_dir = None
        self.processor = None
        
    def setup_temp_environment(self):
        """Create temporary directories for processing"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="streamlit_resume_"))
        
        # Create required directory structure
        (self.temp_dir / "resumes").mkdir(parents=True, exist_ok=True)
        
        # Initialize processor with temp directory
        self.processor = CompleteResumeProcessor(base_dir=str(self.temp_dir))
        
    def process_single_resume(self, uploaded_file) -> dict:
        """
        Process a single uploaded resume file
        Returns dict with success status and file paths
        """
        if not self.temp_dir or not self.processor:
            self.setup_temp_environment()
        
        # Save uploaded file to temp resumes directory
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension not in ['.pdf', '.docx']:
            return {
                'success': False,
                'error': f'Unsupported file format: {file_extension}. Please upload PDF or DOCX files only.'
            }
        
        # Create unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{uploaded_file.name}"
        temp_file_path = self.temp_dir / "resumes" / safe_filename
        
        # Write uploaded file to temp location
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        try:
            # Process the single file using existing functionality
            result = self.processor.process_single_resume(temp_file_path)
            
            if result['success']:
                employee_id = result['employee_id']
                
                # Get paths to generated files
                docx_path = None
                summary_path = None
                
                # Check for DOCX file
                docx_files_dir = self.temp_dir / "complete_output" / "docx_files"
                if docx_files_dir.exists():
                    docx_file = docx_files_dir / f"{employee_id}_resume.docx"
                    if docx_file.exists():
                        docx_path = str(docx_file)
                
                # Check for summary file
                summaries_dir = self.temp_dir / "complete_output" / "summaries"
                if summaries_dir.exists():
                    summary_file = summaries_dir / f"{employee_id}_summary.txt"
                    if summary_file.exists():
                        summary_path = str(summary_file)
                
                return {
                    'success': True,
                    'employee_id': employee_id,
                    'docx_path': docx_path,
                    'summary_path': summary_path,
                    'stages_completed': result.get('stages_completed', []),
                    'outputs_generated': result.get('outputs_generated', [])
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown processing error')
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")
        
        if self.processor:
            self.processor.cleanup()

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="AI Resume Processor",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    /* Make code blocks (summary preview) fully visible and wrap text */
    div[data-testid="stCodeBlock"] {
        max-height: none !important;
        overflow: visible !important;
    }
    div[data-testid="stCodeBlock"] pre {
        white-space: pre-wrap !important; /* wrap long lines */
        word-break: break-word !important;
        overflow-x: hidden !important;   /* no horizontal scrollbar */
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎯 AI-Powered Resume Processor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform your resume with AI enhancement and professional formatting</p>', unsafe_allow_html=True)
    
    # Sidebar intentionally minimal (removed info panel per request)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Resume")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=['pdf', 'docx'],
            help="Upload a PDF or DOCX resume file for AI-powered enhancement"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.markdown(f"**📁 File:** {uploaded_file.name}")
            st.markdown(f"**📊 Size:** {uploaded_file.size:,} bytes")
            st.markdown(f"**🔤 Type:** {uploaded_file.type}")
            
            # Process button
            if st.button("🚀 Process Resume", type="primary", use_container_width=True):
                # Initialize processor
                processor = StreamlitResumeProcessor()
                
                try:
                    with st.spinner("🔄 Processing your resume through AI pipeline..."):
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Update progress
                        status_text.text("🔍 Setting up processing environment...")
                        progress_bar.progress(20)
                        
                        status_text.text("📄 Parsing resume content...")
                        progress_bar.progress(40)
                        
                        status_text.text("🎯 Standardizing and enhancing...")
                        progress_bar.progress(60)
                        
                        status_text.text("📝 Generating summary...")
                        progress_bar.progress(80)
                        
                        # Process the file
                        result = processor.process_single_resume(uploaded_file)
                        
                        status_text.text("✅ Processing complete!")
                        progress_bar.progress(100)
                        
                        # Store result in session state
                        st.session_state['processing_result'] = result
                        st.session_state['processor'] = processor
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    if 'processor' in locals():
                        processor.cleanup()
    
    with col2:
        st.header("📥 Download Results")
        
        # Check if we have processing results
        if 'processing_result' in st.session_state:
            result = st.session_state['processing_result']
            
            if result['success']:
                # Success banner and processing details removed per UX request
                
                # Download buttons
                st.markdown("### 📁 Download Files")
                
                # DOCX Download
                if result.get('docx_path') and os.path.exists(result['docx_path']):
                    with open(result['docx_path'], 'rb') as f:
                        docx_data = f.read()
                    
                    st.download_button(
                        label="📄 Download Enhanced Resume (DOCX)",
                        data=docx_data,
                        file_name=f"{result['employee_id']}_enhanced_resume.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
                else:
                    st.warning("⚠️ DOCX file not available - template may be missing")
                
                # Summary Preview (copyable)
                if result.get('summary_path') and os.path.exists(result['summary_path']):
                    with open(result['summary_path'], 'r', encoding='utf-8') as f:
                        summary_data = f.read()
                    
                    with st.expander("Click to preview professional summary", expanded=True):
                        # Render a copyable, non-scrolling summary block with a copy button
                        safe_html_summary = html_lib.escape(summary_data)
                        avg_chars_per_line = 110
                        estimated_lines = max(3, math.ceil(len(summary_data) / avg_chars_per_line))
                        component_height = min(1000, 24 * estimated_lines + 70)
                        container_html = (
                            """
                            <div style=\"position: relative; border: 1px solid #e6e9ef; background: #f8fafc; border-radius: 8px; padding: 12px 44px 12px 12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; line-height: 1.6; white-space: pre-wrap; word-break: break-word; max-height: 60vh; overflow-y: auto;\">\n\
                              <button id=\"copyBtn\" style=\"position:absolute; top:8px; right:8px; padding:6px 10px; font-size:12px; border:1px solid #d0d7de; background:#fff; border-radius:6px; cursor:pointer;\">Copy</button>\n\
                              <div id=\"summaryText\">"""
                            + safe_html_summary +
                            """</div>\n\
                            </div>
                            """
                        )
                        json_text = json.dumps(summary_data)
                        script_str = (
                            "<script>\n"
                            + "const text = " + json_text + ";\n"
                            + "const btn = document.getElementById('copyBtn');\n"
                            + "btn.addEventListener('click', async () => {\n"
                            + "  try {\n"
                            + "    await navigator.clipboard.writeText(text);\n"
                            + "    btn.innerText = 'Copied!';\n"
                            + "  } catch (e) {\n"
                            + "    const el = document.createElement('textarea');\n"
                            + "    el.value = text; document.body.appendChild(el); el.select(); document.execCommand('copy'); document.body.removeChild(el);\n"
                            + "    btn.innerText = 'Copied!';\n"
                            + "  }\n"
                            + "  setTimeout(()=>btn.innerText='Copy', 1500);\n"
                            + "});\n"
                            + "</script>"
                        )
                        components.html(container_html + script_str, height=500, scrolling=True)
                else:
                    st.warning("⚠️ Summary not available - Google AI API key may be missing")
                
                # Cleanup button
                if st.button("🧹 Clear Results", use_container_width=True):
                    if 'processor' in st.session_state:
                        st.session_state['processor'].cleanup()
                    
                    # Clear session state
                    for key in ['processing_result', 'processor']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.rerun()
            
            else:
                st.markdown(f'<div class="error-box">❌ <strong>Processing Failed:</strong><br>{result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
                
                # Cleanup button for failed processing
                if st.button("🧹 Clear and Try Again", use_container_width=True):
                    if 'processor' in st.session_state:
                        st.session_state['processor'].cleanup()
                    
                    # Clear session state
                    for key in ['processing_result', 'processor']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.rerun()
        
        
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🤖 Powered by AI: LlamaParse • Azure OpenAI • Google AI</p>
        <p>Built with Streamlit • Enhanced with CARS Framework</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
