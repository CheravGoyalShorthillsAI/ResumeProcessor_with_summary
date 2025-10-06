# 🎯 AI-Powered Resume Processing System

A comprehensive AI-driven resume processing platform that transforms PDF and DOCX resumes into enhanced, standardized formats with professional summaries and formatted outputs. Features both command-line processing and a modern Streamlit web interface.

## ✨ Key Features

- **🤖 AI-Powered Enhancement**: Uses Azure OpenAI and Google AI for intelligent resume processing
- **📄 Multi-Format Support**: Processes both PDF and DOCX files with advanced parsing
- **🎨 Professional Templates**: Generates beautifully formatted DOCX resumes using custom templates
- **📝 Smart Summaries**: Creates comprehensive professional summaries using Google AI
- **🌐 Web Interface**: Modern Streamlit UI for easy file upload and processing
- **🔧 Command Line Tools**: Batch processing capabilities for enterprise use
- **📊 Comprehensive Analytics**: Detailed processing reports and usage tracking

## 🚀 Quick Start

### Prerequisites

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment Variables** (Create `.env` file in project root)
   ```bash
   # Required for parsing
   LLAMA_CLOUD_API_KEY=your_llama_parse_api_key
   
   # Required for standardization and enhancement
   AZURE_OPENAI_API_KEY=your_azure_openai_key
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   AZURE_OPENAI_DEPLOYMENT=your_deployment_name
   AZURE_OPENAI_API_VERSION=2024-02-01
   
   # Required for summary generation
   GOOGLE_AI_API_KEY=your_google_ai_api_key
   ```

3. **Prepare Template** (Ensure `Resume_Final_draft.docx` is in project root)

## 🎮 Usage Options

### 🌐 **Web Interface (Recommended)**
```bash
python3 run_streamlit_app.py
```
- **URL**: http://localhost:8501
- **Features**: Drag & drop upload, real-time processing, instant downloads
- **Perfect for**: Individual resume processing, testing, demos

### 🎯 **Complete End-to-End Flow**
```bash
python3 run_complete_flow.py
```
**Pipeline**: PDF/DOCX → Parse → Standardize → **AI Enhance** → Summary + DOCX Generation  
**Outputs**: `complete_output/`
- `parsed/` - Raw extracted content with links
- `standardized/` - Structured JSON data
- `enhanced/` - AI-enhanced resume content
- `summaries/` - Professional summary TXT files  
- `docx_files/` - Formatted DOCX resumes

### 📄 **PDF to DOCX Flow**
```bash
python3 run_pdf_to_docx.py
```
**Pipeline**: PDF/DOCX → Parse → Standardize → **AI Enhance** → DOCX Generation  
**Outputs**: `docx_output/`
- `parsed/` - Raw extracted content
- `standardized/` - Structured JSON data
- `enhanced/` - AI-enhanced content
- `docx_files/` - Formatted DOCX resumes

## 📁 Project Structure

```
summary/
├── streamlit_app.py          # Web interface application
├── complete_flow.py          # End-to-end processing pipeline
├── pdf_to_docx_flow.py      # PDF to DOCX conversion flow
├── retailor.py              # AI resume enhancement engine
├── parser.py                # Resume parsing utilities
├── docx_utils.py            # DOCX template processing
├── run_*.py                 # Execution scripts
├── Resume_Final_draft.docx  # DOCX template file
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
├── resumes/                 # Input directory for batch processing
├── complete_output/         # Complete flow outputs
└── docx_output/            # PDF-to-DOCX flow outputs
```

## 🔧 Processing Pipeline

### Stage 1: **Intelligent Parsing**
- **LlamaParse**: Advanced PDF/DOCX content extraction with structure preservation
- **OCR Fallback**: Tesseract OCR for challenging documents
- **Link Extraction**: Automatic hyperlink detection and preservation
- **Multi-format Support**: Handles PDF, DOCX with consistent results

### Stage 2: **AI Standardization**
- **Azure OpenAI**: Converts raw content to structured JSON format
- **Schema Validation**: Ensures consistent data structure
- **Content Normalization**: Standardizes formatting and organization
- **Metadata Tracking**: Comprehensive processing analytics

### Stage 3: **AI Enhancement**
- **Resume Retailoring**: Intelligent content improvement and optimization
- **Skills Enhancement**: Technology stack analysis and expansion
- **Experience Optimization**: Professional description enhancement
- **Industry Alignment**: Content tailored for professional standards

### Stage 4: **Summary Generation**
- **Google AI Integration**: Advanced natural language generation
- **Professional Formatting**: LinkedIn-ready summary creation
- **Comprehensive Analysis**: 200-300 word detailed summaries
- **Industry Context**: Role-specific and domain-aware content

### Stage 5: **Document Generation**
- **Template-Based**: Professional DOCX formatting using custom templates
- **Consistent Styling**: Uniform appearance across all outputs
- **Preservation of Links**: Maintains hyperlinks and formatting
- **Print-Ready**: Professional quality document output

## 📊 Output Examples

### Generated Files
- **Enhanced Resume**: `{employee_id}_resume.docx`
- **Professional Summary**: `{employee_id}_summary.txt`
- **Structured Data**: `{employee_id}.json` (parsed, standardized, enhanced)
- **Processing Report**: `complete_processing_report.json`

### Summary Sample
```
John Doe is a Senior Software Engineer with 8+ years of experience in full-stack 
development and cloud architecture. He specializes in Python, React, AWS, and 
microservices architecture, with proven expertise in building scalable web 
applications serving millions of users...
```

## 🛠️ Configuration

### API Requirements
- **LlamaParse**: Document parsing and structure extraction
- **Azure OpenAI**: Content standardization and structuring  
- **Google AI**: Professional summary generation

### Optional Features
- **OCR Processing**: Requires Tesseract installation
- **Image Processing**: PDF to image conversion for OCR
- **Template Customization**: Modify `Resume_Final_draft.docx`

## 📈 Batch Processing

For processing multiple resumes:

1. Place files in `resumes/` directory
2. Run: `python3 run_complete_flow.py`
3. Monitor progress in terminal logs
4. Check outputs in `complete_output/` directory
5. Review processing report for analytics

## 🎨 Web Interface Features

- **Drag & Drop Upload**: Easy file selection
- **Real-time Progress**: Visual processing indicators  
- **Instant Preview**: Summary preview with copy functionality
- **One-Click Download**: Enhanced resume and summary downloads
- **Error Handling**: Comprehensive error reporting and recovery
- **Mobile Responsive**: Works on all device sizes

## 🔍 Troubleshooting

### Common Issues
- **Missing API Keys**: Check `.env` file configuration
- **Template Not Found**: Ensure `Resume_Final_draft.docx` exists
- **Processing Failures**: Check logs in `*_processing.log` files
- **OCR Issues**: Verify Tesseract installation

### Support
- Check processing logs for detailed error information
- Verify all API keys are valid and have sufficient credits
- Ensure input files are not corrupted or password-protected

## 🚀 Technology Stack

- **Backend**: Python 3.8+, FastAPI concepts
- **AI/ML**: Azure OpenAI, Google AI, LlamaParse
- **Document Processing**: PyMuPDF, python-docx, docxtpl
- **OCR**: Tesseract, pdf2image, Pillow
- **Web Interface**: Streamlit
- **Data**: JSON, structured schemas

---

**🤖 Powered by**: LlamaParse • Azure OpenAI • Google AI  
**🎯 Built with**: Python • Streamlit • Advanced AI Pipeline