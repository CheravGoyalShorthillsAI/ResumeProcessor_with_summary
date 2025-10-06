#!/usr/bin/env python3
"""
Simple runner for PDF to DOCX flow
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from pdf_to_docx_flow import PDFToDOCXProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("📄 Starting PDF to DOCX Processing")
        
        processor = PDFToDOCXProcessor(base_dir=str(current_dir))
        report = processor.process_all_resumes()
        processor.cleanup()
        
        logger.info(f"✅ Processed {report['successful']}/{report['total']} files successfully")
        logger.info(f"📁 Results in: docx_output/")
        
        return 0 if report['failed'] == 0 else 1
        
    except Exception as e:
        logger.error(f"💥 Failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
