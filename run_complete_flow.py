#!/usr/bin/env python3
"""
Simple runner for Complete End-to-End flow (PDF → Summary + DOCX)
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from complete_flow import CompleteResumeProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Optional: Set Google AI API key for summary generation
    # os.environ['GOOGLE_AI_API_KEY'] = 'your_google_ai_api_key_here'
    
    try:
        logger.info("🎯 Starting Complete End-to-End Processing")
        logger.info("📋 This generates: JSON + Summaries + DOCX files")
        
        processor = CompleteResumeProcessor(base_dir=str(current_dir))
        report = processor.process_all_resumes()
        processor.cleanup()
        
        logger.info(f"✅ Processed {report['successful']}/{report['total']} files successfully")
        logger.info(f"📁 Results in: complete_output/")
        logger.info("📊 Generated outputs:")
        for output_type, count in report.get('output_counts', {}).items():
            logger.info(f"   {output_type}: {count} files")
        
        return 0 if report['failed'] == 0 else 1
        
    except Exception as e:
        logger.error(f"💥 Failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
