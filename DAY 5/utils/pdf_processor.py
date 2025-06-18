import PyPDF2
import streamlit as st
from typing import Optional

class PDFProcessor:
    """Handles PDF text extraction using PyPDF2"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> Optional[str]:
        """
        Extract text from uploaded PDF file
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            str: Extracted text or None if error
        """
        try:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            
            return text.strip()
            
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None
    
    @staticmethod
    def validate_extracted_text(text: str) -> bool:
        """
        Validate if extracted text is meaningful
        
        Args:
            text: Extracted text
            
        Returns:
            bool: True if text is valid for quiz generation
        """
        if not text or len(text.strip()) < 100:
            return False
        
        # Check if text has reasonable word count
        words = text.split()
        return len(words) >= 50