from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
import streamlit as st
from typing import List, Optional

class ContentSummarizer:
    """Handles content summarization using LangChain and Gemini"""
    
    def __init__(self, api_key: str):
        """Initialize with Google API key"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3  # Lower temperature for more focused summaries
        )
    
    def create_bullet_points(self, text: str) -> Optional[List[str]]:
        """
        Convert extracted text into bullet point summaries
        
        Args:
            text: Raw text from PDF
            
        Returns:
            List[str]: List of bullet points or None if error
        """
        
        # Create prompt template for summarization
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are an educational content summarizer. Your task is to convert the given study material into clear, concise bullet points that capture the key concepts, facts, and important information.

            Instructions:
            1. Create 10-15 bullet points
            2. Each bullet point should be a complete, standalone fact or concept
            3. Focus on information that would be good for quiz questions
            4. Make sure bullet points are specific and detailed enough
            5. Avoid overly general statements
            
            Study Material:
            {text}
            
            Please provide the bullet points in the following format:
            • [Bullet point 1]
            • [Bullet point 2]
            • [And so on...]
            """
        )
        
        try:
            # Create the prompt
            prompt = prompt_template.format(text=text[:4000])  # Limit text length
            
            # Get response from Gemini
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse bullet points from response
            bullet_points = self._parse_bullet_points(response.content)
            
            return bullet_points
            
        except Exception as e:
            st.error(f"Error creating bullet points: {str(e)}")
            return None
    
    def _parse_bullet_points(self, response_text: str) -> List[str]:
        """
        Parse bullet points from LLM response
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            List[str]: Cleaned bullet points
        """
        lines = response_text.split('\n')
        bullet_points = []
        
        for line in lines:
            line = line.strip()
            # Look for bullet point indicators
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                # Remove bullet point symbol and clean
                clean_point = line[1:].strip()
                if clean_point and len(clean_point) > 10:  # Ensure meaningful content
                    bullet_points.append(clean_point)
        
        return bullet_points[:15]  # Limit to 15 bullet points