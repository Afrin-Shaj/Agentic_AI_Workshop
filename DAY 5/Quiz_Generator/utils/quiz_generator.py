from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
import streamlit as st
import json
import re
from typing import List, Dict, Optional

class QuizGenerator:
    """Generates MCQ quizzes using LangChain and Gemini"""
    
    def __init__(self, api_key: str):
        """Initialize with Google API key"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.5  # Moderate temperature for creative question generation
        )
    
    def generate_mcq_quiz(self, bullet_points: List[str], num_questions: int = 5) -> Optional[List[Dict]]:
        """
        Generate MCQ quiz from bullet points
        
        Args:
            bullet_points: List of summarized bullet points
            num_questions: Number of questions to generate
            
        Returns:
            List[Dict]: List of quiz questions with options and answers
        """
        
        # Create prompt template for MCQ generation
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are an expert quiz creator. Create {num_questions} multiple-choice questions based on the following bullet points.

            Requirements for each question:
            1. Create clear, unambiguous questions
            2. Provide exactly 4 options (A, B, C, D)
            3. Only one option should be correct
            4. Make incorrect options plausible but clearly wrong
            5. Base questions directly on the provided bullet points
            6. Vary difficulty levels (easy, medium, hard)

            Bullet Points:
            {bullet_points}

            Please format your response as a JSON array with this exact structure:
            [
                {{
                    "question": "Your question here?",
                    "options": {{
                        "A": "Option A text",
                        "B": "Option B text", 
                        "C": "Option C text",
                        "D": "Option D text"
                    }},
                    "correct_answer": "A",
                    "explanation": "Brief explanation of why this is correct"
                }}
            ]

            Make sure to return valid JSON only, no additional text.
            """
        )
        
        try:
            # Prepare bullet points text
            bullet_text = '\n'.join([f"• {point}" for point in bullet_points])
            
            # Create the prompt
            prompt = prompt_template.format(
                num_questions=num_questions,
                bullet_points=bullet_text
            )
            
            # Get response from Gemini
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse JSON response
            quiz_data = self._parse_quiz_json(response.content)
            
            return quiz_data
            
        except Exception as e:
            st.error(f"Error generating quiz: {str(e)}")
            return None
    
    def _parse_quiz_json(self, response_text: str) -> List[Dict]:
        """
        Parse quiz JSON from LLM response
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            List[Dict]: Parsed quiz questions
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                quiz_data = json.loads(json_str)
                return quiz_data
            else:
                # Fallback: try to parse entire response as JSON
                quiz_data = json.loads(response_text)
                return quiz_data
                
        except json.JSONDecodeError:
            st.error("Could not parse quiz questions. Please try again.")
            return []
    
    def calculate_score(self, user_answers: Dict, quiz_questions: List[Dict]) -> Dict:
        """
        Calculate quiz score
        
        Args:
            user_answers: Dictionary of user's answers
            quiz_questions: List of quiz questions with correct answers
            
        Returns:
            Dict: Score information
        """
        total_questions = len(quiz_questions)
        correct_answers = 0
        
        for i, question in enumerate(quiz_questions):
            question_key = f"q_{i}"
            if question_key in user_answers:
                if user_answers[question_key] == question['correct_answer']:
                    correct_answers += 1
        
        score_percentage = (correct_answers / total_questions) * 100
        passed = score_percentage >= 80  # 8 out of 10 marks (80%)
        
        return {
            'correct': correct_answers,
            'total': total_questions,
            'percentage': score_percentage,
            'passed': passed
        }