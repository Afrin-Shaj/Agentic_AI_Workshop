import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize Gemini 1.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# System messages for agents
creator_system = """
You are the Content Creator Agent. Your role is to draft content on topics involving Generative AI, specifically Agentic AI.
Ensure the content is clear, concise, and technically accurate. Reflect on feedback to improve drafts iteratively.
"""

critic_system = """
You are the Content Critic Agent. Your role is to evaluate content drafted by the Content Creator Agent on Agentic AI.
Provide constructive feedback on language clarity and technical accuracy, suggesting specific improvements.
Reflect on the content to ensure feedback is actionable and precise.
"""

# Prompt templates
creator_prompt = PromptTemplate(
    input_variables=["topic", "feedback"],
    template="""
    {creator_system}
    Draft content on {topic}. If feedback is provided, revise the content based on the following feedback: {feedback}.
    Provide the content in markdown format.
    """
)

critic_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
    {critic_system}
    Evaluate the following content for language clarity and technical accuracy:
    {content}
    Provide specific, constructive feedback for improvement.
    """
)

def run_conversation(max_turns: int = 3) -> str:
    topic = "Agentic AI"
    feedback = ""
    content = ""

    for turn in range(max_turns):
        # Content Creator Agent generates or revises content
        creator_response = llm.invoke(creator_prompt.format(
            creator_system=creator_system,
            topic=topic,
            feedback=feedback
        ))
        content = creator_response.content

        # Content Critic Agent evaluates content
        critic_response = llm.invoke(critic_prompt.format(
            critic_system=critic_system,
            content=content
        ))
        feedback = critic_response.content

        # Reflection: Summarize the turn
        print(f"\nTurn {turn + 1}:")
        print("Creator Draft:\n", content)
        print("Critic Feedback:\n", feedback)

    return content

if __name__ == "__main__":
    final_content = run_conversation()
    print("\nFinal Refined Content:\n", final_content)