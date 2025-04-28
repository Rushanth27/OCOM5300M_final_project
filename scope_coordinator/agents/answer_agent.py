from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class AnswerAgent:
    """Agent responsible for generating answers to clarifying questions"""

    def __init__(self, llm_client):
        """
        Initialize the answer agent.

        Args:
            llm_client: The LLM client to use for generating responses
        """
        self.client = llm_client

    def generate_answer(self, question: str, risk_description: str, risk_category: str) -> str:
        """
        Generate an answer for a clarifying question based on the context, limited to 100 words.

        Args:
            question: The clarifying question
            risk_description: The description of the risk being addressed
            risk_category: The category of the risk

        Returns:
            str: Generated answer (100 words or less)
        """
        system_prompt = """You are an expert project manager and technical lead.
        Your task is to provide clear, specific answers to clarifying questions about project risks.
        Provide detailed, practical responses that address the question directly. Do not ask further questions.
        IMPORTANT: Limit your response to 100 words or less."""

        messages = [{
            "role": "user",
            "content": f"""
            Context:
            Risk Category: {risk_category}
            Risk Description: {risk_description}

            Question: {question}

            Please provide a concise answer (maximum 100 words) to this question that addresses the specific risk and helps clarify the project scope."""
        }]

        _, response = self.client.run(
            system_prompt=system_prompt,
            messages=messages
        )

        return response
