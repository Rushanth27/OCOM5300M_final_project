"""Risk identification agent for generating potential risks from project scope."""
from typing import Dict, List
import json

from scope_coordinator.client import BaseLLMClient

class RiskIdentificationAgent:
    """Agent for identifying potential risks from project scope."""

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize the risk identification agent.

        Args:
            llm_client: The LLM client to use for generating responses
        """
        self.client = llm_client

    def identify_risks(self, project_scope: str) -> List[Dict]:
        """
        Analyze project scope and identify potential risks.

        Args:
            project_scope: The project scope description

        Returns:
            List of dictionaries containing identified risks
        """
        system_prompt = """You are an expert risk identification specialist. 
        Your task is to analyze the project scope and identify specific risks.
        
        Return your analysis as a JSON array where each object has the format:
        [
            {
                "category": "risk category",
                "description": "clear description of the risk",
                "consequences": "potential consequences"
            }
        ]
        
        Consider these risk categories:
        - Technical Risks
        - Schedule Risks
        - Resource Risks
        - Operational Risks
        - Security Risks
        - Compliance Risks
        - Integration Risks
        - Performance Risks
        
        Focus on specific, actionable risks rather than generic ones.
        Ensure each risk description is clear and detailed enough for accurate assessment.
        
        Return ONLY the JSON array with no additional text or formatting."""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": project_scope
                    }
                ]
            }
        ]

        try:
            _, response = self.client.run(system_prompt, messages)

            # Clean up the response
            response = response.strip()
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "")
            elif response.startswith("```"):
                response = response.replace("```", "")

            response = response.strip()

            risks = json.loads(response)

            # Validate and clean up the risks
            validated_risks = []
            for risk in risks:
                if all(key in risk for key in ["category", "description", "consequences"]):
                    validated_risks.append({
                        "category": risk["category"].strip(),
                        "description": risk["description"].strip(),
                        "consequences": risk["consequences"].strip()
                    })

            return validated_risks

        except Exception as e:
            print(f"Error identifying risks: {str(e)}")
            print(f"Raw response: {response}")  # Add this for debugging
            # Return a default risk if identification fails
            return [{
                "category": "General Risk",
                "description": "Unable to identify specific risks from the provided scope",
                "consequences": "Potential unknown impacts to project success"
            }]

    def display_identified_risks(self, risks: List[Dict]):
        """
        Display identified risks in a formatted way.

        Args:
            risks: List of identified risks
        """
        print("\n=== Identified Risks ===")
        for i, risk in enumerate(risks, 1):
            print(f"\nRisk {i}:")
            print(f"Category: {risk['category']}")
            print(f"Description: {risk['description']}")
            print(f"Consequences: {risk['consequences']}")
            print("-" * 50)
