"""ClarificationAgent for identifying gaps and generating questions."""
import json
from typing import Dict, List

from scope_coordinator.client import BaseLLMClient


class ClarificationAgent:
    def __init__(self, llm_client: BaseLLMClient):
        self.client = llm_client
        self.max_clarifying_questions = 3  # Add this constant

    def identify_gaps(self, risk_description: str, impact: Dict, likelihood: Dict) -> List[str]:
        print("Identifying gaps...")
        system_prompt = """You are an expert risk assessment clarification assistant. 
        Based on the provided risk assessment, identify specific information gaps that need clarification.

        For the given risk assessment, identify:
        1. Missing technical details needed to assess the impact level
        2. Unclear factors that affect likelihood assessment
        3. Information needed for risk mitigation planning
        4. Dependencies or context that could affect the risk assessment
        5. Historical data or past incidents that could validate the assessment

        Return ONLY a JSON array of strings, where each string describes a specific gap that needs clarification.
        Each gap should be directly tied to understanding or validating the risk assessment.
        Format: ["Need clarification on <specific aspect>", ...]

        Important: Limit your response to the 3 most critical gaps that need clarification.

        Do not include any other text or formatting in your response."""

        # Format risk assessment for the prompt
        context = {
            "risk_description": risk_description,
            "impact": {
                "value": impact["value"],
                "level": impact["level"]
            },
            "likelihood": {
                "value": likelihood["value"],
                "level": likelihood["level"]
            }
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(context)
                    }
                ]
            }
        ]

        try:
            _, response = self.client.run(system_prompt, messages)
            gaps = json.loads(response)
            return gaps
        except Exception as e:
            print(f"Error identifying gaps: {str(e)}")
            return []

    def analyze_risks(self, risk_description: str) -> Dict:
        """Analyze risks to determine key areas needing clarification."""
        system_prompt = """Analyze the provided risk assessment and categorize areas needing clarification.
        You must return your response in the following JSON format only:
        {
            "high_priority": [list of aspects needing immediate clarification],
            "dependencies": [list of potential dependencies affecting the risk],
            "impact_areas": [key areas affected by this risk]
        }

        Consider:
        1. The relationship between impact and likelihood values
        2. The context of the risk description
        3. Potential technical and business implications
        4. Dependencies and downstream effects

        Do not include any additional text, explanations, or formatting.
        Ensure the response is valid JSON."""

        context = {
            "risk_description": risk_description,
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(context, indent=2)
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

            try:
                analysis = json.loads(response)
                required_keys = ["high_priority", "dependencies", "impact_areas"]
                for key in required_keys:
                    if key not in analysis:
                        analysis[key] = []
                return analysis

            except json.JSONDecodeError as je:
                print(f"JSON parsing error: {str(je)}")
                print(f"Raw response: {response}")
                return {
                    "high_priority": [],
                    "dependencies": [],
                    "impact_areas": []
                }

        except Exception as e:
            print(f"Error analyzing risks: {str(e)}")
            return {
                "high_priority": [],
                "dependencies": [],
                "impact_areas": []
            }

    def generate_questions(self, risk_description: str) -> List[str]:
        print(f"Generating questions for risk description: {risk_description}")
        """Generate questions based on gaps and analyzed risks."""
        # First analyze the risks
        risk_analysis = self.analyze_risks(risk_description)

        system_prompt = """Generate specific questions to address the risk assessment provided.
        Focus on:
        1. Validating the impact assessment
        2. Understanding likelihood factors
        3. Identifying potential dependencies
        4. Gathering historical context
        5. Understanding mitigation options

        Return ONLY a JSON array of strings, where each string is a specific question.
        Format each question to clearly reference the aspect being questioned:
        ["Regarding <aspect>, what are the specific <details needed>?", ...]

        Do not include any other text or formatting in your response."""

        context = {
            "risk_description": risk_description,
            "analysis": risk_analysis
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(context, indent=2)
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

            questions = json.loads(response)

            # Ensure questions are properly formatted
            formatted_questions = []
            for q in questions:
                if not any(word in q.lower() for word in ["impact", "likelihood", "risk"]):
                    q = f"Regarding the risk assessment: {q}"
                formatted_questions.append(q)

            return formatted_questions

        except Exception as e:
            print(f"Error generating questions: {str(e)}")
