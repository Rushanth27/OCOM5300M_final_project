import json
from typing import Dict, List
from scope_coordinator.client import BaseLLMClient


class DocumentAgent:
    def __init__(self, llm_client: BaseLLMClient):
        self.client = llm_client

    def update_scope(self, initial_scope: str, clarifying_qa: List[Dict]) -> str:
        """
        Updates the initial scope based on clarifying Q&A responses

        Args:
            initial_scope: The original project scope
            clarifying_qa: List of dictionaries containing Q&A pairs

        Returns:
            str: Updated scope incorporating clarification details
        """
        system_prompt = """You are an expert project scope writer.
        Based on the initial scope and the clarifying Q&A, create an updated, more detailed scope.

        Follow these guidelines:
        1. Maintain the original scope's core objectives
        2. Incorporate new details from Q&A
        3. Add specific technical requirements identified
        4. Include risk mitigation strategies discussed
        5. Clarify any ambiguous points resolved
        6. Structure the scope with clear sections

        Format the response as a well-structured scope document with sections for:
        - Project Overview
        - Detailed Requirements
        - Technical Specifications
        - Risk Considerations
        - Dependencies and Constraints
        """

        # Format the Q&A for better context
        formatted_qa = "\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in clarifying_qa
        ])

        context = {
            "initial_scope": initial_scope,
            "clarifying_qa": formatted_qa
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
            _, updated_scope = self.client.run(system_prompt, messages)
            return updated_scope

        except Exception as e:
            print(f"Error updating scope: {str(e)}")
            return initial_scope

    def generate_document(self, project_info: Dict) -> str:
        system_prompt = """Generate a comprehensive project scope document including 
        requirements, risks, and clarifications."""

        messages = [{"role": "user", "content": json.dumps(project_info)}]
        _, response = self.client.run(system_prompt, messages)
        return response

    def update_scope(self, initial_scope: str, clarifying_qa: List[Dict]) -> str:
        """
        Updates the initial scope based on clarifying Q&A responses

        Args:
            initial_scope: The original project scope
            clarifying_qa: List of dictionaries containing Q&A pairs

        Returns:
            str: Updated scope incorporating clarification details
        """
        system_prompt = """You are an expert project scope writer.
        Based on the initial scope and the clarifying Q&A, create an updated, more detailed scope.

        Follow these guidelines:
        1. Maintain the original scope's core objectives
        2. Incorporate new details from Q&A
        3. Add specific technical requirements identified
        4. Include risk mitigation strategies discussed
        5. Clarify any ambiguous points resolved
        6. Structure the scope with clear sections

        Format the response as a well-structured scope document with sections for:
        - Project Overview
        - Detailed Requirements
        - Technical Specifications
        - Risk Considerations
        - Dependencies and Constraints
        """

        # Format the Q&A for better context
        formatted_qa = "\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in clarifying_qa
        ])

        context = {
            "initial_scope": initial_scope,
            "clarifying_qa": formatted_qa
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
            _, updated_scope = self.client.run(system_prompt, messages)

            # Add a section highlighting changes
            changes_prompt = """Identify and list the key changes and additions 
            made to the original scope based on the clarifying Q&A.
            Format as bullet points."""

            changes_context = {
                "original_scope": initial_scope,
                "updated_scope": updated_scope
            }

            changes_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(changes_context)
                        }
                    ]
                }
            ]

            _, changes_summary = self.client.run(changes_prompt, changes_messages)

#             # Combine updated scope with changes summary
#             final_scope = f"""
# # Updated Project Scope
#
# {updated_scope}
#
# # Key Updates from Clarification Process:
#
# {changes_summary}
#
# Original Scope Reference:
# ------------------------
# {initial_scope}
# """
#             return final_scope
            return updated_scope


        except Exception as e:
            print(f"Error updating scope: {str(e)}")
            return initial_scope