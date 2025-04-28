from abc import ABC, abstractmethod
import os
from typing import Dict, List, Tuple
import json

import openai
from huggingface_hub import InferenceClient


class OpenAIClient(BaseLLMClient):
    """Handles interactions with OpenAI API"""

    def __init__(self, model: str = "gpt-4.1-nano", temperature=0, min_tokens=0, max_tokens=3000): #gpt-4
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.default_model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def validate_credentials(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def run(
            self,
            system_prompt: str,
            messages: List[Dict[str, str]],
            model: str = None
    ) -> Tuple[str, str]:
        # Use the model specified in the method call, or fall back to the default
        model_to_use = model or self.default_model
        
        formatted_messages = [{"role": "system", "content": system_prompt}]
        for message in messages:
            formatted_messages.append({
                "role": "user" if message.get("role") == "user" else "assistant",
                "content": message.get("content", "")
            })

        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=formatted_messages,
            temperature=self.temperature,  # 0 for more deterministic outputs
            max_tokens=self.max_tokens
        )

        return "", response.choices[0].message.content

from huggingface_hub import InferenceClient

class HuggingFaceClient(BaseLLMClient):
    def __init__(self, model="openchat/openchat-3.5-1210", token=None, temperature=0, min_tokens=0, max_tokens=3000):
        self.token = token or os.getenv("HF_TOKEN")
        self.model = model
        self.client = InferenceClient(token=self.token)
        self.temperature = temperature
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
    
    def validate_credentials(self) -> bool:
        try:
            # Test API connectivity with a simple request
            self.client.text_generation("Test", model=self.model, max_new_tokens=1)
            return True
        except Exception:
            return False
    
    def _format_messages(self, system_prompt, messages):
        """Format messages for Hugging Face models"""
        formatted_input = f"<s>[INST] {system_prompt}\n\n"
        
        for i, message in enumerate(messages):
            if message.get("role") == "user":
                formatted_input += f"{message.get('content', '')}"
                if i < len(messages) - 1 and messages[i+1].get("role") == "assistant":
                    formatted_input += " [/INST] "
                else:
                    formatted_input += " [/INST]"
            else:  # assistant
                formatted_input += f"{message.get('content', '')}</s>"
                if i < len(messages) - 1 and messages[i+1].get("role") == "user":
                    formatted_input += "<s>[INST] "
        
        return formatted_input
    
    def run(self, system_prompt, messages, **kwargs):
        # Format messages for the model
        formatted_input = self._format_messages(system_prompt, messages)
        response = self.client.text_generation(
            formatted_input,
            model=self.model,
            max_new_tokens=1024,
            temperature=self.temperature,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens
        )
        return "", response