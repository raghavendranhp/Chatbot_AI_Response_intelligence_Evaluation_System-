import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.models import CruiseSearchQuery


load_dotenv()

import json
import re
from .models import CruiseSearchQuery

class IntentExtractor:
    """
    Extracts structured intent (filters, budget, duration) from natural language
    using an LLM to fill the strict Pydantic model 'CruiseSearchQuery'.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.system_prompt = self._load_prompt("config/extractor_prompt.txt")

    def _load_prompt(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "You are an AI that converts natural language queries into strict JSON."

    def extract_intent(self, user_query: str):
        """
        Returns a CruiseSearchQuery object (Pydantic model) or None if parsing fails.
        """
        #Construct Prompt
        
        schema_json = CruiseSearchQuery.model_json_schema()
        
        prompt = f"""
        {self.system_prompt}
        
        STRICT OUTPUT SCHEMA (JSON):
        {json.dumps(schema_json, indent=2)}
        
        USER QUERY: "{user_query}"
        
        Return ONLY valid JSON. No markdown formatting.
        """

        #Call LLM
        
        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"} 
            )
            content = response.choices[0].message.content
            
        except Exception as e:
            print(f"[Extractor] LLM Call Failed: {e}")
            return None

        #Parse JSON & Validate
        try:
            #Clean potential markdown 
            clean_json = re.sub(r'```json\s*|\s*```', '', content).strip()
            data = json.loads(clean_json)
            
            #Convert dictionary to Pydantic Model 
            return CruiseSearchQuery(**data)
            
        except json.JSONDecodeError:
            print(f"[Extractor] Failed to parse JSON: {content}")
            return None
        except Exception as e:
            print(f"[Extractor] Validation Failed: {e}")
            return None