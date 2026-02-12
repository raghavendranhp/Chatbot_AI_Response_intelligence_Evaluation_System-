import os
import json
from dotenv import load_dotenv  
from .models import CruiseSearchQuery
from .extractor import IntentExtractor
from .structured_ops import StructuredSearchEngine
from .unstructured_ops import UnstructuredSearchEngine
from groq import Groq

#Load environment variables from .env file
load_dotenv()  
class CruiseOrchestrator:
    """
    The Central Brain of SeShat AI.
    It coordinates:
    1. Intent Understanding (What does the user want?)
    2. Data Retrieval (Fetching rows from CSV or text from Docs)
    3. Response Synthesis (Generating the final natural language answer)
    """

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set. Please check your .env file.")
            
        self.llm_client = Groq(api_key=self.api_key)
        
        #initialize Sub-Components
        self.extractor = IntentExtractor(self.llm_client)
        self.structured_engine = StructuredSearchEngine()
        self.unstructured_engine = UnstructuredSearchEngine()

        #load System Prompts
        self.synthesizer_prompt = self._load_prompt("config/synthesizer_prompt.txt")

    def _load_prompt(self, filepath):
        """Helper to load prompt templates"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            #fallback if file is missing
            return "You are a helpful assistant for Egypt Cruises."

    def generate_response(self, prompt):
        """
        Direct wrapper for LLM generation (Used by Evaluator)
        """
        response = self.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content

    def process_query(self, user_query: str):
        """
        Main execution flow.
        Returns a DICTIONARY (not just text) to enable Evaluation.
        """
        #extract Intent & Filters
        extraction_data = self.extractor.extract_intent(user_query)
        
        #if extraction failed, fallback to general info
        if not extraction_data:
            intent = "information"
            filters = {}
        else:
            intent = extraction_data.intent
            filters = extraction_data.dict()

        #retrieve Data 
        context_text = ""
        sources = []
        
        if intent in ["recommendation", "comparison"]:
            #route to CSV Data
            context_text = self.structured_engine.search(filters)
            sources = ["Structured Database (CSV)"]
            
        elif intent in ["information", "greeting"]:
            #route to RAG (Text Docs)
            docs = self.unstructured_engine.search(user_query)
            context_text = "\n\n".join(docs)
            sources = ["Unstructured Documents (RAG)"]
        
        #f allback if no data found
        if not context_text:
            context_text = "No specific data found for this query in the database."

        #synthesize Answer
        final_prompt = (
            f"{self.synthesizer_prompt}\n\n"
            f"=== USER INTENT: {intent.upper()} ===\n"
            f"=== RETRIEVED DATA ===\n{context_text}\n\n"
            f"=== USER QUESTION ===\n{user_query}"
        )

        final_answer = self.generate_response(final_prompt)

        #return Structured Output
        return {
            "answer": final_answer,
            "context": context_text,
            "sources": sources,
            "intent_detected": intent
        }