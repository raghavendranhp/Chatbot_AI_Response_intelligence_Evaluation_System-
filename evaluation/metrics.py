import json
import re
from typing import Dict, Any, Callable

class LLMEvaluator:
    """
    Uses an LLM to subjectively grade the response on complex metrics 
    that regex cannot catch (Relevance, Clarity, Coherence).
    """

    def __init__(self, llm_callable: Callable[[str], str]):
        """
        Args:
            llm_callable: A function that takes a prompt string and returns a text response.
                          (e.g., semantic_search_engine.generate_text)
        """
        self.llm_func = llm_callable

    def _construct_eval_prompt(self, query: str, context: str, ai_response: str) -> str:
        """
        Constructs the prompt for the Judge LLM.
        """
        return f"""
        You are an AI Quality Assurance Judge. Evaluate the following AI response based on the User Query and provided Context.

        ### INPUT DATA
        USER QUERY: "{query}"
        
        RETRIEVED CONTEXT (Ground Truth Information):
        "{context}"

        AI RESPONSE (To be evaluated):
        "{ai_response}"

        ### EVALUATION CRITERIA
        1. Relevance (0.0 - 1.0): Does it directly answer the user's specific question?
        2. Consistency (0.0 - 1.0): Is the information strictly derived from the Context? (0 if it hallucinates or conflicts).
        3. Completeness (0.0 - 1.0): Does it cover all parts of the query?
        4. Clarity (0.0 - 1.0): Is the language clear, professional, and well-structured?

        ### OUTPUT FORMAT
        Return ONLY a strict JSON object with no markdown formatting.
        {{
            "relevance_score": <float>,
            "consistency_score": <float>,
            "completeness_score": <float>,
            "clarity_score": <float>,
            "reasoning": "<Brief explanation of the scores>"
        }}
        """

    def _parse_json_output(self, text: str) -> Dict[str, Any]:
        """
        Robust JSON parser that handles potential Markdown wrappers from LLMs.
        """
        try:
            #Strip markdown code blocks if present
            clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse LLM evaluation",
                "raw_output": text,
                "relevance_score": 0.0, 
                "consistency_score": 0.0
            }

    def compute_metrics(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """
        Main method to run the evaluation.
        """
        prompt = self._construct_eval_prompt(query, context, response)
        
        #call the Judge LLM
        eval_output_text = self.llm_func(prompt)
        
        #parse results
        scores = self._parse_json_output(eval_output_text)
        
        #calculate a weighted overall score
        
        if "error" not in scores:
            overall = (
                scores.get("relevance_score", 0) * 0.3 +
                scores.get("consistency_score", 0) * 0.4 +
                scores.get("completeness_score", 0) * 0.2 +
                scores.get("clarity_score", 0) * 0.1
            )
            scores["overall_quality"] = round(overall, 2)
        
        return scores


if __name__ == "__main__":
   
    def mock_llm(prompt):
        return """
        {
            "relevance_score": 0.9,
            "consistency_score": 0.8,
            "completeness_score": 0.85,
            "clarity_score": 0.95,
            "reasoning": "Good answer, but missed one small detail from context."
        }
        """
    
    evaluator = LLMEvaluator(mock_llm)
    result = evaluator.compute_metrics(
        query="What is the price of the Nile Explorer?",
        context="Nile Explorer costs $850.",
        response="The Nile Explorer is a great ship and it costs $850."
    )
    print(json.dumps(result, indent=2))