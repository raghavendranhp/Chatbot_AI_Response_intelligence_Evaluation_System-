import time
import os
from .rules import RuleBasedEvaluator
from .metrics import LLMEvaluator
from .feedback import FeedbackManager

class EvaluationEngine:
    """
    The Central Intelligence Layer.
    It orchestrates the scoring process by running both Rule-based and LLM-based checks,
    aggregating the scores, and logging the results for future tuning.
    """

    def __init__(self, llm_callable, ground_truth_path="data/ground_truth/Egypt_Cruise_GroundTruth_Dataset.csv", log_dir="logs"):
        """
        Args:
            llm_callable: Function to generate text from LLM (passed to metrics.py).
            ground_truth_path: Path to the verified dataset.
            log_dir: Directory to store evaluation logs.
        """
        #Initialize Sub-Components
        print(f"[Init] Loading Rule Engine with {ground_truth_path}...")
        self.rules = RuleBasedEvaluator(ground_truth_path)
        
        print("[Init] Loading LLM Metric Judge...")
        self.metrics = LLMEvaluator(llm_callable)
        
        print("[Init] Connecting to Feedback Manager...")
        self.feedback = FeedbackManager(log_dir=log_dir)

    def evaluate(self, query, response, context=""):
        """
        Main method to grade a chatbot interaction.
        
        Args:
            query (str): The user's question.
            response (str): The chatbot's answer.
            context (str): The retrieved text (RAG context) used to generate the answer.
            
        Returns:
            dict: A comprehensive scorecard.
        """
        start_time = time.time()

        #Run Deterministic Rules (Fast, Objective)
        rule_result = self.rules.evaluate(response)

        #run LLM Judge
        metric_result = self.metrics.compute_metrics(query, context, response)

        #calculate Composite Score
        rule_score = rule_result.get("rule_score", 1.0)
        llm_score = metric_result.get("overall_quality", 0.0)
        
        final_score = (rule_score * 0.4) + (llm_score * 0.6)

        #consolidate flags
        all_flags = rule_result.get("rule_flags", [])
        if metric_result.get("hallucination_risk", "LOW") == "HIGH":
            all_flags.append("LLM_FLAG: Potential Hallucination detected by Model")

        #construct final packet
        evaluation_packet = {
            "query": query,
            "overall_score": round(final_score, 2),
            "status": "PASS" if final_score > 0.7 else "REVIEW_NEEDED",
            "scores": {
                "rule_adherence": rule_score,
                "llm_quality": llm_score,
                "components": metric_result 
            },
            "flags": all_flags,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }

        #log to feedback loop
        self.feedback.log_event(query, response, metric_result, rule_result)

        return evaluation_packet