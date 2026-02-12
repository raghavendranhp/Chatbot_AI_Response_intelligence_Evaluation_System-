import json
import os
from datetime import datetime
from collections import defaultdict
import statistics

class FeedbackManager:
    """
    Manages the storage and analysis of evaluation data.
    Implements the "Feedback Loop" by aggregating scores to suggest improvements.
    """

    def __init__(self, log_dir="logs", log_file="eval_history.jsonl"):
        """
        Args:
            log_dir: Directory to store logs.
            log_file: Filename for the JSON Lines log.
        """
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_file)
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def log_event(self, query, response, metrics, rule_results):
        """
        Saves a single interaction + evaluation to the log file.
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response_preview": response[:100] + "...",
            "metrics": metrics, 
            "rules": rule_results }
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def generate_report(self):
        """
        Reads the log file and calculates aggregate performance stats.
        Returns a dictionary summary and tuning recommendations.
        """
        if not os.path.exists(self.log_path):
            return "No logs found."

        stats = defaultdict(list)
        total_events = 0
        hallucination_count = 0

        #Read and Aggregate
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                total_events += 1
                
                #Aggregate LLM Metrics
                if "error" not in data["metrics"]:
                    stats["relevance"].append(data["metrics"].get("relevance_score", 0))
                    stats["consistency"].append(data["metrics"].get("consistency_score", 0))
                    stats["clarity"].append(data["metrics"].get("clarity_score", 0))
                
                #Count Rule Failures
                if data["rules"].get("rule_flags"):
                    hallucination_count += 1

        if total_events == 0:
            return "Log file is empty."

        #Calculate Averages
        summary = {
            "total_queries": total_events,
            "avg_relevance": statistics.mean(stats["relevance"]) if stats["relevance"] else 0,
            "avg_consistency": statistics.mean(stats["consistency"]) if stats["consistency"] else 0,
            "avg_clarity": statistics.mean(stats["clarity"]) if stats["clarity"] else 0,
            "hallucination_rate": f"{(hallucination_count / total_events) * 100:.1f}%"
        }

        #Generate Tuning Signals 
        signals = []
        if summary["avg_relevance"] < 0.7:
            signals.append("CRITICAL: Relevance is low. Review 'Retrieval' logic (RAG) or Vector DB embeddings.")
        if summary["avg_consistency"] < 0.8:
            signals.append("WARNING: Consistency is low. Stricter System Prompt constraints needed regarding 'Context Usage'.")
        if summary["avg_clarity"] < 0.6:
            signals.append("ADVICE: Responses are unclear. Tweak the 'Synthesizer' prompt for better formatting.")
        if (hallucination_count / total_events) > 0.1:
            signals.append("URGENT: High Hallucination Rate (>10%). Review Ground Truth validation rules.")

        summary["tuning_recommendations"] = signals
        return summary


if __name__ == "__main__":
    fb = FeedbackManager()
    
    
    fb.log_event("Q1", "Ans1", {"relevance_score": 0.9, "consistency_score": 0.9, "clarity_score": 1.0}, {"rule_flags": []})
    fb.log_event("Q2", "Ans2", {"relevance_score": 0.4, "consistency_score": 0.5, "clarity_score": 0.8}, {"rule_flags": ["Bad Price"]})
    
    
    print(json.dumps(fb.generate_report(), indent=2))