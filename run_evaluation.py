import os
import sys
import pandas as pd
from src.orchestrator import CruiseOrchestrator
from evaluation.engine import EvaluationEngine


GROUND_TRUTH_PATH = "data/ground_truth/Egypt_Cruise_GroundTruth_Dataset.csv"
SAMPLE_QUERIES = [
    # 1. Simple Fact (Should pass easily)
    "What is the price of the Nile Explorer?",
    
    # 2. Complex Constraint (Needs logic)
    "Find me a luxury cruise under $1000 for 5 days.",
    
    # 3. Comparison (Needs structure)
    "Compare the Pharaoh Classic and the Royal Nile.",
    
    # 4. Hallucination Trap (Fake Cruise ID)
    "Tell me about the Galaxy Voyager (CRZ999).",
    
    # 5. Hallucination Trap (Fake Price Constraint)
    "I want a 7-day luxury cruise for $50.",
    
    # 6. Out of Scope (Should be handled gracefully)
    "Do you have cruises to the Caribbean?",
    
    # 7. Ambiguous/Broad
    "Tell me about Egypt cruises.",
    
    # 8. Specific Itinerary Detail
    "Which cruise visits Kom Ombo?"
]

def main():
    print("="*60)
    print("INITIALIZING AI RESPONSE INTELLIGENCE SYSTEM")
    print("="*60)

    # 1. Initialize the Chatbot 
    
    try:
        chatbot = CruiseOrchestrator()
        print("[INFO] Chatbot Orchestrator loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load Chatbot: {e}")
        sys.exit(1)

    # 2. Initialize the Evaluator (The Judge)
    
    try:
        
        evaluator = EvaluationEngine(
            llm_callable=chatbot.llm_client.generate_response, 
            ground_truth_path=GROUND_TRUTH_PATH
        )
        print("[INFO] Evaluation Engine loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load Evaluator: {e}")
        # Fallback for demo if LLM client isn't reachable
        sys.exit(1)

    results = []

    print("\n" + "="*60)
    print(" STARTING BATCH EVALUATION")
    print("="*60)

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n TEST CASE {i}: '{query}'")
        
        # A. Get Chatbot Response
       
        try:
            response = chatbot.process_query(query)
            
           
            if isinstance(response, dict):
                answer_text = response.get("answer", "")
                context_text = response.get("context", "")
            else:
                answer_text = str(response)
                context_text = "" # Context missing, Evaluation will be partial
                
            print(f"   AI Answer: {answer_text[:100]}...")
            
        except Exception as e:
            print(f"    Chatbot Failed: {e}")
            continue

        # B. Evaluate Response
        try:
            eval_report = evaluator.evaluate(query, answer_text, context_text)
            
            # Print Mini Report
            score = eval_report['overall_score']
            status = eval_report['status']
            flags = eval_report['flags']
            
            print(f"     Score: {score}/1.0  [{status}]")
            if flags:
                print(f"    Flags: {flags}")
            
            results.append(eval_report)
            
        except Exception as e:
            print(f"    Evaluation Failed: {e}")

    # Final Summary
    print("\n" + "="*60)
    print(" FINAL DEMO REPORT")
    print("="*60)
    
    df = pd.DataFrame(results)
    if not df.empty:
        print(f"Total Queries: {len(df)}")
        print(f"Average Score: {df['overall_score'].mean():.2f}")
        print(f"Pass Rate: {len(df[df['status'] == 'PASS']) / len(df) * 100:.1f}%")
        
        # Check for Feedback Signals
        print("\n SYSTEM TUNING SIGNALS:")
        report = evaluator.feedback.generate_report()
        if isinstance(report, dict) and "tuning_recommendations" in report:
            for rec in report["tuning_recommendations"]:
                print(f"   - {rec}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
