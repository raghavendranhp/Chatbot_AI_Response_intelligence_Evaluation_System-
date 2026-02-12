import pandas as pd
import re
import os

class RuleBasedEvaluator:
    """
    Performs deterministic checks against the Ground Truth dataset.
    Does NOT use an LLM. Relies on Regex and Pandas.
    """

    def __init__(self, ground_truth_path):
        """
        Args:
            ground_truth_path (str): Path to Egypt_Cruise_GroundTruth_Dataset.csv
        """
        if not os.path.exists(ground_truth_path):
            raise FileNotFoundError(f"Ground Truth file not found at: {ground_truth_path}")
        
        self.df = pd.read_csv(ground_truth_path)
        
        #Pre compute valid sets for fast lookup
        self.valid_cruise_ids = set(self.df['cruise_id'].str.upper().astype(str))
        self.valid_cruise_names = set(self.df['cruise_name'].str.lower())
        self.valid_cities = set(self.df['start_city'].str.lower()).union(set(self.df['end_city'].str.lower()))
        
        # rice Bounds 
        min_price = self.df['price_usd'].min()
        max_price = self.df['price_usd'].max()
        self.price_min = min_price * 0.9
        self.price_max = max_price * 1.1

    def check_hallucinated_ids(self, text):
        """
        Scans text for patterns looking like Cruise IDs (e.g., CRZ001).
        Returns a list of IDs mentioned in text that DO NOT exist in Ground Truth.
        """
        # Pattern: "CRZ" followed by 3 digits 
        
        pattern = r'CRZ\d{3}'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        hallucinated = []
        for m in matches:
            clean_id = m.upper()
            if clean_id not in self.valid_cruise_ids:
                hallucinated.append(clean_id)
        
        return list(set(hallucinated))

    def check_price_sanity(self, text):
        """
        Extracts price-like numbers ($850, 850 USD) and checks if they 
        fall reasonably within the min/max range of the entire catalog.
        """
        #Regex for $123 or 123 USD
        price_pattern = r'\$\s?(\d{3,5})|(\d{3,5})\s?(?:USD|dollars)'
        matches = re.findall(price_pattern, text)
        
        out_of_range_prices = []
        
        for m in matches:
            price_str = m[0] if m[0] else m[1]
            try:
                price = float(price_str)
                if price < self.price_min or price > self.price_max:
                    out_of_range_prices.append(price)
            except ValueError:
                continue
                
        return sorted(list(set(out_of_range_prices)))

    def evaluate(self, response_text):
        """
        Master function to run all rule checks.
        Returns a dictionary with scores and flags.
        """
        flags = []
        
        #IDCheck
        fake_ids = self.check_hallucinated_ids(response_text)
        if fake_ids:
            flags.append(f"HALLUCINATION_RISK: Mentioned invalid IDs {fake_ids}")
            
        #PriceCheck
        bad_prices = self.check_price_sanity(response_text)
        if bad_prices:
            flags.append(f"DATA_ERROR: Prices mentioned {bad_prices} are outside valid range ({self.price_min}-{self.price_max})")

        #Scoring Logic (Deterministic)
        
        score = 1.0
        if fake_ids:
            score -= 0.5  
        if bad_prices:
            score -= 0.2  
            
        return {
            "rule_score": max(0.0, score),
            "rule_flags": flags
        }


if __name__ == "__main__":
   
    evaluator = RuleBasedEvaluator(r"D:\Projects\chatbot_seshat\data\ground_truth\Egypt_Cruise_GroundTruth_Dataset.csv")
    ""
    
    print("Test 1:", evaluator.evaluate("We recommend the Nile Explorer (CRZ001) which costs $850."))
    
    
    print("Test 2:", evaluator.evaluate("Book the Galaxy Cruise (CRZ999) for only $5."))