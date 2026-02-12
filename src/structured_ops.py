import pandas as pd
import os
from src.models import CruiseSearchQuery


_DF_CACHE = None
CSV_PATH = os.path.join("data", "structured", "Egypt_Cruise_Dataset.csv")

import pandas as pd
import os

class StructuredSearchEngine:
    """
    Handles deterministic searches on the CSV dataset (filtering by price, duration, etc.)
    """

    def __init__(self, data_path="data/structured/Egypt_Cruise_Dataset.csv"):
        #Check if file exists to prevent crashing
        if not os.path.exists(data_path):
            print(f"[WARNING] Structured data not found at {data_path}. Creating empty DataFrame.")
            self.df = pd.DataFrame(columns=["cruise_name", "price_usd", "duration_days", "cruise_type", "start_city", "end_city"])
        else:
            self.df = pd.read_csv(data_path)
            
            
            self.df.columns = [c.strip().lower().replace(" ", "_") for c in self.df.columns]

    def search(self, filters: dict) -> str:
        """
        Applies filters (budget, duration, type) and returns a string summary of matching cruises.
        """
        if self.df.empty:
            return "No cruise data available."

        #Start with full dataset
        results = self.df.copy()

        #Filter by Cruise Type 
        if filters.get("cruise_type"):
            
            results = results[results["cruise_type"].str.contains(filters["cruise_type"], case=False, na=False)]

        #Filter by Budget
        if filters.get("min_budget"):
            results = results[results["price_usd"] >= filters["min_budget"]]
        if filters.get("max_budget"):
            results = results[results["price_usd"] <= filters["max_budget"]]

        #Filter by Duration
        if filters.get("min_duration"):
            results = results[results["duration_days"] >= filters["min_duration"]]
        if filters.get("max_duration"):
            results = results[results["duration_days"] <= filters["max_duration"]]

        #Filter by Destinations 
        if filters.get("destinations"):
            for dest in filters["destinations"]:
                
                mask = (
                    results["start_city"].str.contains(dest, case=False, na=False) | 
                    results["end_city"].str.contains(dest, case=False, na=False)
                )
                results = results[mask]

        #Sort Results
        sort_mode = filters.get("sort_by", "price_asc")
        if sort_mode == "price_asc":
            results = results.sort_values("price_usd", ascending=True)
        elif sort_mode == "price_desc":
            results = results.sort_values("price_usd", ascending=False)
        elif sort_mode == "duration_desc":
            results = results.sort_values("duration_days", ascending=False)

        #Format Output
        if results.empty:
            return "No cruises found matching your specific criteria."
        
        #Convert top 5 results to string
        top_results = results.head(5).to_dict(orient="records")
        output_text = "Found the following matching cruises:\n"
        for i, row in enumerate(top_results, 1):
            output_text += (
                f"{i}. {row.get('cruise_name', 'Unnamed Cruise')} "
                f"({row.get('duration_days')} days) - ${row.get('price_usd')}\n"
                f"   Route: {row.get('start_city')} -> {row.get('end_city')}\n"
            )
            
        return output_text