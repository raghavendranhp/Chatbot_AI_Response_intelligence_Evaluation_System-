from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class CruiseSearchQuery(BaseModel):
    """
    This defines the Strict JSON Schema for extracting user intent.
    The LLM must fill this form.
    """
    
    #THE INTENT
    intent: Literal["recommendation", "information", "comparison", "greeting"] = Field(
        ..., 
        description="The primary goal of the user. 'recommendation' for finding cruises, 'information' for general QA, 'comparison' for comparing options."
    )

    #THE FILTERS (For Structured/Pandas Search)
   
    cruise_type: Optional[Literal["Nile", "Red Sea"]] = Field(
        None, 
        description="The region or type of cruise explicitly mentioned."
    )
    
    min_budget: Optional[float] = Field(
        None, 
        description="Minimum price per person in USD."
    )
    
    max_budget: Optional[float] = Field(
        None, 
        description="Maximum price per person in USD."
    )
    
    min_duration: Optional[int] = Field(
        None, 
        description="Minimum number of days requested."
    )
    
    max_duration: Optional[int] = Field(
        None, 
        description="Maximum number of days requested."
    )
    
    destinations: Optional[List[str]] = Field(
        default_factory=list, 
        description="Specific cities or landmarks mentioned (e.g., 'Luxor', 'Cairo', 'Aswan')."
    )

    #SORTING PREFERENCES
    sort_by: Optional[Literal["price_asc", "price_desc", "duration_asc", "duration_desc"]] = Field(
        "price_asc", 
        description="How to sort the results. Default is cheapest first."
    )

    class Config:
        extra = "ignore"