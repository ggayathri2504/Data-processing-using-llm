from typing import List, Union
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# Pydantic models
class Suggestion(BaseModel):
    technique: str = Field(..., description="The name of the preprocessing or feature engineering technique")
    columns: List[str] = Field(..., description="List of column names to apply the technique to")
    parameters: dict = Field(default_factory=dict, description="Optional parameters for the technique")

class LLMOutput(BaseModel):
    suggestions: List[Suggestion] = Field(..., description="List of preprocessing and feature engineering suggestions")

output_parser = PydanticOutputParser(pydantic_object=LLMOutput)