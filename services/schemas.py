from pydantic import BaseModel, Field
from typing import Dict, List


class GeminiFeedback(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    model_config = {
        "extra": "ignore"
    }


class EnhancedResumeResponse(BaseModel):
    enhanced_resume: str
    ats_breakdown: Dict[str, int]
    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)
    feedback: GeminiFeedback

    model_config = {
        "extra": "ignore"
    }
