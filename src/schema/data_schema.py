from typing import Optional, List
import numpy as np
from pydantic import BaseModel, ConfigDict

class OneSample(BaseModel):
    """Data structure for a single data sample."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    image_path: str  # Store path instead of loaded image for memory efficiency
    question: str
    answers: List[str]
    metadata: Optional[dict] = None
    