from typing import Optional, List
import numpy as np
from pydantic import BaseModel, ConfigDict

class OneSample(BaseModel):
    """Data structure for a single data sample."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    image: np.ndarray
    question: str
    answers: List[str]
    metadata: Optional[dict] = None
    