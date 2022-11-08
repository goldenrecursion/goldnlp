from typing import Dict, List, Optional

from pydantic import BaseModel

class SimpleEntity(BaseModel):
    type: str
    start_char: int
    end_char: int
    text: str

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    text: str
    entities: List[SimpleEntity]