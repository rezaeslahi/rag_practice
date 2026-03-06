from typing import ClassVar
from dataclasses import dataclass
from itertools import count

@dataclass
class Chunk():
    _global_id_gen: ClassVar[count] = count(start=0)

    text:str
    doc_ref:str
    id:int

@dataclass
class RetrivedChunk():
    chunk_id:int
    score:float