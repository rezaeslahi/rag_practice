from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from src.config import get_config
import numpy as np
from src.domain import Chunk

@dataclass
class Transformer():
    model:SentenceTransformer = SentenceTransformer(get_config().transformer_model_name)

    def transform_chunks(self,chunks:list[Chunk])->np.ndarray:
        texts:list[str] = [chunk.text for chunk in chunks]
        vectors = self.model.encode(sentences=texts,normalize_embeddings=True)
        return vectors
    def transform_question(self,question:str)->np.ndarray:
        vectors = self.model.encode(sentences=[question],normalize_embeddings=True)
        return vectors
