from dataclasses import dataclass

@dataclass
class Config():
    transformer_model_name:str
    pass
_config:Config|None = None

def get_config()->Config:
    global _config
    if not _config:
        _config = Config(
            transformer_model_name = "BAAI/bge-m3"
        )
    return _config