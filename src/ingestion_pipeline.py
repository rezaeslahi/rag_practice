from pathlib import Path
from src.domain import Chunk
from src.chunker import chunker

def run_ingestion():
    paths = [p for p in (Path.cwd() / "src").rglob("*.py")]
    print(paths)
    for p in paths:
        chunks = chunker(p)
        print(chunks)


if __name__ == "__main__":
    run_ingestion()