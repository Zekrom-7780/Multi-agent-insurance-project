"""Seed ChromaDB with insurance FAQ data from Hugging Face.

Usage: uv run python -m scripts.seed_chroma
"""

import chromadb
from datasets import load_dataset
import pandas as pd

from app.config import settings


def seed_chroma():
    ds = load_dataset("deccan-ai/insuranceQA-v2")
    df = pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)
    df["combined"] = "Question: " + df["input"] + " \n Answer:  " + df["output"]

    # Sample 500 for manageable size
    df = df.sample(500, random_state=42).reset_index(drop=True)

    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(name=settings.CHROMA_COLLECTION)

    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        collection.add(
            documents=batch["combined"].tolist(),
            metadatas=[
                {"question": q, "answer": a}
                for q, a in zip(batch["input"], batch["output"])
            ],
            ids=batch.index.astype(str).tolist(),
        )
        print(f"  Inserted batch {i // batch_size + 1}/{(len(df) - 1) // batch_size + 1}")

    print(f"ChromaDB seeded: {collection.count()} documents in '{settings.CHROMA_COLLECTION}'")


if __name__ == "__main__":
    seed_chroma()
