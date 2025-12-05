# src/build_rules_index.py

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_rules(rules_path: Path) -> List[Dict[str, Any]]:
    with rules_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_embeddings(
    rules: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [r["rule_text"] for r in rules]
    print(f"🧠 Encodage de {len(texts)} règles avec {model_name}...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # similarité cosinus si vecteurs normalisés
    index.add(embeddings.astype("float32"))
    return index


def main():
    base_dir = Path(__file__).resolve().parents[1]
    rules_path = base_dir / "data" / "rules_parsed.json"
    vectordb_dir = base_dir / "vectordb"
    vectordb_dir.mkdir(parents=True, exist_ok=True)

    rules = load_rules(rules_path)
    embeddings = build_embeddings(rules)

    index = build_faiss_index(embeddings)

    # Sauvegarde index FAISS
    faiss_path = vectordb_dir / "rules_faiss.index"
    faiss.write_index(index, str(faiss_path))
    print(f"💾 Index FAISS sauvegardé: {faiss_path}")

    # Sauvegarde métadonnées
    metadata_path = vectordb_dir / "rules_metadata.json"
    # On enlève les champs lourds si besoin, mais là c’est OK
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    print(f"💾 Métadonnées sauvegardées: {metadata_path}")


if __name__ == "__main__":
    main()
