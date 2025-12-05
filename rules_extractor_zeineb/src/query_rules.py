# src/query_rules.py

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class RulesVectorDB:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[1]
        self.base_dir = base_dir

        self.index_path = base_dir / "vectordb" / "rules_faiss.index"
        self.metadata_path = base_dir / "vectordb" / "rules_metadata.json"

        print(f"📥 Chargement index FAISS: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        print(f"📥 Chargement métadonnées: {self.metadata_path}")
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.rules: List[Dict[str, Any]] = json.load(f)

        self.model = SentenceTransformer(model_name)

    def _encode(self, text: str) -> np.ndarray:
        emb = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return emb.astype("float32")

    def search(
        self,
        query: str,
        top_k: int = 5,
        applicable_to: Optional[str] = None,
        min_severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        - applicable_to: "retail", "professional" ou None
        - min_severity: "low" < "medium" < "high"
        """
        q_emb = self._encode(query)
        scores, indices = self.index.search(q_emb, top_k * 3)  # on sur-échantillonne un peu

        sev_rank = {"low": 0, "medium": 1, "high": 2}

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            rule = self.rules[int(idx)]

            # Filtre applicable_to
            if applicable_to is not None:
                if applicable_to not in rule.get("applicable_to", []):
                    continue

            # Filtre severity
            if min_severity is not None:
                rule_sev = rule.get("severity", "low")
                if sev_rank.get(rule_sev, 0) < sev_rank.get(min_severity, 0):
                    continue

            result = {
                "score": float(score),
                **rule
            }
            results.append(result)

            if len(results) >= top_k:
                break

        return results


def cli_demo():
    """
    Petit mode interactif pour tester ton VectorDB_Rules à la main.
    """
    db = RulesVectorDB()

    print("\n================ RAG RÈGLES – DEMO CLI ================\n")
    print("Exemples de requêtes :")
    print("- règles performances")
    print("- règles sur les clients de détail")
    print("- règles sur l’affichage du risque")
    print("Tape 'quit' pour sortir.\n")

    while True:
        q = input("👤 Question: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break

        results = db.search(q, top_k=5, applicable_to="retail")
        print("\n🔎 Résultats:")
        for r in results:
            print(f"- [{r['score']:.3f}] Règle {r['rule_id']} ({r['severity']})")
            print(f"  Section: {r.get('section')}")
            print(f"  Subsection: {r.get('subsection')}")
            print(f"  applicable_to: {r.get('applicable_to')}")
            print(f"  check_type: {r.get('check_type')}")
            print(f"  triggers: {r.get('triggers')}")
            print(f"  Texte: {r['rule_text'][:200]}...")
            print()

        print("======================================================\n")


if __name__ == "__main__":
    cli_demo()
