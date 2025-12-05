from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config_prospectus import DEFAULT_EMBEDDING_MODEL
from query_utils import (
    rerank_results,
    compute_bm25_scores,
    expand_query,
    add_context_to_results,
)


# ═══════════════════════════════════════════════════════════════
# CHARGEMENT INDEX
# ═══════════════════════════════════════════════════════════════

def load_prospectus_index(
    index_path: str | Path,
    metadata_path: str | Path,
) -> Tuple[faiss.IndexFlatIP, Dict[str, Any]]:
    """Recharge l'index FAISS et les métadonnées des chunks."""
    index = faiss.read_index(str(index_path))
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


# ═══════════════════════════════════════════════════════════════
# QUERY SIMPLE (baseline)
# ═══════════════════════════════════════════════════════════════

def query_prospectus(
    query: str,
    index_path: str | Path,
    metadata_path: str | Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
    section_type_filter: List[str] | None = None,
    min_score_threshold: float = 0.0,
    use_reranking: bool = False,
    include_context: bool = False,
    use_query_expansion: bool = False,
) -> List[Dict[str, Any]]:
    """
    Requête RAG AMÉLIORÉE pour recherche dans le prospectus.

    Args:
        query: Question de l'utilisateur
        section_type_filter: Filtrer par types de section (ex: ["objective", "risk_profile"])
        min_score_threshold: Score minimum pour inclure un résultat (0.0 à 1.0)
        use_reranking: Activer le re-ranking avec cross-encoder
        include_context: Inclure les chunks adjacents dans les résultats
        use_query_expansion: Enrichir la query avec des synonymes

    Returns:
        Liste de résultats triés par pertinence
    """
    # Charger l'index
    index, meta = load_prospectus_index(index_path, metadata_path)
    chunks = meta["chunks"]
    used_model = meta.get("model_name", model_name)

    # Expansion de query (optionnelle)
    original_query = query
    if use_query_expansion and section_type_filter:
        query = expand_query(query, section_type_hint=section_type_filter[0])

    # Encoder la query
    model = SentenceTransformer(used_model)
    query_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    # Chercher plus de résultats si on filtre
    search_k = top_k * 3 if section_type_filter else top_k * 2
    distances, indices = index.search(query_emb, min(search_k, len(chunks)))
    idxs = indices[0]
    scores = distances[0]

    # Construire les résultats
    results: List[Dict[str, Any]] = []
    for rank, (i, score) in enumerate(zip(idxs, scores), start=1):
        if i < 0 or i >= len(chunks):
            continue

        # Filtrer par score minimum
        if score < min_score_threshold:
            continue

        chunk = chunks[i]
        chunk_type = chunk["metadata"].get("section_type")

        # Filtrer par section_type
        if section_type_filter and chunk_type not in section_type_filter:
            continue

        results.append({
            "rank": rank,
            "score": float(score),
            "section_name": chunk["section_name"],
            "section_type": chunk_type,
            "fund_name": chunk["metadata"].get("fund_name"),
            "content": chunk["content"],
            "metadata": chunk["metadata"],
        })

        if len(results) >= top_k:
            break

    # Re-ranking (optionnel)
    if use_reranking and len(results) > 1:
        results = rerank_results(original_query, results)

    # Ajouter le contexte (optionnel)
    if include_context:
        results = add_context_to_results(results, chunks, context_window=1)

    return results


# ═══════════════════════════════════════════════════════════════
# QUERY HYBRIDE (embeddings + BM25)
# ═══════════════════════════════════════════════════════════════

def query_prospectus_hybrid(
    query: str,
    index_path: str | Path,
    metadata_path: str | Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    section_type_filter: List[str] | None = None,
    use_reranking: bool = False,
    include_context: bool = False,
) -> List[Dict[str, Any]]:
    """
    Recherche HYBRIDE combinant:
    - Similarité sémantique (embeddings)
    - Recherche par mots-clés (BM25)

    Args:
        semantic_weight: Poids pour le score sémantique (défaut: 0.7)
        keyword_weight: Poids pour le score BM25 (défaut: 0.3)
    """
    # Charger l'index
    index, meta = load_prospectus_index(index_path, metadata_path)
    chunks = meta["chunks"]
    used_model = meta.get("model_name", model_name)

    # 1. Recherche sémantique
    model = SentenceTransformer(used_model)
    query_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    distances, indices = index.search(query_emb, len(chunks))
    semantic_scores = {int(idx): float(score) for idx, score in zip(indices[0], distances[0])}

    # 2. Recherche BM25
    print(f"[HYBRID] Calcul des scores BM25...")
    bm25_scores = compute_bm25_scores(query, chunks)

    # 3. Combiner les scores
    combined_scores = {}
    for i in range(len(chunks)):
        semantic = semantic_scores.get(i, 0.0)
        keyword = bm25_scores.get(i, 0.0)
        combined_scores[i] = (semantic_weight * semantic) + (keyword_weight * keyword)

    # 4. Filtrer par section_type si nécessaire
    if section_type_filter:
        combined_scores = {
            i: score for i, score in combined_scores.items()
            if chunks[i]["metadata"].get("section_type") in section_type_filter
        }

    # 5. Trier et prendre le top_k
    sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # 6. Construire les résultats
    results = []
    for rank, (i, score) in enumerate(sorted_indices, start=1):
        chunk = chunks[i]
        results.append({
            "rank": rank,
            "score": float(score),
            "semantic_score": semantic_scores.get(i, 0.0),
            "keyword_score": bm25_scores.get(i, 0.0),
            "section_name": chunk["section_name"],
            "section_type": chunk["metadata"].get("section_type"),
            "fund_name": chunk["metadata"].get("fund_name"),
            "content": chunk["content"],
            "metadata": chunk["metadata"],
        })

    # Re-ranking (optionnel)
    if use_reranking and len(results) > 1:
        results = rerank_results(query, results)

    # Contexte (optionnel)
    if include_context:
        results = add_context_to_results(results, chunks)

    return results
