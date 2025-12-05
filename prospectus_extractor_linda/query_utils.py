from __future__ import annotations

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import re


# ═══════════════════════════════════════════════════════════════
# RERANKING
# ═══════════════════════════════════════════════════════════════

def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> List[Dict[str, Any]]:
    """
    Re-rank les résultats avec un modèle cross-encoder.
    """
    if not results:
        return results

    print(f"[RERANK] Re-ranking de {len(results)} résultats avec {model_name}...")
    cross_encoder = CrossEncoder(model_name)

    pairs = [(query, r["content"]) for r in results]
    rerank_scores = cross_encoder.predict(pairs)

    for i, result in enumerate(results):
        result["rerank_score"] = float(rerank_scores[i])
        result["original_score"] = result["score"]
        result["score"] = result["rerank_score"]

    results.sort(key=lambda x: x["rerank_score"], reverse=True)

    for i, result in enumerate(results, start=1):
        result["rank"] = i

    return results


# ═══════════════════════════════════════════════════════════════
# RECHERCHE HYBRIDE (BM25)
# ═══════════════════════════════════════════════════════════════

def preprocess_text_for_bm25(text: str) -> List[str]:
    """
    Tokenize et nettoie le texte pour BM25.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def compute_bm25_scores(
    query: str,
    chunks: List[Dict[str, Any]]
) -> Dict[int, float]:
    """
    Calcule les scores BM25 pour tous les chunks.
    """
    corpus = [preprocess_text_for_bm25(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus)

    tokenized_query = preprocess_text_for_bm25(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    bm25_scores_norm = {i: score / max_bm25 for i, score in enumerate(bm25_scores)}

    return bm25_scores_norm


# ═══════════════════════════════════════════════════════════════
# EXPANSION DE QUERY
# ═══════════════════════════════════════════════════════════════

QUERY_EXPANSIONS = {
    "objective": {
        "investment objective": ["investment goal", "fund objective", "target", "aim"],
        "performance": ["return", "gains", "yield", "growth"],
        "benchmark": ["reference index", "comparison index"],
    },
    "risk_profile": {
        "risk": ["risk factor", "danger", "volatility", "uncertainty", "exposure"],
        "loss": ["decline", "drawdown", "depreciation", "negative return"],
    },
    "fees": {
        "fee": ["charge", "cost", "expense", "payment"],
        "management fee": ["advisory fee", "investment management charge"],
        "TER": ["total expense ratio", "ongoing charges"],
    },
    "distribution_policy": {
        "dividend": ["distribution", "income", "payout"],
        "accumulation": ["reinvestment", "capitalization"],
    },
    "strategy": {
        "investment strategy": ["investment approach", "portfolio construction"],
        "allocation": ["exposure", "weighting", "positioning"],
    },
    "esg_policy": {
        "ESG": ["environmental social governance", "sustainability"],
        "exclusion": ["screening", "negative screening"],
    },
    "eligibility": {
        "investor": ["shareholder", "subscriber"],
        "share class": ["unit class", "class of shares"],
    },
}


def expand_query(query: str, section_type_hint: str | None = None) -> str:
    """
    Enrichit la query avec des termes connexes pour améliorer le rappel.
    """
    query_lower = query.lower()
    expanded_terms = [query]

    if section_type_hint and section_type_hint in QUERY_EXPANSIONS:
        for term, synonyms in QUERY_EXPANSIONS[section_type_hint].items():
            if term in query_lower:
                expanded_terms.extend(synonyms[:2])  # Limiter à 2 synonymes

    expanded_query = " ".join(expanded_terms)
    if expanded_query != query:
        print(f"[QUERY EXPANSION] '{query}' → '{expanded_query}'")

    return expanded_query


# ═══════════════════════════════════════════════════════════════
# CONTEXTE (chunks adjacents)
# ═══════════════════════════════════════════════════════════════

def add_context_to_results(
    results: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]],
    context_window: int = 1
) -> List[Dict[str, Any]]:
    """
    Ajoute le contexte avant/après chaque résultat (chunks adjacents).
    """
    for result in results:
        chunk_idx = None
        for i, chunk in enumerate(all_chunks):
            if (chunk["section_name"] == result["section_name"] and
                chunk["content"] == result["content"]):
                chunk_idx = i
                break

        if chunk_idx is None:
            continue

        context_before = []
        for i in range(max(0, chunk_idx - context_window), chunk_idx):
            prev_chunk = all_chunks[i]
            context_before.append({
                "section_name": prev_chunk["section_name"],
                "section_type": prev_chunk["metadata"].get("section_type"),
                "content": (
                    prev_chunk["content"][:300] + "..."
                    if len(prev_chunk["content"]) > 300
                    else prev_chunk["content"]
                ),
            })
        if context_before:
            result["context_before"] = context_before

        context_after = []
        for i in range(chunk_idx + 1, min(len(all_chunks), chunk_idx + context_window + 1)):
            next_chunk = all_chunks[i]
            context_after.append({
                "section_name": next_chunk["section_name"],
                "section_type": next_chunk["metadata"].get("section_type"),
                "content": (
                    next_chunk["content"][:300] + "..."
                    if len(next_chunk["content"]) > 300
                    else next_chunk["content"]
                ),
            })
        if context_after:
            result["context_after"] = context_after

    return results


# ═══════════════════════════════════════════════════════════════
# FILTRAGE INTELLIGENT
# ═══════════════════════════════════════════════════════════════

def filter_results_by_relevance(
    results: List[Dict[str, Any]],
    min_score: float = 0.3,
    remove_duplicates: bool = True
) -> List[Dict[str, Any]]:
    """
    Filtre les résultats selon différents critères de qualité.
    """
    filtered = [r for r in results if r["score"] >= min_score]
    
    if not remove_duplicates:
        return filtered
    
    seen_sections = set()
    unique_results = []
    
    for r in filtered:
        section_id = f"{r['section_name']}_{r['section_type']}"
        if section_id not in seen_sections:
            seen_sections.add(section_id)
            unique_results.append(r)
    
    return unique_results
