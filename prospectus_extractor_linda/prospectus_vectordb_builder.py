from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config_prospectus import DEFAULT_EMBEDDING_MODEL
from parse_prospectus import save_prospectus_chunks_to_json
from section_type_refiner import refine_section_types_for_chunks


def build_prospectus_index(
    chunks: List[Dict[str, Any]],
    index_path: str | Path,
    metadata_path: str | Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """
    Construit l'index FAISS à partir des chunks de texte du prospectus.
    """
    model = SentenceTransformer(model_name)

    texts = [c["content"] for c in chunks]
    print(f"[ETAPE 3B] Encodage de {len(texts)} chunks avec {model_name}...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"[ETAPE 3B] Index FAISS sauvegardé dans {index_path}")

    meta = {
        "model_name": model_name,
        "embeddings_shape": embeddings.shape,
        "chunks": chunks,
    }
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[ETAPE 3B] Métadonnées sauvegardées dans {metadata_path}")

    return index, embeddings


def build_prospectus_vectordb_from_docx(
    prospectus_path: str | Path,
    fund_name: str,
    parsed_json_path: str | Path,
    index_path: str | Path,
    metadata_path: str | Path,
    max_chunk_words: int = 450,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    use_llm_for_types: bool = False,
    llm_api_key: Optional[str] = None,
    llm_base_url: str = "https://tokenfactory.esprit.tn/api",
    llm_model: str = "hosted_vllm/Llama-3.1-70B-Instruct",
) -> None:
    """
    Pipeline complet :
    1) parse prospectus.docx -> JSON chunks
    2) raffine les section_type via LLM (optionnel)
    3) construit l'index FAISS + metadata
    """
    chunks = save_prospectus_chunks_to_json(
        prospectus_path=prospectus_path,
        fund_name=fund_name,
        output_json=parsed_json_path,
        max_chunk_words=max_chunk_words,
    )

    chunks = refine_section_types_for_chunks(
        chunks=chunks,
        use_llm=use_llm_for_types,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )

    build_prospectus_index(
        chunks=chunks,
        index_path=index_path,
        metadata_path=metadata_path,
        model_name=model_name,
    )


if __name__ == "__main__":
    import os

    here = Path(__file__).parent

    PROSPECTUS_PATH = here / "prospectus.docx"
    FUND_NAME = "ODDO BHF US Equity Active UCITS ETF"

    PARSED_JSON = here / "outputs" / "parsed" / "prospectus_parsed.json"
    INDEX_PATH = here / "outputs" / "index" / "prospectus_faiss.index"
    META_PATH = here / "outputs" / "index" / "prospectus_metadata.json"

    LLM_API_KEY = os.environ.get("TOKENFACTORY_API_KEY")

    build_prospectus_vectordb_from_docx(
        prospectus_path=PROSPECTUS_PATH,
        fund_name=FUND_NAME,
        parsed_json_path=PARSED_JSON,
        index_path=INDEX_PATH,
        metadata_path=META_PATH,
        max_chunk_words=450,
        model_name=DEFAULT_EMBEDDING_MODEL,
        use_llm_for_types=bool(LLM_API_KEY),
        llm_api_key=LLM_API_KEY,
    )
