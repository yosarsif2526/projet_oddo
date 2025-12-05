from __future__ import annotations

import argparse
import os
from pathlib import Path

from config_prospectus import DEFAULT_EMBEDDING_MODEL
from prospectus_vectordb_builder import build_prospectus_vectordb_from_docx
from query_prospectus import query_prospectus, query_prospectus_hybrid


def get_paths():
    """
    Calcule tous les chemins utiles en partant du dossier courant
    (là où se trouve main.py).
    """
    here = Path(__file__).parent

    prospectus_path = here / "prospectus.docx"
    outputs_dir = here / "outputs"
    parsed_json_path = outputs_dir / "parsed" / "prospectus_parsed.json"
    index_path = outputs_dir / "index" / "prospectus_faiss.index"
    metadata_path = outputs_dir / "index" / "prospectus_metadata.json"

    return {
        "here": here,
        "prospectus_path": prospectus_path,
        "parsed_json_path": parsed_json_path,
        "index_path": index_path,
        "metadata_path": metadata_path,
    }


def cmd_build(fund_name: str, use_llm_types: bool = False):
    """
    Commande 'build' :
    1) parse prospectus.docx -> prospectus_parsed.json
    2) (optionnel) raffine les section_type via LLM (TokenFactory)
    3) construit la VectorDB (index FAISS + metadata)
    """
    paths = get_paths()
    prospectus_path = paths["prospectus_path"]
    parsed_json_path = paths["parsed_json_path"]
    index_path = paths["index_path"]
    metadata_path = paths["metadata_path"]

    if not prospectus_path.exists():
        raise FileNotFoundError(
            f"❌ Fichier prospectus introuvable : {prospectus_path} "
            f"(assure-toi que 'prospectus.docx' est dans le même dossier que main.py)"
        )

    print("============================================================")
    print("ÉTAPE 3B – BUILD VectorDB_Prospectus")
    print("============================================================")
    print(f"📄 Prospectus : {prospectus_path}")
    print(f"🏷️  Fund name : {fund_name}")
    print("------------------------------------------------------------")

    # 🔑 Clé API TokenFactory (à mettre dans la variable d'environnement TOKENFACTORY_API_KEY)
    llm_api_key = os.environ.get("TOKENFACTORY_API_KEY")

    # Si tu demandes --use-llm-types mais pas de clé → on désactive proprement
    if use_llm_types and not llm_api_key:
        print("⚠️ --use-llm-types est activé mais aucune clé TOKENFACTORY_API_KEY n'est définie.")
        print("   → Le raffinement des section_type par LLM sera désactivé pour ce run.\n")
        use_llm_types = False

    # 🧠 Pipeline complet : parse -> (LLM refine) -> index FAISS
    build_prospectus_vectordb_from_docx(
        prospectus_path=prospectus_path,
        fund_name=fund_name,
        parsed_json_path=parsed_json_path,
        index_path=index_path,
        metadata_path=metadata_path,
        max_chunk_words=450,
        model_name=DEFAULT_EMBEDDING_MODEL,
        use_llm_for_types=use_llm_types,
        llm_api_key=llm_api_key,
    )

    print("\n✅ BUILD terminé : VectorDB_Prospectus prête à être utilisée.")
    print(f"   - JSON chunks : {parsed_json_path}")
    print(f"   - Index FAISS : {index_path}")
    print(f"   - Metadata    : {metadata_path}")
    print("============================================================\n")


def cmd_query(
    query: str,
    top_k: int = 3,
    section_filter: str | None = None,
    use_hybrid: bool = False,
    use_reranking: bool = False,
    include_context: bool = False,
    use_expansion: bool = False,
    min_score: float = 0.0,
):
    """
    Commande 'query' AMÉLIORÉE :
    Interroge la VectorDB_Prospectus avec plusieurs options avancées.
    """
    paths = get_paths()
    index_path = paths["index_path"]
    metadata_path = paths["metadata_path"]

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "❌ Index ou métadonnées introuvables.\n"
            "   → Lance d'abord :  python main.py build"
        )

    print("============================================================")
    print("ÉTAPE 3B – QUERY VectorDB_Prospectus")
    print("============================================================")
    print(f"❓ Query : {query}")
    print(f"🔧 Mode : {'HYBRID' if use_hybrid else 'SEMANTIC'}")
    if section_filter:
        print(f"🏷️  Section filter : {section_filter}")
    if use_reranking:
        print(f"🎯 Re-ranking : ACTIVÉ")
    if include_context:
        print(f"📖 Context : ACTIVÉ")
    if use_expansion:
        print(f"📝 Query expansion : ACTIVÉ")
    if min_score > 0:
        print(f"⚖️  Min score : {min_score}")
    print("------------------------------------------------------------")

    # Convertir section_filter en liste
    section_type_filter = None
    if section_filter:
        section_type_filter = [s.strip() for s in section_filter.split(",")]

    # Choisir entre recherche hybride ou sémantique
    if use_hybrid:
        results = query_prospectus_hybrid(
            query=query,
            index_path=index_path,
            metadata_path=metadata_path,
            model_name=DEFAULT_EMBEDDING_MODEL,
            top_k=top_k,
            semantic_weight=0.7,
            keyword_weight=0.3,
            section_type_filter=section_type_filter,
            use_reranking=use_reranking,
            include_context=include_context,
        )
    else:
        results = query_prospectus(
            query=query,
            index_path=index_path,
            metadata_path=metadata_path,
            model_name=DEFAULT_EMBEDDING_MODEL,
            top_k=top_k,
            section_type_filter=section_type_filter,
            min_score_threshold=min_score,
            use_reranking=use_reranking,
            include_context=include_context,
            use_query_expansion=use_expansion,
        )

    if not results:
        print("⚠️ Aucun résultat pertinent trouvé.")
        return

    print(f"\n📊 {len(results)} résultat(s) trouvé(s) :\n")

    for r in results:
        # Affichage du résultat principal
        score_info = f"score={r['score']:.3f}"
        
        # Si hybride, afficher les scores détaillés
        if use_hybrid and "semantic_score" in r:
            score_info = (
                f"score={r['score']:.3f} "
                f"(sem={r['semantic_score']:.3f}, kw={r['keyword_score']:.3f})"
            )
        
        # Si reranking, afficher les deux scores
        if use_reranking and "original_score" in r:
            score_info += f" [original={r['original_score']:.3f}]"

        print(f"[{r['rank']}] {score_info}")
        print(f"    📂 Section : {r['section_name']}")
        print(f"    🏷️  Type    : {r['section_type']}")
        print("-" * 60)
        
        # Contexte avant (si demandé)
        if include_context and "context_before" in r:
            print("    📖 Context BEFORE:")
            for ctx in r["context_before"]:
                print(f"       └─ {ctx['section_name']}")
                print(f"          {ctx['content'][:150]}...")
            print()
        
        # Contenu principal
        print(f"    💬 Content :")
        print(f"       {r['content'][:500]}")
        if len(r['content']) > 500:
            print("       ...")
        print()
        
        # Contexte après (si demandé)
        if include_context and "context_after" in r:
            print("    📖 Context AFTER:")
            for ctx in r["context_after"]:
                print(f"       └─ {ctx['section_name']}")
                print(f"          {ctx['content'][:150]}...")
            print()
        
        print("=" * 60)
        print()


def cmd_compare(query: str, top_k: int = 3):
    """
    Commande 'compare' :
    Compare les résultats entre recherche sémantique, hybride, et avec reranking.
    """
    paths = get_paths()
    index_path = paths["index_path"]
    metadata_path = paths["metadata_path"]

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "❌ Index ou métadonnées introuvables.\n"
            "   → Lance d'abord :  python main.py build"
        )

    print("============================================================")
    print("ÉTAPE 3B – COMPARAISON DES MÉTHODES DE RECHERCHE")
    print("============================================================")
    print(f"❓ Query : {query}")
    print(f"📊 Top-K : {top_k}")
    print("============================================================\n")

    # 1. Recherche sémantique simple
    print("🔍 MÉTHODE 1 : Recherche sémantique (baseline)")
    print("-" * 60)
    res_semantic = query_prospectus(
        query=query,
        index_path=index_path,
        metadata_path=metadata_path,
        top_k=top_k,
    )
    for r in res_semantic:
        print(f"[{r['rank']}] score={r['score']:.3f} - {r['section_name']}")
    print()

    # 2. Recherche hybride
    print("🔍 MÉTHODE 2 : Recherche hybride (sémantique + BM25)")
    print("-" * 60)
    res_hybrid = query_prospectus_hybrid(
        query=query,
        index_path=index_path,
        metadata_path=metadata_path,
        top_k=top_k,
        semantic_weight=0.7,
        keyword_weight=0.3,
    )
    for r in res_hybrid:
        print(
            f"[{r['rank']}] score={r['score']:.3f} "
            f"(sem={r['semantic_score']:.3f}, kw={r['keyword_score']:.3f}) "
            f"- {r['section_name']}"
        )
    print()

    # 3. Recherche avec reranking
    print("🔍 MÉTHODE 3 : Recherche hybride + reranking")
    print("-" * 60)
    res_reranked = query_prospectus_hybrid(
        query=query,
        index_path=index_path,
        metadata_path=metadata_path,
        top_k=top_k,
        semantic_weight=0.7,
        keyword_weight=0.3,
        use_reranking=True,
    )
    for r in res_reranked:
        print(
            f"[{r['rank']}] rerank_score={r['score']:.3f} "
            f"[original={r['original_score']:.3f}] "
            f"- {r['section_name']}"
        )
    print()

    print("=" * 60)
    print("✅ Comparaison terminée")
    print("=" * 60)


def cmd_all(fund_name: str, use_llm_types: bool = False):
    """
    Commande 'all' :
    1) build (parse + LLM optionnel + index)
    2) petite requête de test sur l'objectif d'investissement
    """
    cmd_build(fund_name=fund_name, use_llm_types=use_llm_types)

    # Query de test : objectif de gestion
    test_query = "What is the investment objective of the fund?"
    print("\n" + "=" * 60)
    print("🧪 TEST QUERY (recherche sémantique)")
    print("=" * 60)
    cmd_query(test_query, top_k=3)


def main():
    parser = argparse.ArgumentParser(
        description="ÉTAPE 3B – Pipeline VectorDB_Prospectus (parse + index + query)"
    )

    parser.add_argument(
        "command",
        choices=["build", "query", "compare", "all"],
        help="Action à exécuter : "
             "'build' (parse + index), "
             "'query' (interroger avec options avancées), "
             "'compare' (comparer les méthodes), "
             "'all' (build + test query).",
    )

    parser.add_argument(
        "--fund-name",
        type=str,
        default="ODDO BHF US Equity Active UCITS ETF",
        help="Nom du fonds (utilisé dans les métadonnées des chunks).",
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Question à poser au prospectus (pour les commandes 'query' et 'compare').",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de résultats à retourner pour une query.",
    )

    parser.add_argument(
        "--use-llm-types",
        action="store_true",
        help="Si présent, utilise le LLM (TokenFactory) pour vérifier/corriger "
             "les section_type des chunks.",
    )

    # ═══════════════════════════════════════════════════════════════
    # NOUVELLES OPTIONS POUR LA RECHERCHE
    # ═══════════════════════════════════════════════════════════════

    parser.add_argument(
        "--section-filter",
        type=str,
        help="Filtrer par section_type (ex: 'objective,strategy' ou 'risk_profile').",
    )

    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Utiliser la recherche hybride (sémantique + BM25).",
    )

    parser.add_argument(
        "--reranking",
        action="store_true",
        help="Activer le re-ranking avec cross-encoder (améliore la précision).",
    )

    parser.add_argument(
        "--context",
        action="store_true",
        help="Inclure les chunks adjacents (contexte avant/après).",
    )

    parser.add_argument(
        "--expansion",
        action="store_true",
        help="Activer l'expansion de query avec synonymes.",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Score minimum pour inclure un résultat (0.0 à 1.0).",
    )

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(fund_name=args.fund_name, use_llm_types=args.use_llm_types)

    elif args.command == "query":
        if not args.query:
            raise ValueError(
                "❌ Pour la commande 'query', tu dois fournir --query \"...\""
            )
        cmd_query(
            query=args.query,
            top_k=args.top_k,
            section_filter=args.section_filter,
            use_hybrid=args.hybrid,
            use_reranking=args.reranking,
            include_context=args.context,
            use_expansion=args.expansion,
            min_score=args.min_score,
        )

    elif args.command == "compare":
        if not args.query:
            raise ValueError(
                "❌ Pour la commande 'compare', tu dois fournir --query \"...\""
            )
        cmd_compare(query=args.query, top_k=args.top_k)

    elif args.command == "all":
        cmd_all(fund_name=args.fund_name, use_llm_types=args.use_llm_types)


if __name__ == "__main__":
    main()
