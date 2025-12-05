from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

import httpx
from openai import OpenAI

from config_prospectus import PROSPECTUS_SECTION_PATTERNS

# Types canoniques autorisés pour section_type
CANONICAL_SECTION_TYPES = sorted(list(PROSPECTUS_SECTION_PATTERNS.keys()) + ["other"])


def _build_client(api_key: str, base_url: str) -> OpenAI:
    """
    Construit un client OpenAI pointant vers TokenFactory.
    La vérification TLS est désactivée comme indiqué dans le mail.
    """
    http_client = httpx.Client(verify=False)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
    )
    return client


def _llm_classify_section(
    client: OpenAI,
    model: str,
    section_name: str,
    content: str,
    original_title: str = "",
) -> Optional[str]:
    """
    Appelle le LLM TokenFactory pour proposer un section_type corrigé.

    Retourne un string parmi CANONICAL_SECTION_TYPES ou None en cas d'erreur.
    """
    allowed_types_str = ", ".join(CANONICAL_SECTION_TYPES)

    system_prompt = (
        "Tu es un classifieur de sections de prospectus UCITS.\n"
        "Tu reçois le titre d'une section et son contenu brut (texte + tableaux flatten).\n"
        "Ton rôle est de déterminer à QUEL TYPE CANONIQUE appartient cette section selon les normes UCITS.\n"
        "Ton choix doit refléter EXACTEMENT le sens principal de la section.\n"
        "Si plusieurs types semblent possibles, tu dois appliquer les règles de priorité décrites ci-dessous.\n"
        "Tu ne dois JAMAIS inventer un type qui n'existe pas dans la liste, ni renvoyer autre chose qu'un JSON.\n"
        "Tu ne dois utiliser 'other' que si aucun type défini ne correspond réellement au contenu.\n\n"
        
        "Types canoniques disponibles:\n"
        "- objective\n"
        "- strategy\n"
        "- risk_profile\n"
        "- fees\n"
        "- distribution_policy\n"
        "- investment_restrictions\n"
        "- esg_policy\n"
        "- eligibility\n"
        "- fund_structure\n"
        "- legal_information\n"
        "- operational_information\n"
        "- reporting_policy\n"
        "- subscription_information\n"
        "- tax_information\n"
        "- fund_governance\n"
        "- documentation_reference\n"
        "- other\n\n"
        
        "DÉFINITIONS DÉTAILLÉES DES TYPES :\n"
        "- objective: objectifs d'investissement, but du fonds, capital growth, performance target, benchmark utilisé "
        "pour objectif ou comparaison, horizon d'investissement, nature fondamentale du fonds.\n\n"
        "- strategy: politique d'investissement, universe d'investissement, screening, méthodes quantitatives, "
        "construction du portefeuille, utilisation de dérivés, calcul de levier, risk budgeting, reference benchmark "
        "quand il sert à guider la gestion.\n\n"
        "- risk_profile: facteurs de risque du fonds, market risk, liquidity risk, concentration, counterparty, "
        "leverage, ESG sustainability risks, stress scenarios.\n\n"
        "- fees: tous les coûts, TER, ongoing charges, performance fees, redemption fees, cash transaction fees, "
        "custody fees.\n\n"
        "- distribution_policy: règles de distribution des revenus, dividendes, accumulation/distribution, fréquence.\n\n"
        "- investment_restrictions: limites UCITS, eligibilité actifs, 10% limit, exposure limits, borrowing conditions.\n\n"
        "- esg_policy: SFDR Annex, caractéristiques environnementales/sociales, exclusions ESG, PAI, ratings ESG, "
        "couverture, sustainable investment objective, green taxonomy, méthodologie d'analyse ESG.\n\n"
        "- eligibility: profil d'investisseur, retail/professional, conditions d'accès, share classes disponibles, "
        "hedged/unhedged, currency eligibility, juridictions autorisées, registration for public distribution.\n\n"
        "- fund_structure: structure du fonds, Investment Manager, Management Company, Depositary, délégation, "
        "service providers, contractual roles opérationnels.\n\n"
        "- legal_information: cadre réglementaire, disclaimers légaux, selling restrictions, mentions MiFID, "
        "compliance regulatory frameworks.\n\n"
        "- operational_information: définitions et processus opérationnels, NAV calculation method, valuation rules, "
        "calendrier opérationnel, Business Days, cut-off times, dealing deadlines, settlement cycles, publication schedules.\n\n"
        "- reporting_policy: transparence du portefeuille, divulgation holdings, fréquence de reporting NAV, "
        "publication des documents sur des sites officiels.\n\n"
        "- subscription_information: procédures de souscription/rachat, listing market, secondary market trading, "
        "market makers, authorised participants, creation/redemption mechanism, dealing conditions, "
        "Initial Offer Period, Dealing Day definitions.\n\n"
        "- tax_information: classification fiscale, withholding tax, tax regime, conditions fiscales spécifiques par pays.\n\n"
        "- fund_governance: gouvernance interne du fonds, responsabilités du Management, structure de supervision, "
        "Investment Management Agreement.\n\n"
        "- documentation_reference: renvois explicites à d'autres documents (Prospectus, Base Prospectus, Directives externes).\n\n"
        "- other: uniquement si AUCUNE autre catégorie ne s'applique. Si un élément correspond même partiellement "
        "à l'une des catégories définies, tu dois choisir cette catégorie plutôt que 'other'.\n\n"

        "RÈGLES DE PRIORITÉ ET D'ARBITRAGE:\n"
        "1) Toujours choisir le type le PLUS SPÉCIFIQUE correspondant au contenu dominant.\n"
        "2) Si une section est dans l'annexe SFDR mais décrit principalement la gestion (construction du portefeuille, "
        "allocation, dérivés), alors 'strategy' est prioritaire sur 'esg_policy'.\n"
        "3) Si une section décrit l'accès à certaines parts, l'éligibilité, la distribution dans un pays, "
        "les share classes, ou la devise des classes, alors 'eligibility' est prioritaire.\n"
        "4) Si une section présente structure, délégation, contrats, roles opérationnels, alors 'fund_structure' "
        "ou 'fund_governance' selon si le texte est factuel (structure) ou juridico-contractuel (governance).\n"
        "5) Selling restrictions, disclaimers légaux, distribution par juridiction → 'legal_information'.\n"
        "6) Calendrier, cut-off times, settlement cycles, valuation methodology, NAV calculation → 'operational_information'.\n"
        "7) Transparence holdings, fréquence de reporting, site web de publication → 'reporting_policy'.\n"
        "8) Listing sur marché secondaire, mécanismes creation/redemption, Dealing Day, Initial Offer Period, "
        "market makers → 'subscription_information'.\n"
        "9) Fiscalité → 'tax_information'.\n"
        "10) Si le texte est intégralement centré sur ESG, exclusions, modèles ESG, indicateurs, ratings, coverage, "
        "alors 'esg_policy'.\n"
        "11) Si le texte décrit des objectifs purs du fonds → 'objective'.\n"
        "12) 'other' n'est valable QUE lorsqu'aucune règle ci-dessus n'est applicable.\n\n"

        "MOTS-CLÉS INDICATIFS (non exclusifs):\n"
        "- Si tu vois: 'Dealing Day', 'Dealing Deadline', 'Initial Offer Period', 'Creation Unit', "
        "alors cela correspond à SUBSCRIPTION INFORMATION ou OPERATIONAL INFORMATION, "
        "mais JAMAIS 'other'.\n"
        "- 'NAV calculation', 'valuation method', 'cut-off time', 'publication schedule' → OPERATIONAL_INFORMATION.\n"
        "- 'Business Day' → OPERATIONAL_INFORMATION.\n"
        "- 'ETF Classes', 'Non-ETF Classes', 'Base Currency', 'Minimum Initial Investment' → ELIGIBILITY.\n\n"
        
        "FORMAT STRICT DE LA RÉPONSE:\n"
        "Tu dois répondre STRICTEMENT en JSON, au format suivant:\n"
        '{"section_type": "objective"}\n'
        "en remplaçant 'objective' par le type identifié.\n"
        "Pas de texte supplémentaire, pas de justification, pas de commentaires.\n"
    )

    user_content = (
        f"Original title: {original_title}\n"
        f"Section name: {section_name}\n\n"
        f"Full content:\n{content}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)

        section_type = str(data.get("section_type", "")).strip().lower()
        if section_type in CANONICAL_SECTION_TYPES:
            return section_type
        else:
            # Si le modèle sort un type inconnu, on tombe sur "other"
            return "other"

    except Exception as e:
        print(f"[ETAPE 3B][LLM] Erreur pendant la classification: {e}")
        return None


def refine_section_types_for_chunks(
    chunks: List[Dict[str, Any]],
    use_llm: bool = False,
    llm_api_key: Optional[str] = None,
    llm_base_url: str = "https://tokenfactory.esprit.tn/api",
    llm_model: str = "hosted_vllm/Llama-3.1-70B-Instruct",
) -> List[Dict[str, Any]]:
    """
    Parcourt la liste des chunks et, si use_llm=True, utilise le LLM pour
    vérifier/corriger le champ metadata['section_type'].

    Si use_llm=False ou pas de clé → on retourne les chunks tels quels.
    """
    if not use_llm:
        print("[ETAPE 3B][LLM] Raffinement des section_type désactivé (use_llm=False).")
        return chunks

    if not llm_api_key:
        print("[ETAPE 3B][LLM] Aucune clé API fournie → raffinement désactivé.")
        return chunks

    client = _build_client(api_key=llm_api_key, base_url=llm_base_url)

    refined_chunks: List[Dict[str, Any]] = []

    print(f"[ETAPE 3B][LLM] Raffinement des section_type pour {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {}) or {}

        original_type = meta.get("section_type", "other")
        section_name = chunk.get("section_name", "")
        content = chunk.get("content", "")
        original_title = meta.get("original_title") or chunk.get("original_title", "")

        suggested_type = _llm_classify_section(
            client=client,
            model=llm_model,
            section_name=section_name,
            content=content,
            original_title=original_title,
        )

        if suggested_type is None:
            # En cas d'erreur LLM, on garde le type original
            final_type = original_type
            print(f"[ETAPE 3B][LLM] Chunk {i}: erreur LLM → section_type conservé = {original_type}")
        else:
            final_type = suggested_type
            if final_type != original_type:
                print(
                    f"[ETAPE 3B][LLM] Chunk {i}: section_type corrigé "
                    f"'{original_type}' → '{final_type}' "
                    f"({section_name})"
                )

        # Mise à jour du metadata
        meta["section_type"] = final_type
        chunk["metadata"] = meta
        refined_chunks.append(chunk)

    print("[ETAPE 3B][LLM] Raffinement des section_type terminé.")
    return refined_chunks
