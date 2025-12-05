# config_prospectus.py

"""
Configuration et constantes pour l'ÉTAPE 3B : VectorDB_Prospectus.

Ce fichier centralise :
- les patterns qui permettent d'identifier les différentes sections du prospectus
- la config du modèle d'embedding
"""

from __future__ import annotations  # <-- doit être en premier
from typing import Dict, List

# Types de sections canoniques :
# - "objective"              → Objectif de gestion / Investment objective
# - "strategy"               → Stratégie d’investissement / Investment policy
# - "risk_profile"           → Facteurs de risque / SFDR / Risk factors
# - "fees"                   → Frais et commissions / Charges and expenses / TER
# - "distribution_policy"    → Politique de distribution / Dividend policy
# - "investment_restrictions"→ Restrictions d’investissement / Borrowing
# - "esg_policy"             → Politique ESG / Annex SFDR
# - "eligibility"            → Profil investisseur / Éligibilité / Enregistrement
# - "other"                  → Sections non classées ou peu pertinentes

PROSPECTUS_SECTION_PATTERNS: Dict[str, List[str]] = {
    "objective": [
        "investment objective",
        "objective of the fund",
        "objectif de gestion",
    ],
    "strategy": [
        "investment policy",
        "investment objective and policies",
        "investment process",
        "investment strategy",
        "esg investment process",
    ],
    "risk_profile": [
        "risk factors",
        "risks specific to",
        "general risks",
        "sustainability risks",
        "sustainability risk",
        "sfdr",
    ],
    "fees": [
        "charges and expenses",
        "fees and expenses",
        "total expense ratio",
        "ongoing charges",
        "management fees",
    ],
    "distribution_policy": [
        "dividend policy",
        "distribution policy",
        "distribution of income",
    ],
    "investment_restrictions": [
        "investment restrictions",
        "borrowing",
        "leverage",
        "concentration limits",
    ],
    "esg_policy": [
        "esg investment process",
        "sustainable investment objective",
        "environmental and/or social characteristics",
        "sustainability indicators",
        "principal adverse impacts",
        "eu taxonomy",
        "what environmental and/or social characteristics",
    ],
    "eligibility": [
        "profile of a typical investor",
        "description of available shares",
        "registration for public distribution and listing",
        "classification as an equity fund",
    ],
}

# Modèle d'embedding par défaut utilisé pour le VectorDB_Prospectus
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Nombre max de mots par chunk (sections prospectus)
MAX_CHUNK_WORDS = 450
