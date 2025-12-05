# src/parse_rules_docx.py

import json
import re
from pathlib import Path
from typing import List, Dict, Any

from docx import Document


# =========================
# 1. Pattern pour les sections
# =========================
# Exemples gérés :
# "1 – Règles générales :"
# "4.1 – Mentions spécifiques relatives à l’ESG – Hors fonds professionnels :"
SECTION_CODE_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s*[–\-]\s*(.+)$")


# =========================
# 2. Fonctions d'enrichissement des métadonnées
# =========================

def infer_applicable_to(text: str) -> List[str]:
    text_lower = text.lower()
    targets = set()

    if any(k in text_lower for k in ["client de détail", "client non professionnel", "retail"]):
        targets.add("retail")
    if any(k in text_lower for k in ["client professionnel", "client pro", "professional"]):
        targets.add("professional")

    # Par défaut : règle valable pour les deux
    if not targets:
        targets.update(["retail", "professional"])

    return sorted(list(targets))


def infer_check_type(text: str) -> List[str]:
    text_lower = text.lower()
    types = set()

    # Citation / source
    if any(k in text_lower for k in ["source", "étude", "donnée chiffrée", "statistique", "référence"]):
        types.add("citation")

    # Structure / mise en forme
    if any(k in text_lower for k in ["présentation", "structure", "rubrique", "encadré", "graphique", "tableau"]):
        types.add("structure")

    # Linguistique / wording
    if any(k in text_lower for k in ["trompeur", "équilibré", "clair", "précis", "compréhensible", "promesse"]):
        types.add("linguistic")

    # Par défaut : on considère que c'est une règle linguistique
    if not types:
        types.add("linguistic")

    return sorted(list(types))


def infer_requires_prospectus_check(text: str) -> bool:
    text_lower = text.lower()
    triggers = [
        "prospectus",
        "document d'information clé", "dic", "dici", "kid",
        "sfdr", "article 6", "article 8", "article 9",
        "sri", "indicateur synthétique de risque",
        "profil de risque", "horizon de placement",
    ]
    return any(k in text_lower for k in triggers)


def infer_requires_disclaimer(text: str) -> bool:
    text_lower = text.lower()
    triggers = [
        "performance passée",
        "performances passées",
        "ne préjugent pas",
        "ne constituent pas",
        "recommandation d'investissement",
        "capital n'est pas garanti",
        "risque de perte en capital",
    ]
    return any(k in text_lower for k in triggers)


def infer_severity(text: str) -> str:
    text_lower = text.lower()
    if any(k in text_lower for k in [
        "interdit", "doit", "doivent", "obligatoire",
        "ne doit pas", "ne doivent pas"
    ]):
        return "high"
    if any(k in text_lower for k in ["devrait", "devraient", "recommandé", "il est préférable"]):
        return "medium"
    return "low"


def extract_triggers(text: str) -> List[str]:
    text_lower = text.lower()
    candidates = [
        "performance", "performances", "risque", "risques",
        "sfdr", "esg", "article 8", "article 9",
        "sri", "capital", "prospectus", "disclaimer",
        "simulation", "scénario", "indice", "benchmark", "volatilité",
    ]
    return [k for k in candidates if k in text_lower]


# =========================
# 3. Parsing du document Word (adapté à TON fichier)
# =========================

def parse_rules_from_docx(doc_path: Path) -> List[Dict[str, Any]]:
    document = Document(str(doc_path))

    rules: List[Dict[str, Any]] = []

    current_section_code: str | None = None
    current_section_title: str | None = None
    rule_counter: int = 0

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style = para.style.name

        # 1) Sections : titres "Heading 3" numérotés
        if style == "Heading 3":
            m = SECTION_CODE_PATTERN.match(text)
            if m:
                # Ex: "1 – Règles générales :"
                current_section_code = m.group(1)          # "1" ou "4.1"
                current_section_title = m.group(2).strip(" :")
            else:
                # Ex: premier titre général "Règles relatives aux..."
                current_section_code = None
                current_section_title = text

            rule_counter = 0
            continue

        # 2) Règles : paragraphes Normaux / listes sous la section courante
        if style in ("Normal", "List Paragraph", "Normal (Web)"):
            if current_section_title is None:
                # On ignore les textes avant la première section numérotée
                continue

            rule_counter += 1
            if current_section_code:
                rule_id = f"{current_section_code}.{rule_counter}"
            else:
                # Cas très rare (si pas de code) -> fallback générique
                rule_id = f"R{len(rules) + 1}"

            full_text = text

            rule_obj: Dict[str, Any] = {
                "rule_id": rule_id,
                "section_code": current_section_code,
                "section": current_section_title,
                "rule_text": full_text,
                "applicable_to": infer_applicable_to(full_text),
                "check_type": infer_check_type(full_text),
                "triggers": extract_triggers(full_text),
                "requires_prospectus_check": infer_requires_prospectus_check(full_text),
                "requires_disclaimer": infer_requires_disclaimer(full_text),
                "severity": infer_severity(full_text),
                "source": {
                    "document": doc_path.name,
                    "page": None,  # docx ne donne pas les numéros de pages
                },
            }

            rules.append(rule_obj)

    return rules


def main():
    base_dir = Path(__file__).resolve().parent


    # ⚠️ Vérifie que ce nom de fichier correspond bien à ton fichier dans data/
    doc_path = base_dir / "Synthèse règles présentations commerciales.docx"

    out_path = base_dir / "rules_parsed.json"

    print(f"📄 Lecture du document de règles: {doc_path}")
    rules = parse_rules_from_docx(doc_path)
    print(f"✅ {len(rules)} règles extraites")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    print(f"💾 Fichier sauvegardé: {out_path}")


if __name__ == "__main__":
    main()
