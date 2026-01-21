# Generated from: 2_31.ipynb
# Converted at: 2025-12-16T23:13:46.037Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# =============================================================================
# SYSTÈME COMPLET DE DÉTECTION - VERSION AMÉLIORÉE AVEC LLM
# =============================================================================

import json
import re
import pandas as pd
import httpx
from openai import OpenAI
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher

# ==========================================
# CONFIGURATION LLM
# ==========================================

def get_llm_client():
    """Initialise le client LLM"""
    http_client = httpx.Client(verify=False)
    return OpenAI(
        api_key="sk-7c0b80cf494746f580cc5ba555d739b2",
        base_url="https://tokenfactory.esprit.tn/api",
        http_client=http_client
    )

# ==========================================
# STRUCTURES DE DONNÉES (INCHANGÉ)
# ==========================================

@dataclass
class Violation:
    """Représente une violation détectée"""
    rule_id: str
    violation_type: str
    severity: str
    slides_involved: List[int]
    explanation: str
    violating_content: str
    correction: str
    confidence: float

@dataclass
class GlobalAnalysisResult:
    """Résultat de l'analyse globale"""
    document_name: str
    analysis_date: str
    violations: List[Violation] = field(default_factory=list)
    recommendations: List[Violation] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

# ==========================================
# DÉTECTEUR DE LANGUE AMÉLIORÉ
# ==========================================

class LanguageDetector:
    """Détecte la langue principale du document"""

    def __init__(self):
        self.en_words = ['the', 'is', 'and', 'to', 'of', 'a', 'in', 'that', 'have', 'for']
        self.fr_words = ['le', 'de', 'et', 'la', 'est', 'un', 'une', 'pour', 'dans', 'que']

    def detect(self, slides: List[Dict]) -> str:
        """Détecte la langue en analysant les slides"""
        all_text = " ".join([s.get('content', '') for s in slides[:5]]).lower()

        en_count = sum(all_text.count(w) for w in self.en_words)
        fr_count = sum(all_text.count(w) for w in self.fr_words)

        # Ratio EN/FR
        if en_count > fr_count * 1.5:
            return 'EN'
        elif fr_count > en_count * 1.5:
            return 'FR'
        else:
            # Détection par mots-clés critiques
            if 'promotional information' in all_text or 'investment strategy' in all_text:
                return 'EN'
            elif 'document promotionnel' in all_text or 'stratégie d\'investissement' in all_text:
                return 'FR'
            return 'EN'  # Par défaut

# ==========================================
# ANALYSEUR DE NEUTRALITÉ AVEC LLM
# ==========================================

class LLMNeutralityAnalyzer:
    """Analyseur de neutralité et cohérence avec LLM"""

    def __init__(self, client: OpenAI, language: str):
        self.client = client
        self.language = language

    def analyze_slide_neutrality(self, slide_content: str, slide_num: int) -> Optional[Violation]:
        """Analyse la neutralité d'une slide avec LLM"""

        prompt = f"""Tu es un expert en conformité réglementaire pour documents financiers.

Analyse le contenu suivant d'une slide de présentation marketing et détecte:
1. Langage promotionnel excessif (superlatifs, promesses de rendement)
2. Affirmations non atténuées (sans "objectif", "vise à", "potentiel")
3. Garanties implicites ou explicites

SLIDE {slide_num}:
\"\"\"
{slide_content[:1000]}
\"\"\"

Réponds UNIQUEMENT en JSON avec:
{{
  "is_violation": true/false,
  "severity": "high"/"medium"/"low",
  "violations_found": ["liste des phrases problématiques"],
  "explanation": "explication courte",
  "confidence": 0.0-1.0
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="hosted_vllm/Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system", "content": "Tu es un expert en conformité financière. Réponds UNIQUEMENT en JSON valide."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            result = json.loads(response.choices[0].message.content)

            if result.get('is_violation', False):
                return Violation(
                    rule_id="1.7",
                    violation_type="NEUTRALITÉ_DISCOURS_LLM",
                    severity=result.get('severity', 'medium'),
                    slides_involved=[slide_num],
                    explanation=result.get('explanation', 'Langage non neutre détecté'),
                    violating_content="; ".join(result.get('violations_found', [])),
                    correction="Reformuler avec un langage neutre et objectif (ex: 'vise à', 'a pour objectif')",
                    confidence=result.get('confidence', 0.8)
                )
        except Exception as e:
            print(f"⚠️ Erreur LLM Neutralité Slide {slide_num}: {e}")

        return None

    def check_slide_coherence(self, slide1: Dict, slide2: Dict) -> Optional[Violation]:
        """Vérifie la cohérence entre deux slides consécutives"""

        if not slide1 or not slide2:
            return None

        s1_num = slide1.get('metadata', {}).get('slide_number', 0)
        s2_num = slide2.get('metadata', {}).get('slide_number', 0)

        prompt = f"""Vérifie la cohérence entre ces deux slides d'une présentation financière:

SLIDE {s1_num}:
{slide1.get('content', '')[:500]}

SLIDE {s2_num}:
{slide2.get('content', '')[:500]}

Détecte:
1. Contradictions factuelles (ex: "ETF" vs "Ce n'est pas un ETF")
2. Incohérences de données (montants, dates)
3. Messages contradictoires sur les risques/avantages

Réponds en JSON:
{{
  "has_contradiction": true/false,
  "contradiction_type": "type",
  "explanation": "explication",
  "confidence": 0.0-1.0
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="hosted_vllm/Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system", "content": "Expert en cohérence documentaire. JSON uniquement."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )

            result = json.loads(response.choices[0].message.content)

            if result.get('has_contradiction', False):
                return Violation(
                    rule_id="1.13",
                    violation_type="INCOHÉRENCE_SLIDES",
                    severity="high",
                    slides_involved=[s1_num, s2_num],
                    explanation=result.get('explanation', 'Contradiction détectée entre slides'),
                    violating_content=f"Slides {s1_num} vs {s2_num}",
                    correction="Vérifier et harmoniser les informations",
                    confidence=result.get('confidence', 0.7)
                )
        except Exception as e:
            print(f"⚠️ Erreur LLM Cohérence Slides {s1_num}-{s2_num}: {e}")

        return None

# ==========================================
# ANALYSEUR DE NEUTRALITÉ (RÈGLES + LLM)
# ==========================================

class NeutralityAnalyzer:
    """Analyseur hybride: règles + LLM"""

    def __init__(self, llm_client: Optional[OpenAI] = None, language: str = 'EN'):
        self.language = language
        self.llm_analyzer = LLMNeutralityAnalyzer(llm_client, language) if llm_client else None

        # Patterns interdits (CONSERVÉ - fonctionne bien)
        self.forbidden_patterns = {
            'promotional': [
                r'\bmeilleur\b', r'\bexcellent\b', r'\bunique\b', r'\bexceptionnel\b',
                r'\binégalé\b', r'\bgaranti\b', r'\bcertain\b', r'\bassure\b',
                r'\bbest\b', r'\bunique\b', r'\bexceptional\b', r'\bunmatched\b',
                r'\bguaranteed\b', r'\bcertain\b', r'\bensure\b', r'\bsafe\b'
            ],
            'performance_promise': [
                r'\bva générer\b', r'\bassure un rendement\b', r'\bpromesse\b',
                r'\bgarantit\b.*\bperformance\b', r'\bwill generate\b',
                r'\bassures a return\b', r'\bpromise\b'
            ]
        }
        self.attenuators = ['objectif', 'target', 'aim', 'vise à', 'potential', 'potentiel']

    def analyze(self, slides: List[Dict]) -> List[Violation]:
        violations = []
        seen_violations = set()  # ✅ AJOUT: Tracker les violations déjà détectées

        # 1. Analyse par règles (RAPIDE)
        for slide in slides:
            content = slide.get('content', '')
            slide_num = slide.get('metadata', {}).get('slide_number', 0)

            for category, patterns in self.forbidden_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # ✅ Créer une clé unique : slide + pattern + catégorie
                        violation_key = (slide_num, pattern, category)

                        if violation_key in seen_violations:
                            continue  # ✅ Skip si déjà détecté

                        start = max(0, match.start() - 50)
                        end = min(len(content), match.end() + 50)
                        context = content[start:end]

                        is_attenuated = any(att in context.lower() for att in self.attenuators)

                        if not is_attenuated:
                            seen_violations.add(violation_key)  # ✅ Marquer comme vu

                            violations.append(Violation(
                                rule_id="1.7",
                                violation_type="NEUTRALITÉ_DISCOURS",
                                severity="high" if category == 'performance_promise' else "medium",
                                slides_involved=[slide_num],
                                explanation=f"Langage non neutre détecté ({category}): '{match.group()}'",
                                violating_content=context.strip(),
                                correction="Reformuler avec un langage neutre et objectif",
                                confidence=0.9
                            ))

        # 2. Analyse LLM (APPROFONDIE - si disponible)
        if self.llm_analyzer:
            print("   🤖 Analyse LLM: Neutralité et Cohérence...")
            # Analyse neutralité sur les slides clés (1-5, 10-15)
            key_slides = [s for s in slides if s.get('metadata', {}).get('slide_number', 0) in [1,2,3,4,5,10,11,12,13,14,15]]

            for slide in key_slides:
                v = self.llm_analyzer.analyze_slide_neutrality(
                    slide.get('content', ''),
                    slide.get('metadata', {}).get('slide_number', 0)
                )
                if v: violations.append(v)

            # Cohérence entre slides consécutives critiques
            for i in range(len(slides) - 1):
                if i < 5:  # Slides 1-5
                    v = self.llm_analyzer.check_slide_coherence(slides[i], slides[i+1])
                    if v: violations.append(v)

        return violations

# ==========================================
# DÉTECTEUR D'ANGLICISMES AMÉLIORÉ
# ==========================================

class AnglicismDetector:
    """Détecteur d'anglicismes UNIQUEMENT si document FR"""

    def __init__(self):
        self.anglicisms = {
            'track record': 'historique de performance',
            'benchmark': 'indice de référence',
            'stock picking': 'sélection de titres',
            'bottom-up': 'analyse fondamentale',
            'reporting': 'rapport',
            'cash flow': 'flux de trésorerie',
            'yield': 'rendement',
            'turnover': 'rotation'
        }

    def analyze(self, slides: List[Dict], detected_language: str) -> List[Violation]:
        violations = []

        # ⚠️ CORRECTION CRITIQUE: Ne s'exécute QUE si FR
        if detected_language != 'FR':
            print(f"   ℹ️ Document en {detected_language}: Règle anglicismes désactivée")
            return []

        for slide in slides:
            content = slide.get('content', '')
            slide_num = slide.get('metadata', {}).get('slide_number', 0)

            # Ignorer slides techniques
            if 'glossaire' in content.lower() or 'glossary' in content.lower():
                continue

            for anglicism, french in self.anglicisms.items():
                if re.search(r'\b' + re.escape(anglicism) + r'\b', content, re.IGNORECASE):
                    violations.append(Violation(
                        rule_id="1.11",
                        violation_type="ANGLICISME",
                        severity="low",
                        slides_involved=[slide_num],
                        explanation=f"Anglicisme '{anglicism}' détecté en document Retail FR.",
                        violating_content=anglicism,
                        correction=f"Utiliser '{french}' ou définir le terme.",
                        confidence=0.85
                    ))
        return violations

# ==========================================
# VALIDATEUR DE DISCLAIMERS AMÉLIORÉ
# ==========================================

class DisclaimerValidator:
    """Validateur avec détection client type automatique"""

    def __init__(self, disclaimers_ref: Dict):
        self.disclaimers_ref = disclaimers_ref

    def _normalize(self, text):
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text or '').lower()).strip()

    def _fuzzy_check(self, text_found, text_required, threshold=0.55):
        if not text_required: return True
        norm_found = self._normalize(text_found)
        norm_req = self._normalize(text_required)
        if norm_req[:50] in norm_found: return True
        return SequenceMatcher(None, norm_found, norm_req).ratio() > threshold

    def _detect_client_type(self, slides: List[Dict]) -> str:
        """Détecte si retail ou professional"""
        all_text = " ".join([s.get('content', '') for s in slides[:3]]).lower()

        # Indicateurs PRO
        if any(kw in all_text for kw in ['professional', 'professionnel', 'mifid', 'qualified investor']):
            return 'professional'

        # Indicateurs RETAIL
        if any(kw in all_text for kw in ['retail', 'non-professional', 'non professionnel', 'kid', 'dic']):
            return 'non_professional'

        # Par défaut RETAIL (plus strict)
        return 'non_professional'

    def analyze(self, slides: List[Dict], footnotes: List[Dict], detected_language: str) -> List[Violation]:
        violations = []

        # Détection auto du type de client
        client_type_key = self._detect_client_type(slides)
        print(f"   ℹ️ Type client détecté: {client_type_key}")

        # Mapping slide -> footnotes
        slide_notes = defaultdict(str)
        for note in footnotes:
            s_num = note.get('metadata', {}).get('slide_number')
            if s_num: slide_notes[s_num] += " " + note.get('content', '')

        # A. Disclaimer Général (Slide 2) - Règle 3.1
        ref_text = self.disclaimers_ref['data'][client_type_key]['commercial_doc_manco_obam_sas'].get(detected_language.lower(), "")
        s2_text = slide_notes[2] + " " + next((s['content'] for s in slides if s.get('metadata', {}).get('slide_number') == 2), "")

        if not self._fuzzy_check(s2_text, ref_text, threshold=0.4):
            violations.append(Violation(
                rule_id="3.1",
                violation_type="DISCLAIMER_MANQUANT",
                severity="high",
                slides_involved=[2],
                explanation=f"Le disclaimer général ({client_type_key}) est manquant ou incomplet sur Slide 2.",
                violating_content="Slide 2 Footnotes",
                correction="Insérer le disclaimer standard complet.",
                confidence=1.0
            ))

        # B. Performance (Règle 4.3.1)
        ref_perf = self.disclaimers_ref['data'][client_type_key]['performance'].get(detected_language.lower(), "")
        keywords_perf = ['performance', 'yield', 'cumulative', 'annualized', 'rendement']

        for slide in slides:
            s_num = slide.get('metadata', {}).get('slide_number')
            content = slide.get('content', '').lower()

            if any(k in content for k in keywords_perf) and (slide.get('metadata', {}).get('has_charts') or slide.get('metadata', {}).get('has_tables')):
                notes_text = slide_notes[s_num]
                if not self._fuzzy_check(notes_text, ref_perf, threshold=0.6):
                     violations.append(Violation(
                        rule_id="4.3.1",
                        violation_type="DISCLAIMER_PERF_MANQUANT",
                        severity="high",
                        slides_involved=[s_num],
                        explanation="Slide de performance sans l'avertissement obligatoire.",
                        violating_content=f"Slide {s_num}",
                        correction=f"Ajouter: {ref_perf[:100]}...",
                        confidence=0.95
                    ))

        # C. Opinion (Règle 1.7)
        ref_op = self.disclaimers_ref['data'][client_type_key]['opinion'].get(detected_language.lower(), "")
        for slide in slides:
            s_num = slide.get('metadata', {}).get('slide_number')
            if 'opinion' in slide.get('content', '').lower() or 'forecast' in slide.get('content', '').lower():
                notes_text = slide_notes[s_num]
                if not self._fuzzy_check(notes_text, ref_op, threshold=0.5):
                     violations.append(Violation(
                        rule_id="1.7",
                        violation_type="DISCLAIMER_OPINION_MANQUANT",
                        severity="medium",
                        slides_involved=[s_num],
                        explanation="Opinions présentes sans disclaimer associé.",
                        violating_content=f"Slide {s_num}",
                        correction="Ajouter le disclaimer sur les opinions.",
                        confidence=0.8
                    ))

        return violations

# ==========================================
# CHECKER PROSPECTUS (INCHANGÉ - FONCTIONNE)
# ==========================================

class ProspectusCoherenceChecker:
    """Vérifie cohérence PPTX vs Prospectus"""

    def __init__(self, prospectus_data: Dict):
        self.truth = prospectus_data.get('prospectus_parsed', prospectus_data.get('metadata', {}))

    def analyze(self, extracted_pptx_data: Dict) -> List[Violation]:
        violations = []
        pptx_details = extracted_pptx_data.get('fund_info_detailed', {})

        mapping = {
            'fund_name': 'fund_name',
            'benchmark': 'benchmark',
            'esg_sfdr_article': 'esg_sfdr_article',
            'investment_objective': 'investment_objective'
        }

        for truth_key, pptx_key in mapping.items():
            truth_val = str(self.truth.get(truth_key, '')).strip().lower()

            pptx_entry = pptx_details.get(pptx_key, {})
            pptx_val = ""
            if isinstance(pptx_entry, dict):
                pptx_val = str(pptx_entry.get('value', '')).strip().lower()
            else:
                pptx_val = str(pptx_entry).strip().lower()

            if not truth_val or not pptx_val:
                continue

            is_coherent = (truth_val in pptx_val) or (pptx_val in truth_val)

            if truth_key == 'esg_sfdr_article':
                is_coherent = (truth_val == pptx_val)

            if not is_coherent:
                violations.append(Violation(
                    rule_id="4.5",
                    violation_type="INCOHÉRENCE_PROSPECTUS",
                    severity="high",
                    slides_involved=[],
                    explanation=f"Divergence sur '{truth_key}'.",
                    violating_content=f"PPTX: '{pptx_val}' vs Prospectus: '{truth_val}'",
                    correction=f"Corriger dans PPTX: {truth_val}",
                    confidence=1.0
                ))

        return violations

# ==========================================
# ORCHESTRATEUR GLOBAL
# ==========================================

class GlobalComplianceAnalyzer:
    """Orchestrateur avec LLM"""

    def __init__(self, disclaimers_ref, rules_ref, prospectus_data, enable_llm=True):
        self.llm_client = get_llm_client() if enable_llm else None
        self.lang_detector = LanguageDetector()
        self.prospectus = ProspectusCoherenceChecker(prospectus_data)

        # Ces analyseurs seront initialisés après détection de langue
        self.neutrality = None
        self.anglicism = None
        self.disclaimer = DisclaimerValidator(disclaimers_ref)

    def run_analysis(self, slides_full, footnotes, extracted_pptx_data):
        print("🚀 Démarrage de l'analyse globale...")

        # 1. Détection de langue
        detected_lang = self.lang_detector.detect(slides_full)
        print(f"   🌐 Langue détectée: {detected_lang}")

        # 2. Initialisation des analyseurs avec la bonne langue
        self.neutrality = NeutralityAnalyzer(self.llm_client, detected_lang)
        self.anglicism = AnglicismDetector()

        # 3. Analyses
        print("   📊 Checking: Cohérence Prospectus...")
        v_prosp = self.prospectus.analyze(extracted_pptx_data)

        print("   📋 Checking: Disclaimers...")
        v_disc = self.disclaimer.analyze(slides_full, footnotes, detected_lang)

        print("   💬 Checking: Neutralité (Règles + LLM)...")
        v_neut = self.neutrality.analyze(slides_full)

        print("   🔤 Checking: Anglicismes...")
        v_angl = self.anglicism.analyze(slides_full, detected_lang)

        all_violations = v_prosp + v_disc + v_neut + v_angl

        doc_name = extracted_pptx_data.get('fund_info_complete', {}).get('fund_name', 'Document Inconnu')

        result = GlobalAnalysisResult(
            document_name=doc_name,
            analysis_date=datetime.now().isoformat(),
            violations=[v for v in all_violations if v.severity == 'high'],
            recommendations=[v for v in all_violations if v.severity != 'high'],
            statistics={'total_issues': len(all_violations)}
        )

        return result

# ==========================================
# EXÉCUTION
# ==========================================

def load_json_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Fichier manquant : {path}")
        return {}
    except Exception as e:
        print(f"⚠️ Erreur lecture {path}: {e}")
        return {}

# Chargement
print("📂 Chargement des données...")
slides_data = load_json_file('example_2/outputs/pptx_slide_full.json')
footnotes_data = load_json_file('example_2/outputs/pptx_footnote.json')
disclaimers_db = load_json_file('inputs/disclamer_bilingual_full.json')
prospectus_db = load_json_file('example_2/outputs/enriched_context_test.json')
pptx_extracted_db = load_json_file('example_2/outputs/enriched_improved_complete_test.json')

# Analyse (avec LLM activé)
analyzer = GlobalComplianceAnalyzer(
    disclaimers_db,
    {},
    prospectus_db,
    enable_llm=True  # ⚠️ Mettre False pour désactiver LLM
)
report = analyzer.run_analysis(slides_data, footnotes_data, pptx_extracted_db)

# Affichage
print(f"\n{'='*60}")
print(f"RAPPORT DE CONFORMITÉ : {report.document_name}")
print(f"Date : {report.analysis_date}")
print(f"{'='*60}\n")

if report.violations:
    print(f"🚨 VIOLATIONS CRITIQUES ({len(report.violations)})")
    df_v = pd.DataFrame([vars(v) for v in report.violations])
    cols = ['rule_id', 'violation_type', 'slides_involved', 'explanation', 'violating_content']
    print(df_v[cols].to_string(index=False))
else:
    print("✅ Aucune violation critique.")

print("\n")
if report.recommendations:
    print(f"⚠️ RECOMMANDATIONS ({len(report.recommendations)})")
    df_r = pd.DataFrame([vars(v) for v in report.recommendations])
    cols = ['rule_id', 'violation_type', 'slides_involved', 'explanation']
    print(df_r[cols].to_string(index=False))
else:
    print("✅ Aucune recommandation.")

print(f"\n{'='*60}")

# Export JSON
output_filename = "example_2/outputs/global_compliance_report.json"
try:
    report_dict = asdict(report)
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    print(f"✅ Rapport sauvegardé: {output_filename}")
except Exception as e:
    print(f"❌ Erreur sauvegarde: {e}")