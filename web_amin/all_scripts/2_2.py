# Generated from: 2_2_rag+llmipynb.ipynb
# Converted at: 2025-12-16T20:46:26.972Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

"""
PHASE 2.2 - FUSION OPTIMALE: Haute Précision + Couverture Complète
============================================================
Stratégie:
1. Force matching structurel (slides 1,2,34,35) - du Code 2
2. RAG enrichi avec garde-fous métier - du Code 1
3. Boost spécifique pour slides à performance - Hybride
4. LLM validation intelligente - du Code 1
============================================================
"""

import json
import numpy as np
import faiss
import pickle
import httpx
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from sentence_transformers import SentenceTransformer
from datetime import datetime
from openai import OpenAI
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# ⚙️ CONFIGURATION FUSIONNÉE
# ============================================================

class Config:
    """Configuration optimale équilibrant précision et rappel"""

    # Chemins fichiers
    PHASE21_OUTPUT = "example_2/outputs/phase2_1_hybrid_output2.json"
    PHASE1_OUTPUT = "example_2/outputs/phase1_correct_output.json"
    METADATA = "example_2/outputs/metadata_final_optimized_test.json"
    DISCLAIMERS = "inputs/disclamer_bilingual_full.json"
    RULES_JSON = "inputs/vector_db_rules_merged_fixed.json"
    RULES_INDEX = "inputs/vectordb_rules.index"
    RULES_PKL = "inputs/vectordb_rules.pkl"
    OUTPUT_JSON = "example_2/outputs/phase2_2_fusion_optimale.json"

    # Modèles
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM Config
    LLM_API_KEY = "sk-7c0b80cf494746f580cc5ba555d739b2"
    LLM_BASE_URL = "https://tokenfactory.esprit.tn/api"
    LLM_MODEL = "hosted_vllm/Llama-3.1-70B-Instruct"
    USE_LLM = True
    LLM_MAX_RETRIES = 2

    # 🎯 PARAMÈTRES HYBRIDES (Clé de la fusion)
    # Structural slides: force matching strict
    TOP_K_STRUCTURAL = 3  # Slides 1,2,34,35

    # Performance slides: haute couverture
    TOP_K_PERFORMANCE = 12  # Slides 12-30 avec perf
    SCORE_THRESHOLD_PERFORMANCE = 0.40

    # Content slides: équilibré
    TOP_K_CONTENT = 6
    SCORE_THRESHOLD_CONTENT = 0.50

    # Base FAISS
    TOP_K_FAISS_BASE = 25

    # Garde-fous
    MAX_GENERIC_RULES_PER_SLIDE = 2
    REQUIRE_CONTENT_MATCH = True
    ENABLE_SPECIFICITY_BOOST = True

# ============================================================
# 🧠 LLM CLIENT
# ============================================================

class LLMClient:
    def __init__(self):
        self.enabled = Config.USE_LLM
        if self.enabled:
            try:
                http_client = httpx.Client(verify=False, timeout=30.0)
                self.client = OpenAI(
                    api_key=Config.LLM_API_KEY,
                    base_url=Config.LLM_BASE_URL,
                    http_client=http_client
                )
                logger.info("✅ LLM activé")
            except Exception as e:
                logger.warning(f"⚠️ LLM init échec: {e}")
                self.enabled = False
        else:
            logger.info("ℹ️ Mode FAISS uniquement")

    def query(self, system_prompt: str, user_prompt: str,
              temperature: float = 0.2, max_tokens: int = 1000) -> str:
        if not self.enabled:
            return ""

        for attempt in range(Config.LLM_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.error("❌ API Key invalide")
                    self.enabled = False
                    return ""
                if attempt == Config.LLM_MAX_RETRIES - 1:
                    logger.warning(f"⚠️ LLM échec: {e}")
        return ""

# ============================================================
# 📊 DATA LOADER
# ============================================================

class DataLoader:
    def __init__(self):
        self.phase21_data = self._load_json(Config.PHASE21_OUTPUT)
        self.phase1_data = self._load_json(Config.PHASE1_OUTPUT)
        self.metadata = self._load_json(Config.METADATA)
        self.disclaimers = self._load_json(Config.DISCLAIMERS)
        self.rules_json = self._load_json(Config.RULES_JSON)

        self.rules_index = faiss.read_index(Config.RULES_INDEX)
        with open(Config.RULES_PKL, 'rb') as f:
            self.rules_mapping = pickle.load(f)

        self.applicable_rules = {
            r['rule_id']: r
            for r in self.phase1_data.get('applicable_rules', [])
        }

        self.rules_by_index = self._build_rules_index()
        self.doc_context = self._extract_document_context()

        logger.info(
            f"✅ Chargé: {len(self.phase21_data.get('chunks_enriched', []))} slides, "
            f"{len(self.applicable_rules)} règles applicables"
        )

    def _load_json(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_rules_index(self) -> Dict[int, str]:
        mapping = {}
        if isinstance(self.rules_json, list):
            for idx, rule in enumerate(self.rules_json):
                if isinstance(rule, dict) and 'rule_id' in rule:
                    mapping[idx] = rule['rule_id']
        elif isinstance(self.rules_json, dict):
            rules_array = self.rules_json.get('rules', [])
            for idx, rule in enumerate(rules_array):
                if isinstance(rule, dict) and 'rule_id' in rule:
                    mapping[idx] = rule['rule_id']
        return mapping

    def _extract_document_context(self) -> Dict:
        fund_info = self.metadata.get('fund_info', {})
        doc_info = self.metadata.get('document_info', {})
        compliance = self.metadata.get('compliance', {}).get('derived', {})

        return {
            'client_type': doc_info.get('target_audience', 'retail'),
            'fund_type': fund_info.get('fund_type', ''),
            'esg_article': fund_info.get('esg_sfdr_article', ''),
            'is_new_product': self.metadata.get('product_status', {}).get('is_new_product', False),
            'is_new_strategy': self.metadata.get('product_status', {}).get('is_new_strategy', False),
            'language': doc_info.get('language', 'FR'),
            'countries': self.metadata.get('distribution', {}).get('target_countries', []),
            'has_benchmark': fund_info.get('benchmark') is not None,
            'can_display_performance': compliance.get('performance_constraints', {}).get('can_display_performance', False)
        }

# ============================================================
# 🛡️ BUSINESS FILTERS (du Code 1 - Garde-fous métier)
# ============================================================

class BusinessFilters:
    @staticmethod
    def is_rule_applicable_to_slide(rule: Dict, slide_chunk: Dict, doc_context: Dict) -> Tuple[bool, str]:
        """Vérifie si règle s'applique vraiment à cette slide"""
        rule_id = rule['rule_id']
        check_type = rule.get('check_type', '')
        flags = slide_chunk.get('flags', {})
        semantic = slide_chunk.get('semantic_analysis', {})

        # 1️⃣ Règles PERFORMANCE
        if 'performance' in check_type.lower():
            if not flags.get('has_performance') and not semantic.get('data_presented', {}).get('performance_values'):
                # Exception pour règles d'ordre de slides
                if 'slide_order' not in check_type.lower():
                    return False, "Règle performance mais slide sans perf"

        # 2️⃣ Règles BENCHMARK
        if 'benchmark' in check_type.lower():
            if not flags.get('has_benchmark_comparison') and not semantic.get('data_presented', {}).get('benchmark_comparison'):
                return False, "Règle benchmark mais slide sans comparaison"

        # 3️⃣ Règles CITATION/GRAPHIQUES
        if 'citation' in check_type.lower():
            visual = semantic.get('visual_elements', {})
            if not visual.get('has_chart') and not visual.get('has_table'):
                return False, "Règle citation mais slide sans graphique/tableau"

        # 4️⃣ Règles OPINIONS
        if 'opinion' in check_type.lower() or 'linguistic' in check_type.lower():
            if not semantic.get('data_presented', {}).get('contains_opinions'):
                # Exception pour règles générales de restriction
                if rule_id not in ['4.2.1', '4.2.3']:
                    return False, "Règle opinion mais slide factuelle"

        # 5️⃣ Règles CLIENT TYPE
        applicable_to = rule.get('applicable_to', [])
        if applicable_to and doc_context['client_type'] not in applicable_to:
            return False, f"Règle {applicable_to} mais document {doc_context['client_type']}"

        return True, "OK"

    @staticmethod
    def calculate_rule_specificity_score(rule: Dict, slide_chunk: Dict) -> float:
        """Score de spécificité (du Code 1)"""
        if not Config.ENABLE_SPECIFICITY_BOOST:
            return 1.0

        score = 1.0
        check_type = rule.get('check_type', '')
        flags = slide_chunk.get('flags', {})
        semantic = slide_chunk.get('semantic_analysis', {})

        # Boost spécificité
        if 'performance' in check_type and flags.get('has_performance'):
            score += 1.5  # 🔥 Boost augmenté pour slides perf
        if 'benchmark' in check_type and flags.get('has_benchmark_comparison'):
            score += 1.5
        if 'citation' in check_type:
            visual = semantic.get('visual_elements', {})
            if visual.get('has_chart') or visual.get('has_table'):
                score += 0.8

        # Pénalité généricité
        if BusinessFilters.is_generic_rule(rule):
            score *= 0.5

        return score

    @staticmethod
    def is_generic_rule(rule: Dict) -> bool:
        """Détecte règles trop génériques"""
        generic_keywords = ['1.1', '1.2', '1.7', '1.13', '4.2.1']
        return rule['rule_id'] in generic_keywords

# ============================================================
# 🎯 DÉTECTEUR STRUCTUREL (du Code 2 - Force matching)
# ============================================================

class StructuralRuleDetector:
    """Détecte les règles structurelles par position de slide"""

    STRUCTURAL_PATTERNS = {
        'page_garde': {
            'check_types': ['page_garde_structure', 'page_garde_precommercialisation'],
            'slide_target': 'first'
        },
        'slide_2': {
            'check_types': ['slide_2_disclaimer', 'slide_2_risk_profile', 'slide_2_countries'],
            'slide_target': 'second'
        },
        'page_fin': {
            'check_types': ['page_fin_structure'],
            'slide_target': 'last'
        }
    }

    @classmethod
    def detect_structural_rules(cls, rules: Dict[str, Dict]) -> Dict[str, List[str]]:
        structural_map = {'slide_1': [], 'slide_2': [], 'last_slides': []}

        for rule_id, rule in rules.items():
            if not isinstance(rule, dict):
                continue

            section = str(rule.get('section', '')).lower()
            check_type = str(rule.get('check_type', '')).lower()

            # Slide 1
            if '2 -- page de garde' in section or check_type in cls.STRUCTURAL_PATTERNS['page_garde']['check_types']:
                structural_map['slide_1'].append(rule_id)

            # Slide 2
            if '3 -- la slide 2' in section or check_type in cls.STRUCTURAL_PATTERNS['slide_2']['check_types']:
                structural_map['slide_2'].append(rule_id)

            # Dernières slides
            if '5 -- page de fin' in section or check_type in cls.STRUCTURAL_PATTERNS['page_fin']['check_types']:
                structural_map['last_slides'].append(rule_id)

        return structural_map

# ============================================================
# 🔍 SÉLECTEUR FUSIONNÉ (LOGIQUE HYBRIDE)
# ============================================================

class FusedRuleSelector:
    """Combine force matching structurel + RAG enrichi + LLM validation"""

    def __init__(self, data_loader: DataLoader, llm_client: LLMClient):
        self.data = data_loader
        self.llm = llm_client
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.filters = BusinessFilters()
        self.structural_detector = StructuralRuleDetector()

        # Détection règles structurelles
        self.structural_rules = self.structural_detector.detect_structural_rules(
            self.data.applicable_rules
        )

        logger.info(f"🏗️ Règles structurelles: Slide1={len(self.structural_rules['slide_1'])}, "
                   f"Slide2={len(self.structural_rules['slide_2'])}, "
                   f"Last={len(self.structural_rules['last_slides'])}")

    # --------------------------------------------------------
    # 🎯 DÉTECTION TYPE DE SLIDE (Clé de la stratégie hybride)
    # --------------------------------------------------------

    def _classify_slide_type(self, slide_chunk: Dict, slide_num: int, total_slides: int) -> str:
        """
        Classifie la slide pour adapter la stratégie de matching:
        - structural: Slides 1, 2, 34, 35 → Force matching
        - performance: Slides 12-30 avec perf → Haute couverture
        - content: Autres → Équilibré
        """
        # Structural
        if slide_num == 1 or slide_num == 2:
            return 'structural'
        if slide_num >= total_slides - 1:
            return 'structural'

        # Performance (critère enrichi)
        flags = slide_chunk.get('flags', {})
        semantic = slide_chunk.get('semantic_analysis', {})
        data = semantic.get('data_presented', {})

        has_perf_content = (
            flags.get('has_performance') or
            data.get('performance_values') or
            data.get('benchmark_comparison') or
            'performance' in semantic.get('semantic_themes', [])
        )

        if 12 <= slide_num <= 30 and has_perf_content:
            return 'performance'

        return 'content'

    # --------------------------------------------------------
    # 📝 QUERY CONTEXTUELLE (du Code 1)
    # --------------------------------------------------------

    def build_contextual_query(self, slide_chunk: Dict) -> str:
        """Query enrichie avec contexte document"""
        slide_num = slide_chunk['metadata']['slide_number']
        semantic = slide_chunk.get('semantic_analysis', {})

        parts = [f"Slide {slide_num}"]

        # Contexte document
        doc_ctx = self.data.doc_context
        context_items = [f"{doc_ctx['client_type']}", f"{doc_ctx['fund_type']}", f"{doc_ctx['esg_article']}"]
        parts.append(f"Doc[{', '.join(context_items)}]")

        # Résumé + thèmes
        resume = semantic.get('resume_slide', '')
        if resume:
            parts.append(f"Résumé: {resume}")

        themes = semantic.get('semantic_themes', [])
        if themes:
            parts.append(f"Thèmes: {', '.join(themes)}")

        # Visuels
        visual = semantic.get('visual_elements', {})
        visual_items = []
        if visual.get('has_chart'): visual_items.append("graphique")
        if visual.get('has_table'): visual_items.append("tableau")
        if visual_items:
            parts.append(f"Visuels: {', '.join(visual_items)}")

        # Données
        data_presented = semantic.get('data_presented', {})
        data_items = []
        if data_presented.get('performance_values'): data_items.append("valeurs performance")
        if data_presented.get('benchmark_comparison'): data_items.append("comparaison benchmark")
        if data_items:
            parts.append(f"Données: {', '.join(data_items)}")

        return " | ".join(parts)

    # --------------------------------------------------------
    # 🏗️ FORCE MATCHING STRUCTUREL (du Code 2)
    # --------------------------------------------------------

    def _get_structural_rules(self, slide_num: int, total_slides: int) -> List[Dict]:
        """Force matching pour slides structurelles"""
        target_ids = []

        if slide_num == 1:
            target_ids = self.structural_rules['slide_1']
        elif slide_num == 2:
            target_ids = self.structural_rules['slide_2']
        elif slide_num >= total_slides - 1:
            target_ids = self.structural_rules['last_slides']

        mandatory_rules = []
        for rid in target_ids:
            if rid in self.data.applicable_rules:
                rule = self.data.applicable_rules[rid].copy()
                rule['faiss_score'] = 1.0
                rule['specificity_score'] = 1.0
                rule['final_score'] = 1.0
                rule['force_matched'] = True
                rule['llm_validated'] = False
                rule['is_generic'] = False
                mandatory_rules.append(rule)

        return mandatory_rules

    # --------------------------------------------------------
    # 🔍 FAISS SEARCH ADAPTATIF (Hybride)
    # --------------------------------------------------------

    def _search_faiss_adaptive(self, query: str, slide_chunk: Dict, slide_type: str) -> List[Dict]:
        """RAG FAISS avec paramètres adaptés au type de slide"""

        # Paramètres adaptatifs
        if slide_type == 'performance':
            top_k = Config.TOP_K_PERFORMANCE
            threshold = Config.SCORE_THRESHOLD_PERFORMANCE
        elif slide_type == 'structural':
            top_k = Config.TOP_K_STRUCTURAL
            threshold = 0.60
        else:  # content
            top_k = Config.TOP_K_CONTENT
            threshold = Config.SCORE_THRESHOLD_CONTENT

        # Encoder query
        query_vector = self.encoder.encode([query])[0].astype('float32').reshape(1, -1)

        # Recherche FAISS
        distances, indices = self.data.rules_index.search(query_vector, Config.TOP_K_FAISS_BASE)

        candidates = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            rule_id = self.data.rules_by_index.get(int(idx))
            if not rule_id or rule_id not in self.data.applicable_rules:
                continue

            score = 1.0 / (1.0 + float(distance))
            if score < threshold:
                continue

            rule = self.data.applicable_rules[rule_id].copy()
            rule['faiss_score'] = round(score, 3)

            # Garde-fou métier
            if Config.REQUIRE_CONTENT_MATCH:
                is_applicable, reason = self.filters.is_rule_applicable_to_slide(
                    rule, slide_chunk, self.data.doc_context
                )
                if not is_applicable:
                    continue

            # Spécificité
            specificity = self.filters.calculate_rule_specificity_score(rule, slide_chunk)
            rule['specificity_score'] = round(specificity, 3)
            rule['final_score'] = round(score * specificity, 3)
            rule['force_matched'] = False
            rule['llm_validated'] = False
            rule['is_generic'] = self.filters.is_generic_rule(rule)

            candidates.append(rule)

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates[:top_k]

    # --------------------------------------------------------
    # 🤖 LLM VALIDATION SÉLECTIVE (du Code 1)
    # --------------------------------------------------------

    def _llm_validate_candidates(self, slide_chunk: Dict, candidates: List[Dict]) -> List[Dict]:
        """LLM validation pour zone grise uniquement"""
        if not self.llm.enabled or not candidates:
            return candidates

        slide_num = slide_chunk['metadata']['slide_number']
        semantic = slide_chunk.get('semantic_analysis', {})

        # Contexte slide compact
        slide_context = {
            "slide_number": slide_num,
            "resume": semantic.get('resume_slide', '')[:200],
            "themes": semantic.get('semantic_themes', []),
            "has_performance": slide_chunk.get('flags', {}).get('has_performance', False),
        }

        # On ne valide que le top et les scores moyens
        to_validate = [c for c in candidates if 0.45 < c['final_score'] < 0.75]

        if not to_validate:
            return candidates

        validated_candidates = candidates.copy()

        for cand in to_validate:
            system_prompt = """Tu es un expert compliance. Analyse si cette règle s'applique à la slide.
Réponds UNIQUEMENT: {"applicable": true/false, "confidence": 0-1}"""

            user_prompt = f"""Slide: {json.dumps(slide_context, ensure_ascii=False)}
Règle ID: {cand['rule_id']}
Type: {cand.get('check_type')}
Texte: {cand.get('rule_text', '')[:200]}

Cette règle s'applique-t-elle?"""

            llm_raw = self.llm.query(system_prompt, user_prompt, temperature=0.1, max_tokens=200)
            if not llm_raw:
                continue

            try:
                txt = llm_raw.strip()
                if txt.startswith("```"):
                    parts = txt.split("```")
                    if len(parts) >= 2:
                        txt = parts[1].replace("json", "").strip()

                data = json.loads(txt)
                if data.get("applicable"):
                    # Trouve et modifie dans validated_candidates
                    for i, v in enumerate(validated_candidates):
                        if v['rule_id'] == cand['rule_id']:
                            validated_candidates[i]['llm_validated'] = True
                            # Petit boost si LLM valide
                            validated_candidates[i]['final_score'] *= 1.1
                            break
                else:
                    # Retirer si LLM rejette
                    validated_candidates = [v for v in validated_candidates if v['rule_id'] != cand['rule_id']]
            except:
                pass

        return validated_candidates

    # --------------------------------------------------------
    # 🎯 TRAITEMENT SLIDE (ORCHESTRATION FUSIONNÉE)
    # --------------------------------------------------------

    def process_slide(self, slide_chunk: Dict, total_slides: int) -> Dict:
        """Traite une slide avec stratégie adaptative"""
        slide_num = slide_chunk['metadata']['slide_number']
        slide_title = slide_chunk['metadata'].get('slide_title', '')

        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Slide {slide_num}: {slide_title}")

        # 1️⃣ Classifier le type de slide
        slide_type = self._classify_slide_type(slide_chunk, slide_num, total_slides)
        logger.info(f"📂 Type: {slide_type}")

        # 2️⃣ Force matching structurel (si applicable)
        mandatory_rules = []
        if slide_type == 'structural':
            mandatory_rules = self._get_structural_rules(slide_num, total_slides)
            logger.info(f"🏗️ {len(mandatory_rules)} règles structurelles forcées")

        # 3️⃣ Query contextuelle
        query = self.build_contextual_query(slide_chunk)

        # 4️⃣ FAISS search adaptatif
        candidates = self._search_faiss_adaptive(query, slide_chunk, slide_type)
        logger.info(f"🔎 FAISS: {len(candidates)} candidats ({slide_type})")

        # 5️⃣ LLM validation (zone grise uniquement)
        if slide_type != 'structural':  # Pas de LLM sur structural (déjà forcé)
            candidates = self._llm_validate_candidates(slide_chunk, candidates)
            logger.info(f"🤖 Après LLM: {len(candidates)} candidats")

        # 6️⃣ Merge final (mandatory + validated)
        seen_ids = set()
        final_rules = []

        # D'abord les mandatory (structural)
        for rule in mandatory_rules:
            if rule['rule_id'] not in seen_ids:
                final_rules.append(rule)
                seen_ids.add(rule['rule_id'])

        # Ensuite les candidates (éviter doublons)
        for rule in candidates:
            if rule['rule_id'] not in seen_ids:
                final_rules.append(rule)
                seen_ids.add(rule['rule_id'])

        # Limiter au top_k selon type
        if slide_type == 'performance':
            final_rules = final_rules[:Config.TOP_K_PERFORMANCE]
        elif slide_type == 'structural':
            final_rules = final_rules[:Config.TOP_K_STRUCTURAL]
        else:
            final_rules = final_rules[:Config.TOP_K_CONTENT]

        logger.info(f"✅ {len(final_rules)} règles sélectionnées")

        # 7️⃣ Stats et validation
        validation = self._post_validation(final_rules, slide_chunk)

        return {
            'slide_number': slide_num,
            'slide_title': slide_title,
            'slide_type': slide_type,
            'query': query,
            'selected_rules': [
                {
                    'rule_id': r['rule_id'],
                    'rule_text': r.get('rule_text', '')[:150] + '...',
                    'check_type': r.get('check_type', ''),
                    'section': r.get('section', ''),
                    'severity': r.get('severity', ''),
                    'faiss_score': r.get('faiss_score', 0),
                    'specificity_score': r.get('specificity_score', 0),
                    'final_score': r.get('final_score', 0),
                    'force_matched': r.get('force_matched', False),
                    'llm_validated': r.get('llm_validated', False),
                    'is_generic': r.get('is_generic', False)
                }
                for r in final_rules
            ],
            'validation': validation,
            'statistics': {
                'faiss_candidates': len(candidates),
                'final_rules': len(final_rules),
                **validation
            }
        }

    def _post_validation(self, rules: List[Dict], slide_chunk: Dict) -> Dict:
        """Validation post-sélection"""
        check_types = Counter([r.get('check_type', '') for r in rules])
        severities = Counter([r.get('severity', '') for r in rules])

        warnings = []
        flags = slide_chunk.get('flags', {})

        # Vérifications cohérence
        if check_types.get('performance_disclaimer', 0) > 0 and not flags.get('has_performance'):
            warnings.append("Règles performance mais slide sans perf")

        generic_count = sum(1 for r in rules if r.get('is_generic'))
        if generic_count > Config.MAX_GENERIC_RULES_PER_SLIDE:
            warnings.append(f"Trop de règles génériques ({generic_count})")

        return {
            'by_check_type': dict(check_types),
            'by_severity': dict(severities),
            'generic_rules_count': generic_count,
            'warnings': warnings,
            'coherence_score': max(0.0, 1.0 - (len(warnings) * 0.2))
        }

    # --------------------------------------------------------
    # 🚀 RUN GLOBAL
    # --------------------------------------------------------

    def run(self) -> Dict:
        """Traite toutes les slides"""
        logger.info("\n" + "="*80)
        logger.info("🚀 PHASE 2.2 FUSION OPTIMALE - Force Match + RAG + LLM")
        logger.info("="*80)

        results = []
        # CORRECTION ICI: Utilisation de phase21_data au lieu du texte corrompu
        chunks = self.data.phase21_data.get('chunks_enriched', [])
        total_slides = len(chunks)

        for i, slide_chunk in enumerate(chunks, 1):
            logger.info(f"\n📊 Progression: {i}/{total_slides}")
            result = self.process_slide(slide_chunk, total_slides)
            results.append(result)

        # Stats globales
        total_rules = sum(r['statistics']['final_rules'] for r in results)
        avg_rules = total_rules / len(results) if results else 0

        rule_counts = Counter()
        slide_type_counts = Counter()
        for r in results:
            slide_type_counts[r.get('slide_type', 'unknown')] += 1
            for rule in r['selected_rules']:
                rule_counts[rule['rule_id']] += 1

        output = {
            'metadata': {
                'phase': '2.2_fusion_optimale',
                'description': 'Force match structurel + RAG enrichi + LLM validation',
                'timestamp': datetime.now().isoformat(),
                'fund_name': self.data.metadata.get('fund_info', {}).get('fund_name', ''),
                'document_context': self.data.doc_context,
                'config': {
                    'top_k_structural': Config.TOP_K_STRUCTURAL,
                    'top_k_performance': Config.TOP_K_PERFORMANCE,
                    'top_k_content': Config.TOP_K_CONTENT,
                    'use_llm': Config.USE_LLM,
                    'require_content_match': Config.REQUIRE_CONTENT_MATCH
                }
            },
            'slides_analysis': results,
            'global_statistics': {
                'total_slides': len(results),
                'total_rules_selected': total_rules,
                'average_rules_per_slide': round(avg_rules, 2),
                'unique_rules_used': len(rule_counts),
                'slide_types_distribution': dict(slide_type_counts),
                'structural_rules_detected': {
                    'slide_1': len(self.structural_rules['slide_1']),
                    'slide_2': len(self.structural_rules['slide_2']),
                    'last_slides': len(self.structural_rules['last_slides'])
                },
                'most_frequent_rules': [
                    {
                        'rule_id': rid,
                        'occurrences': count,
                        'percentage': round(count / len(results) * 100, 1)
                    }
                    for rid, count in rule_counts.most_common(10)
                ]
            }
        }

        logger.info("\n" + "="*80)
        logger.info("✅ PHASE 2.2 FUSION TERMINÉE")
        logger.info(f"📊 {total_rules} règles | {avg_rules:.1f} moy/slide")
        logger.info(f"📂 Types: {dict(slide_type_counts)}")
        logger.info("="*80)

        return output

# ============================================================
# ▶️ EXÉCUTION
# ============================================================

def main():
    print("\n" + "="*80)
    print("🚀 PHASE 2.2 FUSION OPTIMALE")
    print("="*80 + "\n")
    data_loader = DataLoader()
    llm_client = LLMClient()
    selector = FusedRuleSelector(data_loader, llm_client)

    output = selector.run()

    # Sauvegarde
    output_path = Path(Config.OUTPUT_JSON)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Sauvegardé: {output_path}")
    print(f"📊 Slides: {len(output['slides_analysis'])}")
    print(f"📊 Règles: {output['global_statistics']['total_rules_selected']}")
    print(f"📊 Répartition: {output['global_statistics']['slide_types_distribution']}")

if __name__ == "__main__":
    main()