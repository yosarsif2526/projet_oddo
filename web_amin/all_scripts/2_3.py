# Generated from: 2_3_ipynb.ipynb
# Converted at: 2025-12-16T21:18:25.747Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

"""
============================================================================
PHASE 2.3 - VÉRIFICATION MULTI-PROMPT OPTIMISÉE
============================================================================
AMÉLIORATIONS:
✅ Extraction correcte des métadonnées (is_new_product, etc.)
✅ Classification: VIOLATIONS (100% certaines) vs RECOMMANDATIONS
✅ Règles 1.x globales: vérification multi-slides
✅ Gestion adaptée des nouveaux produits (pas de performance obligatoire)

✅ AJOUT (SANS MODIFIER LA LOGIQUE): gestion des règles de plage (RANGE_RULES)
- Si une règle de plage est CONFORME sur au moins 1 slide de la plage,
  alors on SUPPRIME les violations de cette règle sur les autres slides de la même plage.
============================================================================
"""

import json
import re
import httpx
from openai import OpenAI
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# ⚙️ CONFIGURATION
# ============================================================================

class Config:
    """Configuration centrale - ADAPTER SELON VOS FICHIERS"""
    OUTPUT_225 = "example_2/outputs/phase2_25_filtered_ultra.json"
    PHASE1_RULES = "example_2/outputs/phase1_correct_output.json"
    PPTX_CHUNKS = "example_2/outputs/pptx_slide_full.json"
    ENRICHED_CONTEXT = "example_2/outputs/metadata_final_optimized_test.json"
    DISCLAIMERS_JSON = "inputs/disclamer_bilingual_full.json"
    REGISTRATION_JSON = "inputs/fund_registration_analysis_youssef.json"
    LLM_API_KEY = "sk-7c0b80cf494746f580cc5ba555d739b2"
    LLM_BASE_URL = "https://tokenfactory.esprit.tn/api"
    LLM_MODEL = "hosted_vllm/Llama-3.1-70B-Instruct"
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 1500
    OUTPUT_JSON = "example_2/outputs/phase2_3_violations_optimized.json"
    SLIDE_1_ALLOWED_RULES = ["2.1", "2.2", "2.3", "2.4"]
    GLOBAL_RULES = ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "1.12", "1.13", "1.14", "1.15", "1.16"]

    # ✅ RANGE_RULES (inchangé, on ajoute juste l’usage)
    # - min_slide => [min_slide .. max_slide du doc]
    # - range: (start, end) => [start .. end]
    RANGE_RULES = {"1.5": {"min_slide": 30}, "5.1": {"min_slide": 30}, "4.4": {"range": (30, 34)}}

    NEW_PRODUCT_EXEMPT_RULES = ["4.3.2", "4.3.3", "4.3.9", "4.3.15"]
    VIOLATION_THRESHOLD = 0.85


# ✅ AJOUT: helpers range rules (sans modifier la logique)
def _resolve_range(rule_id: str, max_slide: int) -> Tuple[int, int]:
    rr = Config.RANGE_RULES.get(rule_id, {}) if hasattr(Config, "RANGE_RULES") else {}
    if isinstance(rr, dict) and "range" in rr and isinstance(rr["range"], (list, tuple)) and len(rr["range"]) == 2:
        start, end = rr["range"]
        return int(start), int(end)
    start = int(rr.get("min_slide", 1)) if isinstance(rr, dict) else 1
    end = int(rr.get("max_slide", max_slide)) if isinstance(rr, dict) else max_slide
    return start, end


def _in_range(rule_id: str, slide_num: int, max_slide: int) -> bool:
    if slide_num is None:
        return False
    start, end = _resolve_range(rule_id, max_slide)
    return start <= int(slide_num) <= end


# ============================================================================
# 🤖 CLIENT LLM
# ============================================================================

class LLMClient:
    def __init__(self):
        try:
            http_client = httpx.Client(verify=False, timeout=60.0)
            self.client = OpenAI(api_key=Config.LLM_API_KEY, base_url=Config.LLM_BASE_URL, http_client=http_client)
            self.enabled = True
            logger.info("✅ LLM client initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation LLM: {e}")
            self.enabled = False

    def query(self, system: str, user: str, temp: float = 0.2) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temp,
                max_tokens=Config.LLM_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"⚠️ LLM query failed: {e}")
            return None

# ============================================================================
# 📊 DATA LOADER
# ============================================================================

class DataLoader:
    def __init__(self):
        logger.info("📁 Chargement des données...")
        self.output_225 = self._load_json(Config.OUTPUT_225)
        logger.info(f"✅ Output 2.25 chargé")
        self.rules_db = self._load_phase1_rules(Config.PHASE1_RULES)
        logger.info(f"✅ {len(self.rules_db)} règles depuis Phase 1")
        self.chunks = self._load_json(Config.PPTX_CHUNKS)
        self.slides_full = self._index_slides_full()
        self.footnotes_by_slide = self._index_footnotes()
        logger.info(f"✅ {len(self.slides_full)} slides indexées")
        self.enriched_context = self._try_load(Config.ENRICHED_CONTEXT)
        self.disclaimers = self._try_load(Config.DISCLAIMERS_JSON)
        self.registration = self._try_load(Config.REGISTRATION_JSON)
        self.doc_context = self._extract_context_corrected()
        logger.info("✅ Toutes les données chargées")
        logger.info(f"📋 Context extrait: {json.dumps(self.doc_context, indent=2, ensure_ascii=False)}\n")

    def _load_json(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _try_load(self, path: str) -> Optional[Dict]:
        try:
            return self._load_json(path)
        except:
            logger.warning(f"⚠️ {path} non trouvé (optionnel)")
            return None

    def _load_phase1_rules(self, path: str) -> Dict[str, Dict]:
        data = self._load_json(path)
        rules_array = data.get('applicable_rules', data.get('rules', []))
        if not rules_array:
            logger.warning("⚠️ Aucune règle trouvée")
            return {}
        return {r['rule_id']: r for r in rules_array}

    def _index_slides_full(self) -> Dict[int, str]:
        slides = {}
        chunks_list = self.chunks if isinstance(self.chunks, list) else self.chunks.get('chunks', [])
        for chunk in chunks_list:
            if chunk.get('chunk_type') == 'slide_full':
                meta = chunk.get('metadata', {})
                slide_num = meta.get('slide_number')
                if slide_num:
                    slides[slide_num] = chunk.get('content', '')
        return slides

    def _index_footnotes(self) -> Dict[int, List[str]]:
        footnotes = {}
        chunks_list = self.chunks if isinstance(self.chunks, list) else self.chunks.get('chunks', [])
        for chunk in chunks_list:
            if chunk.get('chunk_type') == 'footnote':
                meta = chunk.get('metadata', {})
                slide_num = meta.get('slide_number')
                if slide_num:
                    footnotes.setdefault(slide_num, []).append(chunk.get('content', ''))
        return footnotes

    def _extract_context_corrected(self) -> Dict:
        if not self.enriched_context:
            logger.warning("⚠️ Pas de metadata_final_optimized.json")
            return self._extract_from_225()
        metadata = self.enriched_context.get('metadata', {})
        fund_info = self.enriched_context.get('fund_info', {})
        derived = self.enriched_context.get('derived', {})
        client_type = metadata.get('client_type', 'retail')
        is_new_product = metadata.get('is_new_product', False)
        inception_date = fund_info.get('inception_date', '')
        if inception_date and ('2025' in str(inception_date) or 'XX' in str(inception_date)):
            is_new_product = True
        perf_constraints = derived.get('performance_constraints', {})
        can_display_performance = perf_constraints.get('can_display_performance', False)
        if is_new_product:
            can_display_performance = False
        context = {
            'client_type': client_type,
            'fund_name': fund_info.get('fund_name', 'N/A'),
            'fund_type': fund_info.get('fund_type', ''),
            'benchmark': fund_info.get('benchmark', 'N/A'),
            'esg_article': fund_info.get('esg_sfdr_article'),
            'sri_level': fund_info.get('sri_risk_level'),
            'inception_date': inception_date,
            'is_new_product': is_new_product,
            'can_display_performance': can_display_performance,
            'countries': metadata.get('target_countries', []),
            'management_entity': metadata.get('management_entity', 'ODDO BHF AM SAS')
        }
        return context

    def _extract_from_225(self) -> Dict:
        meta_225 = self.output_225.get('metadata', {}).get('document_context', {})
        return {
            'client_type': meta_225.get('client_type', 'retail'),
            'fund_name': meta_225.get('fund_name', 'N/A'),
            'fund_type': meta_225.get('fund_type', ''),
            'benchmark': meta_225.get('benchmark', 'N/A'),
            'is_new_product': meta_225.get('is_new_product', False),
            'can_display_performance': meta_225.get('can_display_performance', False),
            'countries': meta_225.get('target_countries', [])
        }

# ============================================================================
# 🔍 VÉRIFICATEURS SPÉCIALISÉS
# ============================================================================

class BaseVerifier:
    def __init__(self, llm: LLMClient, data: DataLoader):
        self.llm = llm
        self.data = data

    def _parse_json(self, raw: str) -> Dict:
        clean = re.sub(r'```json\s*', '', raw)
        clean = re.sub(r'```\s*', '', clean).strip()
        try:
            return json.loads(clean)
        except:
            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _add_confidence_score(self, result: Dict) -> Dict:
        if not result.get('violation_detected'):
            result['confidence'] = 0.0
            result['classification'] = 'CONFORME'
            return result
        confidence = 0.5
        if result.get('violating_text'):
            confidence += 0.2
        if result.get('explanation') and len(result.get('explanation', '')) > 50:
            confidence += 0.1
        if result.get('elements_missing') and len(result.get('elements_missing', [])) > 0:
            confidence += 0.1
        if result.get('forbidden_phrases_found') and len(result.get('forbidden_phrases_found', [])) > 0:
            confidence += 0.2
        confidence = min(confidence, 1.0)
        result['confidence'] = round(confidence, 2)
        if confidence >= Config.VIOLATION_THRESHOLD:
            result['classification'] = 'VIOLATION'
        else:
            result['classification'] = 'RECOMMANDATION'
        return result

class DisclaimerVerifier(BaseVerifier):
    def verify(self, rule: Dict, slide_num: int, slide_text: str, footnotes: List[str]) -> Dict:
        rule_id = rule['rule_id']
        disclaimer_id = rule.get('requires_disclaimer')
        expected_text = self._get_expected_disclaimer(disclaimer_id)
        all_text = slide_text + "\n\n" + "\n\n".join(footnotes)
        system = """Tu es un expert compliance spécialisé dans les mentions légales (disclaimers).
Réponds UNIQUEMENT en JSON valide (sans markdown, sans ```).
Sois STRICT: une violation n'est détectée QUE si le disclaimer est CLAIREMENT absent ou incomplet."""
        user = f"""RÈGLE {rule_id}: {rule.get('rule_text', '')[:200]}
DISCLAIMER ATTENDU (ID: {disclaimer_id}): {expected_text[:500]}
CONTENU SLIDE {slide_num} + FOOTNOTES: {all_text[:2500]}
Réponds en JSON:
{{"violation_detected": true/false, "disclaimer_found": true/false, "found_text": "texte ou null",
"match_quality": "exact/equivalent/missing/partial", "explanation": "courte explication",
"correction": "correction si violation"}}"""
        raw = self.llm.query(system, user)
        if not raw:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Failed'}
        try:
            result = self._parse_json(raw)
            result.update({'rule_id': rule_id, 'check_type': 'disclaimer', 'severity': rule.get('severity', 'medium')})
            return self._add_confidence_score(result)
        except:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Parse error'}

    def _get_expected_disclaimer(self, disclaimer_id: Optional[str]) -> str:
        if not disclaimer_id or not self.data.disclaimers:
            return "N/A"
        client_type = self.data.doc_context.get('client_type', 'retail')
        disclaimers_data = self.data.disclaimers.get('data', {})
        key = 'non_professional' if client_type == 'retail' else 'professional'
        client_disclaimers = disclaimers_data.get(key, {})
        disclaimer = client_disclaimers.get(disclaimer_id)
        if isinstance(disclaimer, dict):
            return disclaimer.get('fr', disclaimer.get('en', 'N/A'))
        return str(disclaimer) if disclaimer else 'N/A'

class PerformanceVerifier(BaseVerifier):
    def verify(self, rule: Dict, slide_num: int, slide_text: str, footnotes: List[str]) -> Dict:
        rule_id = rule['rule_id']
        if (self.data.doc_context.get('is_new_product', False) and rule_id in Config.NEW_PRODUCT_EXEMPT_RULES):
            logger.info(f"   ⏭️ Règle {rule_id} exemptée (nouveau produit)")
            return {'rule_id': rule_id, 'violation_detected': False, 'exempted': True, 'reason': 'Nouveau produit - règle non applicable', 'confidence': 0.0, 'classification': 'CONFORME'}
        all_text = slide_text + "\n\n" + "\n\n".join(footnotes)
        benchmark = self.data.doc_context.get('benchmark', 'N/A')
        system = """Tu es un expert compliance spécialisé dans les règles de PERFORMANCE.
Réponds UNIQUEMENT en JSON valide (sans markdown, sans ```).
Sois STRICT: une violation n'est détectée QUE si la règle est CLAIREMENT enfreinte.
Les graphiques comparatifs pédagogiques (hors performance du fonds) ne sont PAS des violations."""
        user = f"""RÈGLE {rule_id}: {rule.get('rule_text', '')[:250]}
CONTEXTE: Benchmark={benchmark}, Client={self.data.doc_context.get('client_type')}
Nouveau produit: {self.data.doc_context.get('is_new_product')}
CONTENU SLIDE {slide_num}: {all_text[:2500]}
Réponds en JSON:
{{"violation_detected": true/false, "performance_found": true/false, "benchmark_comparison": true/false,
"disclaimer_present": true/false, "violation_type": "type ou null", "violating_text": "texte exact",
"explanation": "explication", "correction": "correction"}}"""
        raw = self.llm.query(system, user)
        if not raw:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Failed'}
        try:
            result = self._parse_json(raw)
            result.update({'rule_id': rule_id, 'check_type': 'performance', 'severity': rule.get('severity', 'high')})
            return self._add_confidence_score(result)
        except:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Parse error'}

class CitationVerifier(BaseVerifier):
    def verify(self, rule: Dict, slide_num: int, slide_text: str, footnotes: List[str]) -> Dict:
        rule_id = rule['rule_id']
        system = """Tu es un expert compliance spécialisé dans les CITATIONS et SOURCES.
Réponds UNIQUEMENT en JSON valide (sans markdown, sans ```).
Sois STRICT: une violation n'est détectée QUE si des données chiffrées ou graphiques MANQUENT clairement de source."""
        user = f"""RÈGLE {rule_id}: {rule.get('rule_text', '')}
CONTENU SLIDE {slide_num}: {slide_text[:2000]}
FOOTNOTES: {chr(10).join(footnotes) if footnotes else "Aucune"}
Réponds en JSON:
{{"violation_detected": true/false, "elements_requiring_citation": [], "citations_found": [],
"citations_missing": [], "violating_text": "description", "explanation": "explication",
"correction": "correction"}}"""
        raw = self.llm.query(system, user)
        if not raw:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Failed'}
        try:
            result = self._parse_json(raw)
            result.update({'rule_id': rule_id, 'check_type': 'citation', 'severity': rule.get('severity', 'medium')})
            return self._add_confidence_score(result)
        except:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Parse error'}

class LinguisticVerifier(BaseVerifier):
    FORBIDDEN = ["sous-évalué", "surévalué", "acheter", "vendre", "recommandons", "opportunité d'achat"]

    def verify(self, rule: Dict, slide_num: int, slide_text: str) -> Dict:
        rule_id = rule['rule_id']
        system = """Tu es un expert compliance spécialisé dans le LANGAGE MARKETING FINANCIER.
Réponds UNIQUEMENT en JSON valide (sans markdown, sans ```).
Sois STRICT: une violation n'est détectée QUE si des termes INTERDITS sont CLAIREMENT utilisés."""
        user = f"""RÈGLE {rule_id}: {rule.get('rule_text', '')}
EXEMPLES INTERDITS: {', '.join(self.FORBIDDEN)}
CONTENU SLIDE {slide_num}: {slide_text[:2500]}
Réponds en JSON:
{{"violation_detected": true/false, "forbidden_phrases_found": [], "violating_text": "texte exact",
"violation_type": "type", "explanation": "explication", "correction": "correction"}}"""
        raw = self.llm.query(system, user)
        if not raw:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Failed'}
        try:
            result = self._parse_json(raw)
            result.update({'rule_id': rule_id, 'check_type': 'linguistic', 'severity': rule.get('severity', 'high')})
            return self._add_confidence_score(result)
        except:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Parse error'}

class StructureVerifier(BaseVerifier):
    def verify(self, rule: Dict, slide_num: int, slide_text: str, slide_title: str) -> Dict:
        rule_id = rule['rule_id']
        check_type = rule.get('check_type', '')
        if 'page_garde' in check_type:
            required = ["nom du fonds", "mois et année", "document promotionnel", "cible"]
        elif 'slide_2' in check_type:
            required = ["disclaimer standard", "profil de risque"]
        else:
            required = []
        system = """Tu es un expert compliance spécialisé dans la STRUCTURE.
Réponds UNIQUEMENT en JSON valide (sans markdown, sans ```).
Sois STRICT: une violation n'est détectée QUE si des éléments OBLIGATOIRES sont CLAIREMENT absents."""
        user = f"""RÈGLE {rule_id}: {rule.get('rule_text', '')}
ÉLÉMENTS OBLIGATOIRES: {', '.join(required)}
SLIDE {slide_num}: {slide_title}
CONTENU: {slide_text[:2500]}
Réponds en JSON:
{{"violation_detected": true/false, "elements_found": [], "elements_missing": [],
"violating_text": "description", "explanation": "explication", "correction": "correction"}}"""
        raw = self.llm.query(system, user)
        if not raw:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Failed'}
        try:
            result = self._parse_json(raw)
            result.update({'rule_id': rule_id, 'check_type': 'structure', 'severity': rule.get('severity', 'high')})
            return self._add_confidence_score(result)
        except:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Parse error'}

class ProspectusVerifier(BaseVerifier):
    def verify(self, rule: Dict, slide_num: int, slide_text: str) -> Dict:
        rule_id = rule['rule_id']
        benchmark = self.data.doc_context.get('benchmark', 'N/A')
        system = """Tu es un expert compliance spécialisé dans la COHÉRENCE avec le PROSPECTUS.
Réponds UNIQUEMENT en JSON valide (sans markdown, sans ```).
Sois STRICT: une incohérence n'est détectée QUE si l'information est CLAIREMENT contradictoire."""
        user = f"""RÈGLE {rule_id}: {rule.get('rule_text', '')}
BENCHMARK OFFICIEL: {benchmark}
CONTENU SLIDE {slide_num}: {slide_text[:2500]}
Réponds en JSON:
{{"violation_detected": true/false, "incoherences_found": [], "violating_text": "texte",
"explanation": "explication", "correction": "correction"}}"""
        raw = self.llm.query(system, user)
        if not raw:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Failed'}
        try:
            result = self._parse_json(raw)
            result.update({'rule_id': rule_id, 'check_type': 'prospectus_coherence', 'severity': rule.get('severity', 'high')})
            return self._add_confidence_score(result)
        except:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Parse error'}

class RegistrationVerifier(BaseVerifier):
    def verify(self, rule: Dict, slide_num: int, slide_text: str) -> Dict:
        rule_id = rule['rule_id']
        authorized = []
        if self.data.registration:
            flat_results = self.data.registration.get('flat_results', {})
            all_countries = set()
            for fund_data in flat_results.values():
                countries = fund_data.get('registered_countries', [])
                all_countries.update(countries)
            authorized = sorted(list(all_countries))
        system = """Tu es un expert compliance spécialisé dans les PAYS DE COMMERCIALISATION.
Réponds UNIQUEMENT en JSON valide (sans markdown, sans ```).
Sois STRICT: une violation n'est détectée QUE si des pays NON AUTORISÉS sont CLAIREMENT mentionnés."""
        user = f"""RÈGLE {rule_id}: {rule.get('rule_text', '')}
PAYS AUTORISÉS: {', '.join(authorized) if authorized else 'Non disponible'}
CONTENU SLIDE {slide_num}: {slide_text[:2500]}
Réponds en JSON:
{{"violation_detected": true/false, "countries_mentioned": [], "unauthorized_countries": [],
"violating_text": "texte", "explanation": "explication", "correction": "correction"}}"""
        raw = self.llm.query(system, user)
        if not raw:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Failed'}
        try:
            result = self._parse_json(raw)
            result.update({'rule_id': rule_id, 'check_type': 'registration', 'severity': rule.get('severity', 'high')})
            return self._add_confidence_score(result)
        except:
            return {'rule_id': rule_id, 'violation_detected': None, 'error': 'Parse error'}

# ============================================================================
# 🎯 ROUTER
# ============================================================================

class VerifierRouter:
    def __init__(self, llm: LLMClient, data: DataLoader):
        self.disclaimer = DisclaimerVerifier(llm, data)
        self.performance = PerformanceVerifier(llm, data)
        self.citation = CitationVerifier(llm, data)
        self.linguistic = LinguisticVerifier(llm, data)
        self.structure = StructureVerifier(llm, data)
        self.prospectus = ProspectusVerifier(llm, data)
        self.registration = RegistrationVerifier(llm, data)

    def route(self, rule: Dict, slide_num: int, slide_text: str, footnotes: List[str], slide_title: str) -> Dict:
        check_type = rule.get('check_type', '').lower()
        if 'disclaimer' in check_type:
            return self.disclaimer.verify(rule, slide_num, slide_text, footnotes)
        elif 'performance' in check_type:
            return self.performance.verify(rule, slide_num, slide_text, footnotes)
        elif 'citation' in check_type:
            return self.citation.verify(rule, slide_num, slide_text, footnotes)
        elif 'linguistic' in check_type:
            return self.linguistic.verify(rule, slide_num, slide_text)
        elif any(x in check_type for x in ['page_garde', 'slide_2', 'structure']):
            return self.structure.verify(rule, slide_num, slide_text, slide_title)
        elif 'prospectus' in check_type:
            return self.prospectus.verify(rule, slide_num, slide_text)
        elif 'registration' in check_type or 'countries' in check_type:
            return self.registration.verify(rule, slide_num, slide_text)
        else:
            return self.structure.verify(rule, slide_num, slide_text, slide_title)

# ============================================================================
# 🔍 VÉRIFICATEUR GLOBAL
# ============================================================================

class GlobalRuleVerifier:
    def __init__(self, llm: LLMClient, data: DataLoader):
        self.llm = llm
        self.data = data

    def verify_across_document(self, rule: Dict, all_slides: Dict[int, str]) -> Tuple[bool, Optional[int], str]:
        rule_id = rule['rule_id']
        rule_text = rule.get('rule_text', '')
        all_content = "\n\n---SLIDE SEPARATOR---\n\n".join(
            f"SLIDE {num}: {content[:1000]}" for num, content in sorted(all_slides.items())
        )
        system = """Tu es un expert compliance spécialisé dans les RÈGLES GÉNÉRALES.
Ton rôle: vérifier si l'information requise est présente QUELQUE PART dans le document.
Réponds UNIQUEMENT en JSON valide (sans markdown, sans ```)."""
        user = f"""RÈGLE {rule_id}: {rule_text}
CONTENU COMPLET DU DOCUMENT (toutes les slides):
{all_content[:5000]}
Réponds en JSON:
{{"found": true/false, "slide_number": numéro ou null, "explanation": "où l'info a été trouvée ou pourquoi elle manque"}}"""
        raw = self.llm.query(system, user, temp=0.1)
        if not raw:
            return (False, None, "Erreur LLM")
        try:
            clean = re.sub(r'```json\s*', '', raw)
            clean = re.sub(r'```\s*', '', clean).strip()
            result = json.loads(clean)
            found = result.get('found', False)
            slide_num = result.get('slide_number')
            explanation = result.get('explanation', '')
            return (found, slide_num, explanation)
        except:
            return (False, None, "Erreur parsing")

# ============================================================================
# 🎯 ORCHESTRATEUR
# ============================================================================

class Phase23Orchestrator:
    def __init__(self):
        logger.info("\n" + "="*80)
        logger.info("🚀 PHASE 2.3 - VÉRIFICATION OPTIMISÉE")
        logger.info("="*80 + "\n")
        self.llm = LLMClient()
        self.data = DataLoader()
        self.router = VerifierRouter(self.llm, self.data)
        self.global_verifier = GlobalRuleVerifier(self.llm, self.data)

    # ✅ AJOUT: nettoyage range rules (sans changer le reste)
    def _cleanup_range_rule_violations(self, results: Dict, range_ok_slides: Dict[str, set]) -> Dict:
        max_slide = max(self.data.slides_full.keys()) if self.data.slides_full else 0

        for rule_id, ok_slides in (range_ok_slides or {}).items():
            if not ok_slides:
                continue  # aucune preuve de conformité => on ne touche rien
            if rule_id not in Config.RANGE_RULES:
                continue

            ok_slides_sorted = sorted(list(ok_slides))

            for slide_entry in results.get('slides_analysis', []):
                sn = slide_entry.get('slide_number')
                if sn is None:
                    continue
                if not _in_range(rule_id, sn, max_slide):
                    continue

                before = len(slide_entry.get('violations', []))
                if before == 0:
                    continue

                slide_entry['violations'] = [
                    v for v in slide_entry.get('violations', [])
                    if v.get('rule_id') != rule_id
                ]
                after = len(slide_entry.get('violations', []))
                removed = before - after

                if removed > 0:
                    # mettre à jour le compteur local
                    slide_entry['violations_count'] = after

                    # (optionnel) trace d’audit
                    slide_entry.setdefault('range_rules_cleanup', []).append({
                        'rule_id': rule_id,
                        'removed': removed,
                        'because_verified_on_slides': ok_slides_sorted
                    })

        return results

    def run(self) -> Dict:
        slides_from_225 = self.data.output_225.get('slides_analysis', [])
        logger.info(f"📊 {len(slides_from_225)} slides à vérifier\n")
        results = {
            'metadata': {
                'phase': '2.3_optimized_verification',
                'timestamp': datetime.now().isoformat(),
                'fund_name': self.data.doc_context.get('fund_name', 'N/A'),
                'document_context': self.data.doc_context,
                'source': 'Output 2.25 (filtered rules)',
                'improvements': ['Classification VIOLATIONS vs RECOMMANDATIONS', 'Vérification globale pour règles 1.x', 'Gestion adaptée des nouveaux produits', 'Extraction corrigée des métadonnées']
            },
            'slides_analysis': []
        }

        # ✅ ÉTAPE 1: Vérifier les règles globales (1.x) une seule fois
        global_rules_status = self._verify_global_rules()

        # ✅ AJOUT: preuves "OK" pour les range rules
        max_slide = max(self.data.slides_full.keys()) if self.data.slides_full else 0
        range_ok_slides = {rid: set() for rid in Config.RANGE_RULES.keys()}

        # ✅ ÉTAPE 2: Vérifier slide par slide
        for slide_data in tqdm(slides_from_225, desc="Vérification slides"):
            slide_num = slide_data.get('slide_number')
            slide_title = slide_data.get('slide_title', '')
            selected_rules = slide_data.get('selected_rules', [])

            logger.info(f"\n{'─'*80}")
            logger.info(f"📄 Slide {slide_num}: {slide_title}")

            if slide_num == 1:
                original_count = len(selected_rules)
                selected_rules = [r for r in selected_rules if r.get('rule_id') in Config.SLIDE_1_ALLOWED_RULES]
                if original_count > len(selected_rules):
                    logger.info(f"   🔒 Slide 1 filtrée: {original_count} → {len(selected_rules)} règles")

            # ✅ Filtrer les règles globales déjà vérifiées
            local_rules = []
            for r in selected_rules:
                rule_id = r.get('rule_id')
                if rule_id in Config.GLOBAL_RULES:
                    global_status = global_rules_status.get(rule_id, {})
                    if global_status.get('found'):
                        logger.info(f"   ✅ {rule_id} (OK globalement sur slide {global_status.get('slide_number')})")
                        continue
                local_rules.append(r)

            logger.info(f"🔍 {len(local_rules)} règles à vérifier localement")

            slide_text = self.data.slides_full.get(slide_num, "")
            footnotes = self.data.footnotes_by_slide.get(slide_num, [])

            if not slide_text:
                logger.warning(f"⚠️ Slide {slide_num} sans contenu, ignorée")
                continue

            violations = []
            recommendations = []

            for rule_summary in local_rules:
                rule_id = rule_summary.get('rule_id')
                rule_full = self.data.rules_db.get(rule_id)

                if not rule_full:
                    logger.warning(f"⚠️ Règle {rule_id} non trouvée")
                    continue

                check_type = rule_full.get('check_type', 'unknown')
                logger.info(f"  🔎 {rule_id} ({check_type})")

                verification = self.router.route(rule_full, slide_num, slide_text, footnotes, slide_title)

                if verification.get('exempted'):
                    logger.info(f"    ⏭️ EXEMPTÉE: {verification.get('reason')}")
                    continue

                # ✅ AJOUT: si règle de plage + slide dans la plage + OK => on garde la preuve
                if rule_id in Config.RANGE_RULES and _in_range(rule_id, slide_num, max_slide):
                    if verification.get('violation_detected') is False:
                        range_ok_slides[rule_id].add(slide_num)

                if verification.get('violation_detected'):
                    classification = verification.get('classification', 'RECOMMANDATION')
                    confidence = verification.get('confidence', 0.0)

                    if classification == 'VIOLATION':
                        violations.append(verification)
                        logger.warning(f"    ❌ VIOLATION (conf: {confidence}): {verification.get('violation_type', 'unknown')}")
                    else:
                        recommendations.append(verification)
                        logger.info(f"    ⚠️ RECOMMANDATION (conf: {confidence})")
                elif verification.get('violation_detected') is False:
                    logger.info(f"    ✅ OK")
                else:
                    logger.warning(f"    ⚠️ Erreur vérification")

            # ✅ Ajouter violations des règles globales si non trouvées
            for rule_id in Config.GLOBAL_RULES:
                if rule_id in [r.get('rule_id') for r in selected_rules]:
                    global_status = global_rules_status.get(rule_id, {})
                    if not global_status.get('found'):
                        violations.append({
                            'rule_id': rule_id,
                            'violation_detected': True,
                            'check_type': 'global_rule',
                            'severity': 'high',
                            'explanation': global_status.get('explanation', 'Non trouvé dans le document'),
                            'confidence': 0.9,
                            'classification': 'VIOLATION',
                            'scope': 'document'
                        })

            results['slides_analysis'].append({
                'slide_number': slide_num,
                'slide_title': slide_title,
                'rules_checked': len(local_rules),
                'violations_count': len(violations),
                'recommendations_count': len(recommendations),
                'violations': violations,
                'recommendations': recommendations
            })

        # ✅ AJOUT: post-processing range rules (suppression des violations répétées)
        results = self._cleanup_range_rule_violations(results, range_ok_slides)

        # ✅ STATISTIQUES FINALES
        all_violations = []
        all_recommendations = []
        for s in results['slides_analysis']:
            all_violations.extend(s['violations'])
            all_recommendations.extend(s['recommendations'])

        violations_by_severity = Counter(v.get('severity', 'unknown') for v in all_violations)
        violations_by_type = Counter(v.get('check_type', 'unknown') for v in all_violations)
        violations_by_rule = Counter(v['rule_id'] for v in all_violations)

        results['global_statistics'] = {
            'total_slides': len(results['slides_analysis']),
            'total_violations': len(all_violations),
            'total_recommendations': len(all_recommendations),
            'slides_with_violations': sum(1 for s in results['slides_analysis'] if s['violations_count'] > 0),
            'slides_with_recommendations': sum(1 for s in results['slides_analysis'] if s['recommendations_count'] > 0),
            'slides_clean': sum(1 for s in results['slides_analysis'] if s['violations_count'] == 0),
            'violations_by_severity': dict(violations_by_severity),
            'violations_by_type': dict(violations_by_type),
            'top_violated_rules': [{'rule_id': rid, 'count': count} for rid, count in violations_by_rule.most_common(10)],
            'global_rules_status': global_rules_status
        }

        with open(Config.OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ PHASE 2.3 TERMINÉE")
        logger.info(f"📊 {len(all_violations)} VIOLATIONS (haute confiance)")
        logger.info(f"📊 {len(all_recommendations)} RECOMMANDATIONS (moyenne confiance)")
        logger.info(f"💾 Sauvegardé: {Config.OUTPUT_JSON}")
        logger.info(f"{'='*80}\n")

        return results

    def _verify_global_rules(self) -> Dict[str, Dict]:
        """Vérifie toutes les règles globales (1.x) une seule fois"""
        logger.info("\n" + "="*80)
        logger.info("🌍 VÉRIFICATION DES RÈGLES GLOBALES (1.x)")
        logger.info("="*80 + "\n")

        global_status = {}

        for rule_id in Config.GLOBAL_RULES:
            rule = self.data.rules_db.get(rule_id)
            if not rule:
                continue

            logger.info(f"🔍 Règle {rule_id}: {rule.get('rule_text', '')[:80]}...")

            found, slide_num, explanation = self.global_verifier.verify_across_document(rule, self.data.slides_full)

            global_status[rule_id] = {
                'found': found,
                'slide_number': slide_num,
                'explanation': explanation
            }

            if found:
                logger.info(f"   ✅ Trouvé sur slide {slide_num}")
            else:
                logger.warning(f"   ❌ NON TROUVÉ: {explanation}")

        logger.info(f"\n{'='*80}\n")
        return global_status

# ============================================================================
# 📊 RAPPORT
# ============================================================================

def print_report(results: Dict):
    stats = results['global_statistics']
    print("\n" + "="*80)
    print("📊 RAPPORT FINAL - PHASE 2.3 OPTIMISÉE")
    print("="*80)
    print(f"\n📄 Document: {results['metadata']['fund_name']}")
    print(f"📅 Date: {results['metadata']['timestamp'][:19]}")
    context = results['metadata']['document_context']
    print(f"\n📋 CONTEXTE")
    print(f"   • Type client: {context.get('client_type')}")
    print(f"   • Nouveau produit: {'OUI' if context.get('is_new_product') else 'NON'}")
    print(f"   • Date inception: {context.get('inception_date', 'N/A')}")
    print(f"   • Peut afficher performance: {'OUI' if context.get('can_display_performance') else 'NON'}")

    print(f"\n🔍 SLIDES")
    print(f"   Total analysées: {stats['total_slides']}")
    print(f"   ✅ Conformes: {stats['slides_clean']}")
    print(f"   ⚠️ Avec violations: {stats['slides_with_violations']}")
    print(f"   💡 Avec recommandations: {stats['slides_with_recommendations']}")

    print(f"\n⚠️ RÉSULTATS")
    print(f"   🔴 VIOLATIONS (haute confiance): {stats['total_violations']}")
    print(f"   💡 RECOMMANDATIONS (moyenne confiance): {stats['total_recommendations']}")

    if stats['total_violations'] > 0:
        print(f"\n   Par sévérité:")
        for sev, count in sorted(stats['violations_by_severity'].items()):
            emoji = "🔴" if sev == "high" else "🟠" if sev == "medium" else "🟡"
            print(f"      {emoji} {sev.upper()}: {count}")

        print(f"\n   Par type:")
        for vtype, count in sorted(stats['violations_by_type'].items())[:5]:
            print(f"      • {vtype}: {count}")

        print(f"\n   Top 5 règles violées:")
        for i, item in enumerate(stats['top_violated_rules'][:5], 1):
            print(f"      {i}. {item['rule_id']}: {item['count']} fois")

    print(f"\n🌍 RÈGLES GLOBALES (1.x)")
    global_rules = stats.get('global_rules_status', {})
    ok_count = sum(1 for v in global_rules.values() if v.get('found'))
    total_count = len(global_rules)
    print(f"   ✅ Conformes: {ok_count}/{total_count}")
    for rule_id, status in sorted(global_rules.items()):
        emoji = "✅" if status.get('found') else "❌"
        slide_info = f" (slide {status.get('slide_number')})" if status.get('found') else ""
        print(f"      {emoji} {rule_id}{slide_info}")

    print("\n" + "="*80)

# ============================================================================
# ▶️ MAIN
# ============================================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║        PHASE 2.3 - VÉRIFICATION OPTIMISÉE (VIOLATIONS vs RECO)    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    try:
        orchestrator = Phase23Orchestrator()
        results = orchestrator.run()
        print_report(results)
        print(f"\n✅ Phase 2.3 terminée avec succès!")
        print(f"📁 Fichier de sortie: {Config.OUTPUT_JSON}\n")
        return results
    except FileNotFoundError as e:
        print(f"\n❌ ERREUR: Fichier manquant - {e}")
        print(f"\n📋 Vérifiez que ces fichiers existent:")
        print(f"   - {Config.OUTPUT_225}")
        print(f"   - {Config.PHASE1_RULES}")
        print(f"   - {Config.PPTX_CHUNKS}")
        return None
    except KeyboardInterrupt:
        print("\n⚠️ Interruption par l'utilisateur")
        return None
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()