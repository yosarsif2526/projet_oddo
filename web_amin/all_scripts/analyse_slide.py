# Generated from: analyse slide (2).ipynb
# Converted at: 2025-12-16T19:46:40.054Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell


"""
PHASE 2.1 - FIX DÉTECTION DISCLAIMERS (MATCHING HYBRIDE)
=========================================================

STRATÉGIE:
1. Extraire les footnotes/disclaimers de la slide (regex patterns)
2. Matcher avec patterns textuels forts (performance, risk, source)
3. Fallback: embeddings locaux sur les extraits (pas toute la slide)

INPUTS:
- pptx_slide_full.json
- metadata_final_optimized_test.json
- phase1_correct_output.json
- disclamer_bilingual_full.json
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import httpx
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION
# ============================================================

def get_llm_client():
    http_client = httpx.Client(verify=False)
    return OpenAI(
        api_key="sk-7c0b80cf494746f580cc5ba555d739b2",
        base_url="https://tokenfactory.esprit.tn/api",
        http_client=http_client
    )

print("⏳ Chargement modèle embeddings multilingue...")
EMBEDDING_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✓ Modèle embeddings chargé")

def get_embedding(text: str) -> List[float]:
    text_truncated = text[:2000]
    try:
        embedding = EMBEDDING_MODEL.encode(text_truncated, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"⚠️ Erreur embedding: {e}")
        return [0.0] * 384

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

# ============================================================
# PATTERNS DISCLAIMERS (REGEX)
# ============================================================

DISCLAIMER_PATTERNS = {
    "performance": [
        r"past performance.*not.*reliable.*indicator.*future",
        r"performances? passées? ne présagent? pas",
        r"performances? passées?.*constantes? dans le temps",
        r"past performance.*not constant over time",
    ],
    "simulations_future_performance": [
        r"simulation.*does not constitute.*forecast",
        r"simulation.*ne constitue pas.*prévision",
        r"deviate.*upwards.*downwards",
        r"investissement pourra s'écarter",
    ],
    "citation_source": [
        r"source\s*:\s*[A-Z]",  # "Source: Bloomberg"
        r"data as of \d{2}/\d{2}/\d{4}",
        r"données au \d{2}/\d{2}/\d{4}",
    ],
    "sri_in_marketing_document": [
        r"summary risk indicator.*SRI",
        r"indicateur synthétique de risque.*SRI",
        r"risk profile.*fund changes over time",
    ],
    "esg_risk": [
        r"ESG.*affect.*performance",
        r"critères ESG.*affecter",
        r"sustainability.*differ.*similar funds",
    ],
    "opinion": [
        r"opinions.*market forecasts.*publication date",
        r"opinions.*anticipations de marché",
        r"subject to change.*market conditions",
    ],
    "issuers_mentioned": [
        r"does not constitute.*investment recommendation",
        r"ne constitue pas.*recommandation",
        r"examples.*no investment recommendations",
    ],
}

# ============================================================
# EXTRACTEUR FOOTNOTES/DISCLAIMERS
# ============================================================

def extract_potential_disclaimers(slide_content: str) -> List[Tuple[str, str]]:
    """
    Extrait les portions de texte susceptibles de contenir des disclaimers.
    Retourne: [(type, texte), ...]
    """
    extracts = []
    
    if not slide_content:
        return extracts
    
    # 1. Notes de bas de page (après "Notes:", "Source:", "Data as of")
    notes_match = re.search(r"(?:Notes?|Source|Data as of|Données au)[\s:]+(.+)", 
                            slide_content, re.IGNORECASE | re.DOTALL)
    if notes_match:
        notes_text = notes_match.group(1).strip()
        extracts.append(("footnote", notes_text[:1000]))  # limite 1000 chars
    
    # 2. Phrases avec "past performance", "risk", "disclaimer"
    for pattern in [
        r"([^.]*(?:past performance|performances passées)[^.]{0,200}\.)",
        r"([^.]*(?:risk|risque)[^.]{0,200}\.)",
        r"([^.]*(?:disclaimer|avertissement)[^.]{0,200}\.)",
        r"([^.]*(?:simulation|forecast|prévision)[^.]{0,200}\.)",
        r"([^.]*(?:opinion|recommendation)[^.]{0,200}\.)",
    ]:
        matches = re.findall(pattern, slide_content, re.IGNORECASE)
        for match in matches:
            extracts.append(("sentence", match.strip()))
    
    # 3. Si aucune extraction, prendre les 500 derniers caractères (souvent disclaimers)
    if not extracts:
        footer = slide_content[-500:].strip()
        if footer:
            extracts.append(("footer", footer))
    
    return extracts

# ============================================================
# ANALYSEUR HYBRIDE
# ============================================================

class SlideAnalyzerHybrid:
    """
    Analyse slides avec matching hybride regex + embeddings ciblés.
    """
    
    SEMANTIC_THRESHOLD = 0.75  # seuil embeddings sur extraits
    
    def __init__(self,
                 chunks_slide_full_path: str,
                 metadata_path: str,
                 phase1_rules_path: str,
                 disclaimers_path: str):
        
        print("="*70)
        print("🚀 PHASE 2.1 - FIX DISCLAIMERS (HYBRID)")
        print("="*70)
        
        # LLM
        print("\n🤖 Initialisation LLM...")
        self.llm = get_llm_client()
        print("   ✓ Client LLM connecté")
        
        # Chunks
        print("\n📦 Chargement chunks slides...")
        with open(chunks_slide_full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.slides = data if isinstance(data, list) else data.get('chunks', data)
        print(f"   ✓ {len(self.slides)} slides")
        
        # Metadata
        print("\n📁 Chargement metadata...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        compliance = self.metadata.get('compliance', {})
        derived = compliance.get('derived', self.metadata.get('derived', {}))
        self.client_type = derived.get('client_type', 'retail')
        self.fund_info = self.metadata.get('fund_info', {})
        self.fund_name = self.fund_info.get('fund_name', 'N/A')
        print(f"   ✓ Client: {self.client_type}")
        
        # Règles
        print("\n📋 Chargement règles...")
        with open(phase1_rules_path, 'r', encoding='utf-8') as f:
            phase1_data = json.load(f)
        self.applicable_rules = phase1_data.get('applicable_rules', [])
        print(f"   ✓ {len(self.applicable_rules)} règles")
        
        # Disclaimers
        print("\n📝 Chargement disclaimers...")
        with open(disclaimers_path, 'r', encoding='utf-8') as f:
            disclaimers_raw = json.load(f)
        disclaimers_data = disclaimers_raw.get('data', disclaimers_raw)
        self.disclaimers_raw = disclaimers_data.get(
            self.client_type,
            disclaimers_data.get('non_professional', {})
        )
        
        # Build embeddings + patterns
        print("   ⏳ Préparation disclaimers...")
        self.disclaimer_configs = self._build_disclaimer_configs()
        print(f"   ✓ {len(self.disclaimer_configs)} disclaimers prêts")
        
        # Mapping règles
        self.rules_by_check_type = {}
        for rule in self.applicable_rules:
            ct = rule.get('check_type', 'other')
            self.rules_by_check_type.setdefault(ct, []).append(rule)
        
        print("\n✅ Initialisation terminée")
    
    def _build_disclaimer_configs(self) -> Dict[str, Dict]:
        """Build config avec patterns + embeddings"""
        result = {}
        
        for did, val in self.disclaimers_raw.items():
            if isinstance(val, str):
                text_fr = text_en = val
            elif isinstance(val, dict):
                text_fr = val.get('fr', '')
                text_en = val.get('en', '')
            else:
                continue
            
            text_combined = f"{text_fr} {text_en}".strip()
            if not text_combined:
                continue
            
            # Patterns regex
            patterns = DISCLAIMER_PATTERNS.get(did, [])
            
            # Embedding
            embedding = get_embedding(text_combined)
            
            result[did] = {
                "id": did,
                "text_fr": text_fr,
                "text_en": text_en,
                "patterns": patterns,
                "embedding": embedding
            }
        
        return result
    
    # ==========================
    # ANALYSE SÉMANTIQUE (LLM)
    # ==========================
    
    def analyze_slide_semantic(self, slide_content: str, slide_num: int) -> Dict:
        content_truncated = slide_content[:2500]
        
        prompt = f"""Tu es un expert compliance marketing français.

Analyse cette slide et réponds EN FRANÇAIS, en JSON valide (sans markdown):

SLIDE {slide_num}:
{content_truncated}

INSTRUCTIONS:
- "performance_values" = true UNIQUEMENT si chiffres de performance du FONDS/INDICE
- "benchmark_comparison" = true si comparaison fonds vs indice
- "contains_opinions" = true si opinions/projections/scénarios

Format JSON STRICT (français):
{{
  "slide_number": {slide_num},
  "resume_slide": "Résumé 1-2 phrases",
  "semantic_themes": ["theme1", "theme2"],
  "visual_elements": {{
    "has_chart": false,
    "chart_type": "",
    "has_table": false,
    "has_images": false
  }},
  "data_presented": {{
    "performance_values": false,
    "benchmark_comparison": false,
    "time_period": "",
    "contains_opinions": false
  }},
  "disclaimers_present": {{
    "source_citation": false,
    "performance_warning": false,
    "risk_disclaimer": false,
    "legal_mentions": false
  }}
}}"""
        
        try:
            response = self.llm.chat.completions.create(
                model="hosted_vllm/Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system", "content": "Expert compliance français. JSON uniquement."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=900,
                top_p=0.9
            )
            
            raw = response.choices[0].message.content
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*", "", raw).strip()
            analysis = json.loads(raw)
            
            analysis.setdefault("slide_number", slide_num)
            analysis.setdefault("resume_slide", "")
            analysis.setdefault("semantic_themes", [])
            
            return analysis
            
        except Exception as e:
            print(f"\n⚠️ Erreur LLM slide {slide_num}: {e}")
            return {
                "slide_number": slide_num,
                "resume_slide": "Erreur",
                "semantic_themes": [],
                "visual_elements": {},
                "data_presented": {},
                "disclaimers_present": {},
                "error": True
            }
    
    # ==========================
    # DÉTECTION HYBRIDE
    # ==========================
    
    def _detect_present_disclaimers(self, slide_content: str) -> Dict[str, Dict]:
        """
        Matching hybride:
        1. Regex patterns (rapide, précis)
        2. Embeddings sur extraits (pas toute la slide)
        """
        present = {}
        
        # Extraire portions disclaimer
        extracts = extract_potential_disclaimers(slide_content)
        
        if not extracts:
            return present
        
        # Pour chaque disclaimer
        for did, cfg in self.disclaimer_configs.items():
            best_score = 0.0
            best_context = ""
            match_type = None
            
            # 1. REGEX PATTERNS
            patterns = cfg.get("patterns", [])
            for pattern in patterns:
                for extract_type, extract_text in extracts:
                    if re.search(pattern, extract_text, re.IGNORECASE):
                        best_score = 1.0  # match parfait
                        best_context = extract_text[:200]
                        match_type = f"regex_{extract_type}"
                        break
                if best_score == 1.0:
                    break
            
            # 2. EMBEDDINGS (seulement si pas de regex match)
            if best_score < 1.0:
                for extract_type, extract_text in extracts:
                    extract_emb = get_embedding(extract_text)
                    score = cosine_similarity(extract_emb, cfg["embedding"])
                    
                    if score > best_score:
                        best_score = score
                        best_context = extract_text[:200]
                        match_type = f"embedding_{extract_type}"
            
            # Seuil
            threshold = 0.85 if match_type and "regex" in match_type else self.SEMANTIC_THRESHOLD
            
            if best_score >= threshold:
                present[did] = {
                    "score": round(best_score, 3),
                    "context": best_context,
                    "match_type": match_type
                }
        
        return present
    
    def _get_required_disclaimers(self, analysis: Dict, slide_content: str) -> List[str]:
        """Détermine disclaimers requis"""
        required = []
        
        data = analysis.get('data_presented', {})
        visual = analysis.get('visual_elements', {})
        themes = analysis.get('semantic_themes', [])
        disc = analysis.get('disclaimers_present', {})
        
        text_lower = (slide_content or "").lower()
        themes_str = " ".join(themes).lower()
        
        # Performance
        if data.get('performance_values', False) and not disc.get('performance_warning', False):
            required.append("performance")
        
        # Simulations
        if "simulation" in themes_str or "scenario" in themes_str or "projection" in themes_str:
            required.append("simulations_future_performance")
        
        # Citation
        if visual.get("has_chart", False) and not disc.get("source_citation", False):
            required.append("citation_source")
        
        # SRI
        if "sri" in text_lower or "summary risk indicator" in text_lower:
            required.append("sri_in_marketing_document")
        
        # ESG
        esg_article = self.fund_info.get("esg_sfdr_article", "")
        if "article 8" in esg_article.lower() or "article 9" in esg_article.lower():
            if "esg" in themes_str or "durable" in themes_str:
                required.append("esg_risk")

        # Émetteurs
        if visual.get("has_images", False) or "company" in themes_str:
            required.append("issuers_mentioned")
        
        return list(dict.fromkeys(required))
    
    def check_disclaimers(self, slide_content: str, semantic_analysis: Dict) -> Dict:
        """Vérification complète"""
        result = {
            "required": [],
            "present": [],
            "found": [],
            "missing": [],
            "scores": {},
            "found_details": {}
        }
        
        # Détection hybride
        present_map = self._detect_present_disclaimers(slide_content)
        present_ids = list(present_map.keys())
        
        # Requis
        required_ids = self._get_required_disclaimers(semantic_analysis, slide_content)
        
        # Found = requis ∩ présent
        found_ids = [did for did in required_ids if did in present_ids]
        missing_ids = [did for did in required_ids if did not in present_ids]
        
        result["required"] = required_ids
        result["found"] = found_ids
        
        # Present
        for did, info in present_map.items():
            result["present"].append({
                "id": did,
                "score": info["score"],
                "context": info["context"],
                "match_type": info["match_type"]
            })
            result["scores"][did] = info["score"]
        
        # Found details
        for did in found_ids:
            info = present_map.get(did)
            if info:
                result["found_details"][did] = {
                    "score": info["score"],
                    "context": info["context"],
                    "match_type": info["match_type"]
                }
        
        # Missing
        for did in missing_ids:
            cfg = self.disclaimer_configs.get(did)
            if not cfg:
                continue
            text_fr = cfg.get("text_fr", "")
            short_text = text_fr[:150] + "..." if len(text_fr) > 150 else text_fr
            result["missing"].append({
                "id": did,
                "text": short_text,
                "location": self._get_disclaimer_location(did),
                "mandatory": True
            })
        
        return result
    
    def _get_disclaimer_location(self, disclaimer_id: str) -> str:
        locations = {
            "performance": "near_performance",
            "simulations_future_performance": "near_projection",
            "citation_source": "near_chart",
            "sri_in_marketing_document": "same_slide_as_sri",
            "esg_risk": "near_esg_mention",
            "opinion": "near_opinion",
            "issuers_mentioned": "near_company"
        }
        return locations.get(disclaimer_id, "unknown")
    
    def identify_relevant_rules(self, semantic_analysis: Dict) -> List[Dict]:
        """Identifie règles pertinentes"""
        relevant = []
        
        data = semantic_analysis.get("data_presented", {})
        visual = semantic_analysis.get("visual_elements", {})
        
        if data.get("performance_values", False):
            for rule in self.rules_by_check_type.get("performance_disclaimer", [])[:3]:
                relevant.append({
                    "rule_id": rule["rule_id"],
                    "rule_text": rule["rule_text"][:100] + "...",
                    "check_type": "performance_disclaimer",
                    "reason": "Performance détectée",
                    "priority": "high"
                })
        
        if data.get("benchmark_comparison", False):
            for rule in self.rules_by_check_type.get("performance_benchmark_comparison", [])[:3]:
                relevant.append({
                    "rule_id": rule["rule_id"],
                    "rule_text": rule["rule_text"][:100] + "...",
                    "check_type": "performance_benchmark_comparison",
                    "reason": "Comparaison benchmark",
                    "priority": "high"
                })
        
        if visual.get("has_chart", False):
            for rule in self.rules_by_check_type.get("citation_verification", [])[:3]:
                relevant.append({
                    "rule_id": rule["rule_id"],
                    "rule_text": rule["rule_text"][:100] + "...",
                    "check_type": "citation_verification",
                    "reason": "Graphiques",
                    "priority": "medium"
                })
        
        return relevant
    
    # ==========================
    # PIPELINE
    # ==========================
    
    def analyze_all_slides(self, output_path: str = "phase2_1_hybrid_output.json") -> Dict:
        print("\n" + "="*70)
        print("🔍 ANALYSE SLIDES (MATCHING HYBRIDE)")
        print("="*70)
        
        analyzed = []
        stats = {
            "total": len(self.slides),
            "success": 0,
            "errors": 0,
            "with_missing_disclaimers": 0,
            "with_performance": 0,
            "disclaimers_detected": 0
        }
        
        for i, slide_chunk in enumerate(self.slides, 1):
            slide_num = slide_chunk.get("metadata", {}).get("slide_number", i)
            content = slide_chunk.get("content", "")
            
            print(f"📄 Slide {slide_num:2d}/{stats['total']}...", end="", flush=True)
            
            try:
                semantic = self.analyze_slide_semantic(content, slide_num)
                disc_status = self.check_disclaimers(content, semantic)
                rel_rules = self.identify_relevant_rules(semantic)
                
                flags = {
                    "needs_disclaimer_check": len(disc_status["missing"]) > 0,
                    "needs_linguistic_check": any(r["check_type"] == "linguistic_restrictions" for r in rel_rules),
                    "needs_prospectus_check": any("strategie" in t or "allocation" in t for t in semantic.get("semantic_themes", [])),
                    "has_performance": semantic["data_presented"].get("performance_values", False),
                    "has_benchmark_comparison": semantic["data_presented"].get("benchmark_comparison", False)
                }
                
                enriched = slide_chunk.copy()
                enriched["semantic_analysis"] = semantic
                enriched["disclaimers_status"] = disc_status
                enriched["relevant_rules"] = rel_rules
                enriched["flags"] = flags
                
                analyzed.append(enriched)
                
                stats["success"] += 1
                if disc_status["missing"]:
                    stats["with_missing_disclaimers"] += 1
                if flags["has_performance"]:
                    stats["with_performance"] += 1
                if disc_status["present"]:
                    stats["disclaimers_detected"] += len(disc_status["present"])
                
                print(" ✓")
                
            except Exception as e:
                print(f" ❌ {str(e)[:60]}")
                err_chunk = slide_chunk.copy()
                err_chunk["analysis_error"] = str(e)
                analyzed.append(err_chunk)
                stats["errors"] += 1
        
        result = {
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "fund_name": self.fund_name,
                "client_type": self.client_type,
                "total_rules_applied": len(self.applicable_rules),
                "method": "hybrid_regex_embeddings"
            },
            "chunks_enriched": analyzed,
            "statistics": stats
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("✅ PHASE 2.1 TERMINÉE (HYBRID)")
        print("="*70)
        print(f"📊 Total: {stats['total']}")
        print(f"   ✓ Succès: {stats['success']}")
        print(f"   ⚠️ Disclaimers manquants: {stats['with_missing_disclaimers']}")
        print(f"   📈 Avec performance: {stats['with_performance']}")
        print(f"   🎯 Disclaimers détectés: {stats['disclaimers_detected']}")
        print(f"\n💾 Output: {output_path}")
        
        return result


# ============================================================
# EXÉCUTION
# ============================================================

CHUNKS = "example_2/outputs/pptx_slide_full.json"
METADATA = "example_2/outputs/metadata_final_optimized_test.json"
RULES = "example_2/outputs/phase1_correct_output.json"
DISCLAIMERS = "inputs/disclamer_bilingual_full.json"
OUTPUT = "example_2/outputs/phase2_1_hybrid_output2.json"

analyzer = SlideAnalyzerHybrid(
    chunks_slide_full_path=CHUNKS,
    metadata_path=METADATA,
    phase1_rules_path=RULES,
    disclaimers_path=DISCLAIMERS
)

result = analyzer.analyze_all_slides(output_path=OUTPUT)

# Inspection
print("\n" + "="*70)
print("🔍 INSPECTION SLIDE 24")
print("="*70)

slide_24 = next((s for s in result["chunks_enriched"] 
                 if s.get("metadata", {}).get("slide_number") == 24), None)
if slide_24:
    print(f"\n📄 Slide 24")
    print(f"Résumé: {slide_24['semantic_analysis']['resume_slide']}")
    ds = slide_24.get("disclaimers_status", {})
    print(f"\nRequis: {ds.get('required', [])}")
    print(f"Présents: {[p['id'] + ' (' + p['match_type'] + ')' for p in ds.get('present', [])]}")
    print(f"Trouvés: {ds.get('found', [])}")
    print(f"Manquants: {[m['id'] for m in ds.get('missing', [])]}")