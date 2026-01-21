# Generated from: 2_25_filtre_struct+llm.ipynb
# Converted at: 2025-12-16T20:46:36.514Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

"""
============================================================================
PHASE 2.25 - FILTRATION ULTRA-INTELLIGENTE PRÉ-VÉRIFICATION (CORRIGÉE)
============================================================================
Corrections:
1. Regex patterns corrigés (flags inline)
2. Exception pour slides structurelles (1, 34, 35) même si courtes
3. Lecture correcte du format chunks (content directement)
4. Ajout de l'exécution principale et des tests
============================================================================
"""

import json
import re
import httpx
from openai import OpenAI
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import logging
from tqdm.auto import tqdm
from collections import Counter

# Tenter d'importer fitz pour le fallback PDF
try:
    import fitz
    FIT_Z_AVAILABLE = True
except ImportError:
    FIT_Z_AVAILABLE = False

# ============================================================================
# ⚙️ CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Configuration optimisée"""

    # 📁 INPUTS
    PHASE22_OUTPUT = "example_2/outputs/phase2_2_fusion_optimale.json"
    PHASE1_RULES = "example_2/outputs/phase1_correct_output.json"
    PPTX_CHUNKS = "example_2/outputs/pptx_slide_full.json"
    PPTX_PDF = "example_2/expl2.pdf"
    VECTOR_DB_RULES = "example_2/outputs/phase1_correct_output.json"

    # 🤖 LLM
    LLM_API_KEY = "sk-7c0b80cf494746f580cc5ba555d739b2"
    LLM_BASE_URL = "https://tokenfactory.esprit.tn/api"
    LLM_MODEL = "hosted_vllm/Llama-3.1-70B-Instruct"
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 2500

    # 🎯 SLIDES STRUCTURELS (garde TOUTES les règles, même si courtes)
    # 1 (Cover), 2 (Disclaimer), 34 (Disclaimer 2), 35 (Contact)
    STRUCTURAL_SLIDES = {1, 2, 34, 35}

    # 🔍 SEUILS DE FILTRAGE
    MIN_RELEVANCE_SCORE =6  # Score minimum LLM (0-10)
    MIN_SLIDE_LENGTH = 80   # Longueur minimale (SAUF slides structurelles)
    MAX_TOC_LENGTH = 600

    # 🐛 DEBUG
    DEBUG_MODE = True
    USE_PDF_FALLBACK = False and FIT_Z_AVAILABLE # Désactivé par défaut si chunks OK

    # 📊 OUTPUT
    OUTPUT_JSON = "example_2/outputs/phase2_25_filtered_ultra.json"


# ============================================================================
# 🤖 CLIENT LLM
# ============================================================================

class LLMClient:
    def __init__(self):
        try:
            # Désactiver la vérification SSL pour l'environnement interne si nécessaire
            http_client = httpx.Client(verify=False, timeout=90.0)
            self.client = OpenAI(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_BASE_URL,
                http_client=http_client
            )
            self.enabled = True
            logger.info("✅ LLM initialisé")
        except Exception as e:
            logger.error(f"❌ LLM init failed: {e}")
            self.enabled = False

    def query(self, system: str, user: str, temp: float = 0.2) -> Optional[str]:
        if not self.enabled:
            return None

        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temp,
                max_tokens=Config.LLM_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"⚠️ LLM query failed: {e} - Retrying with lower tokens...")
            # Tentative avec moins de tokens en cas d'erreur
            try:
                response = self.client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature=temp,
                    max_tokens=1500 # Réduction
                )
                return response.choices[0].message.content.strip()
            except Exception as e2:
                logger.error(f"❌ LLM query failed even after retry: {e2}")
                return None


# ============================================================================
# 📊 DATA LOADER (CORRIGÉ pour format chunks)
# ============================================================================

class DataLoader:
    def __init__(self):
        logger.info("📁 Chargement des données...")

        self.phase22_data = self._load_json(Config.PHASE22_OUTPUT)
        self.phase1_rules = self._load_json(Config.PHASE1_RULES)
        self.chunks = self._load_chunks(Config.PPTX_CHUNKS)
        self.rules_db = self._load_rules(Config.VECTOR_DB_RULES)

        # Indexation slides
        self.slides_full = self._index_slides_from_chunks()

        if not self.slides_full and Config.USE_PDF_FALLBACK:
            logger.warning("⚠️ Chunks vides, tentative extraction PDF...")
            self.slides_full = self._extract_from_pdf()

        self.applicable_rules = self._extract_applicable_rules()

        if not self.slides_full:
            logger.error("❌ AUCUNE slide chargée !")
        else:
            logger.info(f"✅ {len(self.slides_full)} slides | {len(self.applicable_rules)} règles applicables")

    def _load_json(self, path: str) -> Dict:
        # Vérification de l'existence du fichier
        if not Path(path).exists():
            logger.error(f"❌ Fichier non trouvé: {path}")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Erreur de décodage JSON dans {path}: {e}")
                return {}

    def _load_rules(self, path: str) -> Dict[str, Dict]:
        data = self._load_json(path)

        if 'rules' in data:
            rules = data['rules']
        elif 'applicable_rules' in data:
            rules = data['applicable_rules']
        elif isinstance(data, list):
            rules = data
        else:
            logger.warning("⚠️ Format de règles inconnu")
            rules = []

        return {r['rule_id']: r for r in rules if 'rule_id' in r}

    def _load_chunks(self, path: str) -> List[Dict]:
        """Charge chunks au format: {chunk_id, chunk_type, content, metadata}"""
        data = self._load_json(path)
        chunks = data if isinstance(data, list) else data.get('chunks', [])

        if Config.DEBUG_MODE and chunks:
            logger.debug(f"📦 {len(chunks)} chunks chargés")
            sample = chunks[0]
            logger.debug(f"  Clés: {list(sample.keys())}")
            logger.debug(f"  Format détecté: {sample.get('chunk_type')}")

        return chunks

    def _index_slides_from_chunks(self) -> Dict[int, str]:
        """
        Indexe slides depuis le format des chunks, en agrégeant les éléments.
        """
        slides = {}

        # Première passe : Collecter tout le contenu par slide_number
        aggregated_content: Dict[int, List[Tuple[str, str]]] = {} # {slide_num: [(chunk_type, content), ...]}

        for chunk in self.chunks:
            chunk_type = chunk.get('chunk_type', '')
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            slide_num = metadata.get('slide_number')

            if not slide_num or not isinstance(slide_num, int) or not content:
                continue

            if slide_num not in aggregated_content:
                aggregated_content[slide_num] = []

            # Prioriser l'ordre: full > title > element
            if chunk_type == 'slide_full':
                # Si 'slide_full' existe, il écrase les autres éléments
                aggregated_content[slide_num] = [('slide_full', content)]
            else:
                aggregated_content[slide_num].append((chunk_type, content))

        # Deuxième passe : Consolider (en évitant la duplication si slide_full existe)
        for slide_num, content_list in aggregated_content.items():
            if content_list[0][0] == 'slide_full':
                slides[slide_num] = content_list[0][1]
            else:
                # Filtrer et joindre les titres et éléments
                unique_elements = {}
                for chunk_type, content in content_list:
                    # Utiliser un dictionnaire pour s'assurer que le contenu est unique ou pour prioriser
                    if chunk_type == 'slide_title':
                        unique_elements['title'] = content
                    elif chunk_type == 'slide_element' and content not in unique_elements.values():
                        # Ajouter les éléments un par un
                        unique_elements[f'el_{len(unique_elements)}'] = content

                # Reconstruire le texte en garantissant le titre en premier si disponible
                final_text = "\n".join(
                    [unique_elements.get('title', '')] +
                    [v for k, v in unique_elements.items() if k != 'title' and v]
                ).strip()

                if final_text:
                    slides[slide_num] = final_text

        if not slides:
            logger.error("❌ AUCUNE slide indexée !")
        else:
            logger.info(f"✅ {len(slides)} slides indexées")
            if Config.DEBUG_MODE:
                # Afficher quelques exemples
                for num in sorted(list(slides.keys()))[:3]:
                    preview = slides[num][:150].replace('\n', ' ')
                    logger.debug(f"  Slide {num}: {len(slides[num])} chars | {preview}...")

        return slides

    def _extract_applicable_rules(self) -> Set[str]:
        """Extrait rule_id applicables depuis Phase 1"""
        applicable = set()
        for rule in self.phase1_rules.get('applicable_rules', []):
            applicable.add(rule['rule_id'])
        return applicable

    def _extract_from_pdf(self) -> Dict[int, str]:
        """Fallback: extraction PDF (nécessite PyMuPDF/fitz)"""
        if not FIT_Z_AVAILABLE:
            logger.error("❌ PyMuPDF (fitz) non installé. Impossible d'utiliser le fallback PDF.")
            return {}

        try:
            pdf_path = Path(Config.PPTX_PDF)
            if not pdf_path.exists():
                logger.warning(f"⚠️ Fichier PDF non trouvé à {pdf_path}")
                return {}

            doc = fitz.open(pdf_path)
            slides = {}

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                slides[page_num + 1] = text

            doc.close()
            logger.info(f"✅ {len(slides)} slides extraites du PDF")
            return slides

        except Exception as e:
            logger.error(f"❌ Erreur extraction PDF: {e}")
            return {}


# ============================================================================
# 🔍 DÉTECTEUR ULTRA-PUISSANT (CORRIGÉ)
# ============================================================================

class UltraSlideDetector:
    """Détection ROBUSTE des slides non-pertinentes"""

    # Patterns regex CORRIGÉS (flags inline pour éviter l'erreur)
    TOC_PATTERNS = [
        r'\b0[1-9]\b.*\n.*\b0[1-9]\b',  # 01 ... 02 ... 03
        r'\b\d{1,2}\.\s+[A-Z]', # 1. TITLE ou 2. TITLE
        r'(?i:table\s+of\s+contents|sommaire|agenda)', # Keywords (flag inline)
    ]

    # Mots-clés sommaire
    TOC_KEYWORDS = {
        'table of contents', 'sommaire', 'agenda', 'overview',
        'what are', 'why invest', 'the case for', 'investment process', 'portfolio',
        'section', 'partie', 'chapitre'
    }

    # Patterns titre seul
    TITLE_ONLY_PATTERNS = [
        r'^[A-Z\s]{10,80}$', # Ex: UN GRAND TITRE EN MAJUSCULES
        r'^[0-9\s]+$', # Ex: 12
        r'^\s*$',
    ]

    @classmethod
    def analyze_slide(cls, slide_num: int, slide_text: str, slide_title: str) -> Dict[str, Any]:
        """
        Analyse COMPLÈTE d'une slide

        Returns:
            {
                'is_structural': bool,
                'is_empty': bool,
                'is_title_only': bool,
                'is_toc': bool,
                'should_skip': bool,
                'confidence': float,
                'reason': str
            }
        """

        clean_text = re.sub(r'\s+', ' ', slide_text.strip())
        text_length = len(clean_text)

        # 1. Slide structurelle critique (TOUJOURS garder)
        if slide_num in Config.STRUCTURAL_SLIDES:
            return {
                'is_structural': True,
                'is_empty': False,
                'is_title_only': False,
                'is_toc': False,
                'should_skip': False,
                'confidence': 1.0,
                'reason': f'Slide structurelle #{slide_num} (règles spéciales)'
            }

        # 2. Slide vide (SEULEMENT si NON structurelle)
        if text_length < Config.MIN_SLIDE_LENGTH:
            # Vérifier si c'est VRAIMENT vide (moins de 20 chars)
            is_truly_empty = text_length < 20

            return {
                'is_structural': False,
                'is_empty': True,
                'is_title_only': False,
                'is_toc': False,
                'should_skip': True,
                'confidence': 1.0 if is_truly_empty else 0.8,
                'reason': f'Contenu trop court: {text_length} chars'
            }

        # 3. Titre seul (heuristiques multiples)
        title_only_score = 0.0

        # Pas de ponctuation de fin, ou uniquement une phrase
        if len(re.findall(r'[.!?]', clean_text)) <= 1:
            title_only_score += 0.3

        # Très court ET pas de phrases complexes (mots > 3)
        if text_length < 200 and len(re.findall(r'\b\w{4,}\b', clean_text)) < 10:
            title_only_score += 0.4

        # Match patterns titre
        for pattern in cls.TITLE_ONLY_PATTERNS:
            if re.match(pattern, clean_text): # Utiliser match pour le début de ligne
                title_only_score += 0.3
                break

        if title_only_score >= 0.7:
            return {
                'is_structural': False,
                'is_empty': False,
                'is_title_only': True,
                'is_toc': False,
                'should_skip': True,
                'confidence': min(title_only_score, 1.0),
                'reason': f'Titre ou section courte (score: {title_only_score:.2f})'
            }

        # 4. Table des matières (analyse multi-critères)
        toc_score = 0.0
        combined_text = (slide_text + " " + slide_title).lower()

        # Mots-clés sommaire
        keyword_matches = sum(1 for kw in cls.TOC_KEYWORDS if kw in combined_text)
        if keyword_matches >= 2:
            toc_score += 0.4
        elif keyword_matches == 1:
            toc_score += 0.2

        # Patterns regex sommaire (CORRIGÉ)
        for pattern in cls.TOC_PATTERNS:
            try:
                if re.search(pattern, slide_text, re.DOTALL | re.IGNORECASE):
                    toc_score += 0.3
                    break
            except re.error as e:
                logger.warning(f"⚠️ Regex error: {pattern} - {e}")
                continue

        # Liste numérotée (01, 02, 03...)
        numbered_items = re.findall(r'\b0[1-9]\b', slide_text)
        if len(numbered_items) >= 3:
            toc_score += 0.3

        # Longueur compatible sommaire
        if Config.MIN_SLIDE_LENGTH < text_length < Config.MAX_TOC_LENGTH:
            toc_score += 0.1

        # Ratio lignes courtes (typique sommaire)
        lines = [l.strip() for l in slide_text.split('\n') if l.strip()]
        if lines:
            short_lines = sum(1 for l in lines if len(l) < 50)
            if short_lines / len(lines) > 0.6:
                toc_score += 0.2

        if toc_score >= 0.7:
            return {
                'is_structural': False,
                'is_empty': False,
                'is_title_only': False,
                'is_toc': True,
                'should_skip': True,
                'confidence': min(toc_score, 1.0),
                'reason': f'Table des matières ou section break (score: {toc_score:.2f})'
            }

        # 5. Contenu réel
        return {
            'is_structural': False,
            'is_empty': False,
            'is_title_only': False,
            'is_toc': False,
            'should_skip': False,
            'confidence': 1.0 - max(title_only_score, toc_score),
            'reason': 'Contenu réel détecté'
        }


# ============================================================================
# 🧹 FILTREUR ULTRA-INTELLIGENT
# ============================================================================

class UltraSmartFilter:
    """Filtrage LLM PRÉCIS avec scoring 0-10"""

    def __init__(self, llm: LLMClient, rules_db: Dict[str, Dict], applicable_rules: Set[str]):
        self.llm = llm
        self.rules_db = rules_db
        self.applicable_rules = applicable_rules

    def filter_rules(self, slide_num: int, slide_text: str, slide_title: str,
                     rules_from_phase22: List[Dict], slide_analysis: Dict) -> List[Dict]:
        """
        Filtre avec SCORING de pertinence 0-10
        """

        # Slide structurelle ou pas de règles → garder tout
        if not rules_from_phase22 or slide_analysis['is_structural']:
            return rules_from_phase22

        # Construire contexte enrichi
        rules_context = self._build_enriched_context(rules_from_phase22)

        # Prompt LLM ULTRA-PRÉCIS
        system = """Tu es un expert compliance qui SCORE LA PERTINENCE de règles pour une slide.

**TA TÂCHE**: Pour chaque règle, attribue un score de pertinence 0-10:
- **0-3**: Règle TOTALEMENT hors-sujet ou trop générique (à retirer)
- **4-6**: Règle moyennement pertinente (cas limite)
- **7-10**: Règle TRÈS pertinente (à garder absolument)

**CRITÈRES DE SCORING**:
✅ **Score 8-10**: Règle DIRECTEMENT applicable au contenu de la slide (ex: performance, risque, citations).
✅ **Score 5-7**: Règle INDIRECTEMENT liée ou d'application générale (ex: cohérence du prospectus, style).
❌ **Score 0-4**: Règle NON pertinente, même si le moteur de recherche l'a remontée (faible similarité sémantique).

**RÈGLES GÉNÉRALES (Baseline >= 5)**:
- Les règles critiques ou structurelles, comme celles sur la cohérence avec le prospectus (1.8, 1.9, etc.) ou les sources (1.3), doivent généralement avoir un score d'au moins 5, sauf si le contenu de la slide est clairement sans rapport.

**IMPORTANT**:
- Sois STRICT mais JUSTE
- Réponds UNIQUEMENT en JSON valide (sans markdown, sans en-tête ni pied de page, juste l'objet JSON).

Réponds UNIQUEMENT en JSON valide (sans markdown):
{
  "rules_scores": [
    {"rule_id": "1.3", "score": 8, "reason": "Slide contient graphique et nécessite une source."},
    {"rule_id": "3.3", "score": 2, "reason": "Règle de pays de commercialisation, non pertinente pour une slide de stratégie d'investissement."}
  ],
  "summary": "Résumé du filtrage et des principales règles conservées."
}"""

        user = f"""**SLIDE {slide_num}: {slide_title}**

**CONTENU SLIDE** (extrait):
{slide_text[:2000]}

**RÈGLES À SCORER**:
{rules_context}

**TÂCHE**: Score chaque règle 0-10 selon pertinence.

JSON attendu:
{{
  "rules_scores": [
    {{"rule_id": "X.Y", "score": N, "reason": "..."}},
    ...
  ],
  "summary": "Résumé"
}}"""

        raw = self.llm.query(system, user, temp=Config.LLM_TEMPERATURE)

        if not raw:
            logger.warning(f"⚠️ LLM failed slide {slide_num}, garde tout")
            return rules_from_phase22

        try:
            result = self._parse_json(raw)
            rules_scores = {r['rule_id']: r['score'] for r in result.get('rules_scores', []) if isinstance(r.get('score'), int)}

            # Filtrer selon MIN_RELEVANCE_SCORE
            filtered = []
            for rule in rules_from_phase22:
                rule_id = rule['rule_id']
                # Utiliser le score LLM si disponible, sinon le seuil minimum pour ne pas filtrer
                score = rules_scores.get(rule_id, Config.MIN_RELEVANCE_SCORE)

                # S'assurer que les scores des règles critiques (même s'ils sont dans la liste)
                # sont gérés par le LLM, mais le seuil est appliqué.

                if score >= Config.MIN_RELEVANCE_SCORE:
                    rule['llm_relevance_score'] = score
                    filtered.append(rule)
                else:
                    if Config.DEBUG_MODE:
                         reason = next((r['reason'] for r in result.get('rules_scores', []) if r.get('rule_id') == rule_id), "Score LLM trop bas.")
                         logger.debug(f"  [Rule {rule_id}] Retirée (Score: {score}) - {reason}")


            removed = len(rules_from_phase22) - len(filtered)
            if removed > 0:
                summary = result.get('summary', 'N/A')[:100]
                logger.info(f"  🧹 {removed} règles retirées: {summary}")

            return filtered

        except Exception as e:
            logger.warning(f"⚠️ Parse error or LLM issue slide {slide_num}: {e}. Garde toutes les règles. Raw: {raw[:300]}...")
            # En cas d'échec de parsing, on garde le résultat précédent pour la sûreté
            return rules_from_phase22

    def _build_enriched_context(self, rules: List[Dict]) -> str:
        """Contexte enrichi des règles"""
        lines = []
        for r in rules:
            rule_id = r['rule_id']
            rule_full = self.rules_db.get(rule_id, {})
            rule_text = rule_full.get('rule_text', 'N/A').replace('\n', ' ')[:120]
            check_type = r.get('check_type', 'unknown')
            severity = r.get('severity', 'unknown')

            lines.append(
                f"- **{rule_id}** ({check_type} | {severity}): {rule_text}..."
            )

        return "\n".join(lines)

    def _parse_json(self, raw: str) -> Dict:
        """Parse JSON robuste"""
        # Nettoyage des backticks et autres wrappers markdown
        clean = re.sub(r'```json\s*', '', raw, flags=re.IGNORECASE)
        clean = re.sub(r'```\s*', '', clean).strip()

        # Recherche de l'objet JSON (du premier '{' au dernier '}')
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            clean = match.group(0)

        # Tentative de correction simple des chaînes mal formatées (échappement)
        clean = clean.replace('\\"', '"').replace('\n', ' ')

        return json.loads(clean)


# ============================================================================
# 🎯 ORCHESTRATEUR
# ============================================================================

class UltraPhase225:
    def __init__(self):
        logger.info("\n" + "="*80)
        logger.info("🚀 PHASE 2.25 - FILTRATION ULTRA-INTELLIGENTE")
        logger.info("="*80 + "\n")

        self.llm = LLMClient()
        self.data = DataLoader()
        self.detector = UltraSlideDetector()
        self.filter = UltraSmartFilter(self.llm, self.data.rules_db, self.data.applicable_rules)

    def run(self) -> Dict:
        slides_from_phase22 = self.data.phase22_data.get('slides_analysis', [])

        if not slides_from_phase22:
             logger.error("❌ Aucune slide trouvée dans le fichier de la Phase 2.2. Arrêt du processus.")
             return {}

        results = {
            'metadata': {
                'phase': '2.25_ultra_intelligent',
                'timestamp': datetime.now().isoformat(),
                'description': 'Filtrage LLM scoring 0-10 + détection robuste (TOC, vide, titre seul)',
                'config': {
                    'min_relevance_score': Config.MIN_RELEVANCE_SCORE,
                    'min_slide_length': Config.MIN_SLIDE_LENGTH,
                    'structural_slides': list(Config.STRUCTURAL_SLIDES)
                }
            },
            'slides_analysis': [],
            'statistics': {
                'total_slides': 0,
                'empty_slides': 0,
                'title_only_slides': 0,
                'toc_slides': 0,
                'structural_slides': 0,
                'content_slides': 0,
                'total_rules_before': 0,
                'filtered_rules': 0,
                'kept_rules': 0
            }
        }

        # Calcul des règles totales avant filtration pour les stats finales
        total_rules_before = sum(len(s.get('selected_rules', [])) for s in slides_from_phase22)
        results['statistics']['total_rules_before'] = total_rules_before


        for slide_analysis in tqdm(slides_from_phase22, desc="Filtration Ultra"):
            slide_num = slide_analysis['slide_number']
            slide_title = slide_analysis.get('slide_title', '')
            rules_phase22 = slide_analysis.get('selected_rules', [])

            results['statistics']['total_slides'] += 1

            # Récupérer contenu
            slide_text = self.data.slides_full.get(slide_num, "")

            if not slide_text:
                logger.warning(f"⚠️ Pas de contenu slide {slide_num}. Règles ignorées.")
                results['slides_analysis'].append({
                    'slide_number': slide_num,
                    'slide_title': slide_title,
                    'slide_type': 'no_content',
                    'selected_rules': [],
                    'detection': {'reason': 'Contenu manquant'},
                    'statistics': {'phase22_rules': len(rules_phase22), 'filtered_rules': 0, 'removed_rules': len(rules_phase22)}
                })
                results['statistics']['filtered_rules'] += len(rules_phase22)
                continue

            if Config.DEBUG_MODE:
                logger.info(f"\n📄 Slide {slide_num}: {slide_title[:60]}")

            # DÉTECTION
            detection = self.detector.analyze_slide(slide_num, slide_text, slide_title)

            if Config.DEBUG_MODE:
                logger.debug(f"  🔍 {detection['reason']} (conf: {detection['confidence']:.2f})")

            # Stats de détection
            if detection['is_structural']:
                results['statistics']['structural_slides'] += 1
            elif detection['is_empty']:
                results['statistics']['empty_slides'] += 1
            elif detection['is_title_only']:
                results['statistics']['title_only_slides'] += 1
            elif detection['is_toc']:
                results['statistics']['toc_slides'] += 1
            else:
                results['statistics']['content_slides'] += 1

            # Skip si nécessaire (règles retirées et comptées)
            if detection['should_skip']:
                if Config.DEBUG_MODE:
                    logger.warning(f"  ⏭️  SKIP: {detection['reason']}. {len(rules_phase22)} règles retirées.")

                results['slides_analysis'].append({
                    'slide_number': slide_num,
                    'slide_title': slide_title,
                    'slide_type': 'skipped',
                    'selected_rules': [],
                    'detection': detection,
                    'statistics': {'phase22_rules': len(rules_phase22), 'filtered_rules': 0, 'removed_rules': len(rules_phase22)}
                })
                results['statistics']['filtered_rules'] += len(rules_phase22)
                continue

            # FILTRAGE LLM
            if Config.DEBUG_MODE:
                logger.info(f"  📦 Règles Phase 2.2: {len(rules_phase22)}")

            filtered_rules = self.filter.filter_rules(
                slide_num, slide_text, slide_title, rules_phase22, detection
            )

            # Validation finale (Assurer que seules les règles applicables sont conservées)
            final_rules = [
                r for r in filtered_rules
                if r['rule_id'] in self.data.applicable_rules
            ]

            removed = len(rules_phase22) - len(final_rules)
            results['statistics']['filtered_rules'] += removed
            results['statistics']['kept_rules'] += len(final_rules)

            if Config.DEBUG_MODE:
                logger.info(f"  ✅ Finales: {len(final_rules)} (retiré: {removed})")

            # Calcul du score LLM moyen pour les stats
            avg_llm_score = round(sum(r.get('llm_relevance_score', 0) for r in final_rules) / len(final_rules), 2) if final_rules else 0

            # Sauvegarder
            results['slides_analysis'].append({
                'slide_number': slide_num,
                'slide_title': slide_title,
                'slide_type': 'structural' if detection['is_structural'] else 'content',
                'selected_rules': final_rules,
                'detection': detection,
                'statistics': {
                    'phase22_rules': len(rules_phase22),
                    'filtered_rules': len(final_rules),
                    'removed_rules': removed,
                    'avg_llm_score': avg_llm_score
                }
            })

        # STATISTIQUES GLOBALES FINALES
        total_before = results['statistics']['total_rules_before']
        total_after = results['statistics']['kept_rules']

        results['statistics']['reduction_rate'] = round(
            (total_before - total_after) / total_before * 100, 1
        ) if total_before > 0 else 0

        # SAUVEGARDE
        with open(Config.OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"\n" + "="*80)
        logger.info(f"✅ FILTRATION ULTRA TERMINÉE")
        logger.info(f"📊 Règles avant: {total_before}")
        logger.info(f"📊 Règles après: {total_after}")
        logger.info(f"📉 Réduction: {results['statistics']['reduction_rate']}%")
        logger.info(f"📑 Slides:")
        logger.info(f"   - Structurelles: {results['statistics']['structural_slides']}")
        logger.info(f"   - Contenu: {results['statistics']['content_slides']}")
        logger.info(f"   - Vides/Titres/TOC (Skippées): {results['statistics']['empty_slides'] + results['statistics']['title_only_slides'] + results['statistics']['toc_slides']}")
        logger.info(f"💾 Résultat sauvegardé: {Config.OUTPUT_JSON}")
        logger.info("="*80 + "\n")

        return results


# ============================================================================
# 🧪 TESTS DE DÉTECTION
# ============================================================================

def test_ultra_detection():
    """Tests unitaires du détecteur de slides (TOC, Vide, Titre seul)"""

    # Rétablir la configuration courte pour le test si elle a été modifiée
    global Config
    temp_min_length = Config.MIN_SLIDE_LENGTH
    Config.MIN_SLIDE_LENGTH = 80

    test_cases = [
        {
            'name': 'Sommaire typique (Numérotation 0X)',
            'num': 6,
            'title': 'Agenda',
            'content': '01 WHAT ARE ACTIVE ETFs\n02 WHY INVEST IN THE US\n03 THE CASE FOR MOMENTUM\n04 INVESTMENT PROCESS',
            'expected_skip': True,
            'expected_type': 'toc'
        },
        {
            'name': 'Sommaire (Keywords et Regex simple)',
            'num': 7,
            'title': 'Table of Contents',
            'content': '1. Introduction\n2. US Equity Market\n3. Portfolio Strategy\n4. Conclusion',
            'expected_skip': True,
            'expected_type': 'toc'
        },
        {
            'name': 'Titre seul (Majuscules)',
            'num': 11,
            'title': 'STRATEGIC OUTLOOK',
            'content': 'STRATEGIC OUTLOOK',
            'expected_skip': True,
            'expected_type': 'title_only'
        },
        {
            'name': 'Contenu réel (OK)',
            'num': 12,
            'title': 'Introduction Why invest in the USA?',
            'content': '''World's largest economy with GDP 29 trillion $.
Innovation Center. Major Financial Market 63% global cap.
S&P500 10.3% average annual performance since 1957. This slide contains detailed market data, charts, and several sentences of analysis, easily exceeding the minimum length threshold.''',
            'expected_skip': False,
            'expected_type': 'content'
        },
        {
            'name': 'Slide structurelle 1 (Cover)',
            'num': 1,
            'title': 'Cover',
            'content': 'ODDO BHF\nSEPT 2025',
            'expected_skip': False,
            'expected_type': 'structural'
        },
        {
            'name': 'Slide structurelle 35 (Contact)',
            'num': 35,
            'title': 'Contacts',
            'content': 'Contact info\nEmail: contact@oddo.fr',
            'expected_skip': False,
            'expected_type': 'structural'
        },
        {
            'name': 'Slide vide/courte (Non structurelle)',
            'num': 99,
            'title': 'Empty slide',
            'content': 'Source: Company name.', # ~20 chars (sous 80)
            'expected_skip': True,
            'expected_type': 'empty'
        },
    ]

    detector = UltraSlideDetector()

    logger.info("\n" + "🧪 TEST DÉTECTION ROBUSTE" + "\n" + "=" * 80)

    all_passed = True

    for case in test_cases:
        result = detector.analyze_slide(case['num'], case['content'], case['title'])
        passed_skip = result['should_skip'] == case['expected_skip']

        # Déterminer le type réel pour la comparaison
        detected_type = 'structural' if result['is_structural'] else \
                        'empty' if result['is_empty'] else \
                        'title_only' if result['is_title_only'] else \
                        'toc' if result['is_toc'] else 'content'

        passed_type = detected_type == case['expected_type']

        status = "✅ PASS" if passed_skip and passed_type else "❌ FAIL"

        if not passed_skip or not passed_type:
            all_passed = False

        logger.info(f"{status} | {case['name']} (Slide {case['num']})")
        logger.info(f"    - Skip: Exp={case['expected_skip']}, Act={result['should_skip']}")
        logger.info(f"    - Type: Exp={case['expected_type']}, Act={detected_type} (Reason: {result['reason']})")

    logger.info("=" * 80)
    logger.info(f"RÉSUMÉ TEST DÉTECTION: {'✅ TOUS PASSÉS' if all_passed else '❌ ÉCHECS DÉTECTÉS'}")
    logger.info("=" * 80)

    # Rétablir la configuration
    Config.MIN_SLIDE_LENGTH = temp_min_length


# ============================================================================
# 🖥️ EXÉCUTION PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    if Config.DEBUG_MODE:
        test_ultra_detection()

    # Vérification des fichiers d'entrée critiques
    if not Path(Config.PHASE22_OUTPUT).exists():
        logger.error(f"❌ Fichier d'entrée critique manquant: {Config.PHASE22_OUTPUT}. Impossible de continuer.")
    elif not Path(Config.PPTX_CHUNKS).exists():
        logger.error(f"❌ Fichier de chunks PPTX manquant: {Config.PPTX_CHUNKS}. Impossible de continuer.")
    else:
        try:
            processor = UltraPhase225()
            processor.run()
        except Exception as e:
            logger.critical(f"Erreur fatale dans l'orchestrateur UltraPhase225: {e}", exc_info=True)