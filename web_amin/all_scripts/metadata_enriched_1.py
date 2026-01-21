#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: etape_2_3.ipynb
Conversion Date: 2025-12-14T08:57:51.404Z
"""

# ceci le metadata celle de pptx


# Installation des dépendances (à exécuter une seule fois dans Colab si besoin)


import json
import httpx
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import docx
from openai import OpenAI
import warnings
import PyPDF2

warnings.filterwarnings('ignore')


class ReferenceDataLoader:
    """Charge les fichiers de référence UNE SEULE FOIS au démarrage"""

    def __init__(self, disclaimers_path: str, registration_path: str, rules_path: Optional[str] = None):
        print("📚 Chargement des fichiers de référence...")
        self.disclaimers = self._load_disclaimers(disclaimers_path)
        self.registration_df = self._load_registration(registration_path)
        self.rules = self._load_rules(rules_path) if rules_path else {}
        print(f"✅ {len(self.disclaimers)} disclaimers chargés")
        print(f"✅ {len(self.registration_df)} fonds dans Registration")
        print(f"✅ Fichiers de référence prêts\n")

    def _load_disclaimers(self, path: str) -> Dict[str, Dict[str, str]]:
        """Charge les disclaimers depuis Excel ou JSON"""
        try:
            if path.endswith('.json'):
                return self._load_disclaimers_from_json(path)
            else:
                return self._load_disclaimers_from_excel(path)
        except Exception as e:
            print(f"⚠️ Erreur chargement disclaimers: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _load_disclaimers_from_json(self, path: str) -> Dict[str, Dict[str, str]]:
        """Charge disclaimers depuis JSON"""
        try:
            print("📄 Chargement disclaimers depuis JSON...")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            non_pro = data['data']['non_professional']
            pro = data['data']['professional']
            disclaimers = {}

            for key, text_retail in non_pro.items():
                text_pro = pro.get(key, None)
                disclaimer_id = self._normalize_json_key(key)
                original_name = self._create_readable_name(key)

                disclaimers[disclaimer_id] = {
                    'id': disclaimer_id,
                    'original_name': original_name,
                    'text_retail': self._clean_text(text_retail),
                    'text_professional': self._clean_text(text_pro)
                }

            print(f"✅ {len(disclaimers)} disclaimers chargés depuis JSON")
            for idx, (disc_id, disc_data) in enumerate(list(disclaimers.items())[:5]):
                has_retail = '✅' if disc_data['text_retail'] else '❌'
                has_pro = '✅' if disc_data['text_professional'] else '❌'
                print(f"   [{disc_id}] Retail: {has_retail} | Pro: {has_pro}")

            return disclaimers
        except Exception as e:
            print(f"⚠️ Erreur parsing JSON disclaimers: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _load_disclaimers_from_excel(self, path: str) -> Dict[str, Dict[str, str]]:
        """Charge disclaimers depuis Excel"""
        try:
            df = pd.read_excel(path, engine='openpyxl')
            print(f"📋 Colonnes du fichier disclaimers : {list(df.columns)}")

            col_type = df.columns[0]
            col_retail = None
            col_pro = None

            for col in df.columns:
                col_lower = str(col).lower()
                if 'non professional' in col_lower or 'non professionnel' in col_lower or 'retail' in col_lower:
                    col_retail = col
                elif 'professional' in col_lower or 'professionnel' in col_lower:
                    col_pro = col

            if not col_retail or not col_pro:
                print(f"⚠️ Colonnes retail/pro non trouvées")
                return {}

            print(f"✅ Colonnes détectées : Type='{col_type}', Retail='{col_retail}', Pro='{col_pro}'")

            disclaimers = {}
            for idx, row in df.iterrows():
                disclaimer_type = str(row.get(col_type, "")).strip()
                if not disclaimer_type or disclaimer_type == 'nan':
                    continue

                disclaimer_id = self._normalize_id(disclaimer_type)
                text_retail = self._clean_text(row.get(col_retail, ""))
                text_pro = self._clean_text(row.get(col_pro, ""))

                disclaimers[disclaimer_id] = {
                    "id": disclaimer_id,
                    "original_name": disclaimer_type,
                    "text_retail": text_retail,
                    "text_professional": text_pro
                }

                if idx < 3:
                    print(
                        f"   [{disclaimer_id}] Retail: {'✅' if text_retail else '❌'} | Pro: {'✅' if text_pro else '❌'}"
                    )

            return disclaimers
        except Exception as e:
            print(f"⚠️ Erreur chargement Excel: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _normalize_json_key(self, key: str) -> str:
        """Convertit une clé JSON en ID de disclaimer normalisé"""
        mappings = {
            'obam_presentation': 'OBAM_Presentation',
            'commercial_doc_manco_obam_sas': 'Commercial_doc_OBAM_SAS',
            'commercial_doc_manco_obam_gmbh': 'Commercial_doc_OBAM_GmbH',
            'commercial_doc_manco_obam_lux': 'Commercial_doc_OBAM_Lux',
            'commercial_doc_strategies_manco_obam_sas': 'Commercial_doc_strategies_OBAM_SAS',
            'commercial_doc_strategies_manco_obam_gmbh': 'Commercial_doc_strategies_OBAM_GmbH',
            'commercial_doc_strategies_manco_obam_lux': 'Commercial_doc_strategies_OBAM_Lux',
            'commercial_doc_lux_funds_raif': 'Commercial_doc_Lux_RAIF',
            'funds_reporting_manco_obam_sas': 'Funds_reporting_OBAM_SAS',
            'new_offer_products_strategy_only': 'New_offer_strategy_only',
            'new_offer_products_strategy_with_track_record': 'New_offer_with_track_record',
            'opinion': 'Opinion',
            'performance': 'Performance',
            'esg_risk': 'ESG_Risk',
            'issuers_mentioned': 'Issuers_mentioned',
            'back_tested_performance': 'Back_tested_performance',
            'simulations_future_performance': 'Simulations_future_performance',
            'ytm_ytw_usage': 'YtM_YtW',
            'add_info_fww_fundstars': 'Additional_FWW_Fundstars',
            'add_info_perf_german_prof_clients': 'Additional_German_pro_performance',
            'add_info_msci_esg_usage': 'Additional_MSCI_ESG',
            'add_info_german_lux_funds_ch': 'Additional_DE_LUX_Switzerland',
            'add_info_french_funds_ch': 'Additional_FR_Switzerland',
            'add_info_sicav_obam_lux_ch': 'Additional_SICAV_LUX_Switzerland',
            'add_info_sicav_obam_france_ch': 'Additional_SICAV_FR_Switzerland',
            'weekly_reporting_money_market_reg': 'Reporting_hebdo_monetaire',
            'sfdr_art_6': 'SFDR_ART_6',
            'sfdr_art_8': 'SFDR_ART_8',
            'sfdr_art_9': 'SFDR_ART_9',
            'video': 'VIDEO',
            'sri_in_marketing_document': 'SRI_marketing'
        }
        return mappings.get(key, key.replace('_', ' ').title().replace(' ', '_'))

    def _create_readable_name(self, key: str) -> str:
        """Crée un nom lisible depuis une clé JSON"""
        readable_names = {
            'obam_presentation': 'OBAM Presentation',
            'commercial_doc_manco_obam_sas': 'Commercial documentation : management company = OBAM SAS',
            'commercial_doc_manco_obam_gmbh': 'Commercial documentation : management company = OBAM GmbH',
            'commercial_doc_manco_obam_lux': 'Commercial documentation : management company = OBAM Lux',
            'opinion': 'Opinion',
            'performance': 'Performance',
            'esg_risk': 'ESG Risk',
            'issuers_mentioned': 'Issuers mentioned',
            'back_tested_performance': 'Back-tested performance',
            'simulations_future_performance': 'Simulations of future performance',
            'ytm_ytw_usage': 'Utilisation du Yield to Maturity (YtM)/ Yield to Worst (YtW)',
            'sri_in_marketing_document': 'SRI in marketing Document',
            'sfdr_art_6': 'SFDR ART.6',
            'sfdr_art_8': 'SFDR ART.8',
            'sfdr_art_9': 'SFDR ART.9',
            'new_offer_products_strategy_only': 'New offer products (strategy only)',
            'new_offer_products_strategy_with_track_record':
                'New offer products (strategy mentioning funds track record)'
        }
        return readable_names.get(key, key.replace('_', ' ').title())

    def _load_registration(self, path: str) -> pd.DataFrame:
        """Charge le fichier Registration of Funds"""
        try:
            df = pd.read_excel(path, engine='openpyxl')
            print(f"📊 Registration : {len(df)} fonds, {len(df.columns)} colonnes")
            return df
        except Exception as e:
            print(f"⚠️ Erreur chargement Registration: {e}")
            return pd.DataFrame()

    def _load_rules(self, path: str) -> Dict:
        """Charge les règles depuis le fichier .docx"""
        return {}

    def _normalize_id(self, name: str) -> str:
        """Convertit un nom de disclaimer en ID propre"""
        simplifications = {
            'OBAM Presentation': 'OBAM_Presentation',
            'Commercial documentation : management company = OBAM SAS': 'Commercial_doc_OBAM_SAS',
            'Commercial documentation : management company = OBAM GmbH': 'Commercial_doc_OBAM_GmbH',
            'Commercial documentation : management company = OBAM Lux': 'Commercial_doc_OBAM_Lux',
            'Commercial documentation (strategies) : management Company = OBAM SAS':
                'Commercial_doc_strategies_OBAM_SAS',
            'New offer products (strategy only)': 'New_offer_strategy_only',
            'Back-tested performance': 'Back_tested_performance',
            'Simulations  of  future performance': 'Simulations_future_performance',
            'SRI in marketing Document': 'SRI_marketing',
            'Opinion': 'Opinion',
            'Performance': 'Performance',
            'ESG Risk': 'ESG_Risk',
            'Issuers mentioned': 'Issuers_mentioned'
        }
        if name in simplifications:
            return simplifications[name]
        clean = name.replace(':', '').replace('=', '').replace('(', '').replace(')', '')
        clean = clean.replace('  ', ' ').strip()
        clean = clean.replace(' ', '_')
        return clean

    def _clean_text(self, text) -> Optional[str]:
        """Nettoie le texte d'un disclaimer"""
        if pd.isna(text):
            return None
        text = str(text).strip()
        if text == 'nan' or len(text) == 0:
            return None
        return text


class ProspectusExtractor:
    """Extrait les informations du prospectus COMPLET avec Llama 70B"""

    def __init__(self, api_key: str):
        http_client = httpx.Client(verify=False)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://tokenfactory.esprit.tn/api",
            http_client=http_client
        )
        self.model = "hosted_vllm/Llama-3.1-70B-Instruct"

    def extract(self, prospectus_path: str) -> Dict[str, Any]:
        """Extrait les informations clés du prospectus via LLM"""
        print(f"\n{'=' * 70}")
        print(f"🔍 EXTRACTION PROSPECTUS - VERSION COMPLÈTE")
        print(f"{'=' * 70}\n")
        print(f"📄 Fichier: {Path(prospectus_path).name}")

        prospectus_text = self._read_prospectus_complete(prospectus_path)
        extracted_data = self._extract_with_llm(prospectus_text)

        print(f"\n{'=' * 70}")
        print("✅ PROSPECTUS ANALYSÉ")
        print(f"{'=' * 70}\n")
        return extracted_data

    def _read_prospectus_complete(self, path: str) -> str:
        """Lit le prospectus COMPLET (toutes les pages/paragraphes)"""
        file_ext = Path(path).suffix.lower()
        print(f"📖 Lecture du prospectus ({file_ext})...")

        try:
            if file_ext == '.pdf':
                return self._read_pdf_complete(path)
            elif file_ext == '.docx':
                return self._read_docx_complete(path)
            else:
                raise ValueError(f"Format non supporté: {file_ext}")
        except Exception as e:
            print(f"❌ Erreur lecture prospectus: {e}")
            raise

    def _read_pdf_complete(self, path: str) -> str:
        """Lit TOUTES les pages d'un PDF"""
        with open(path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"   📄 Nombre de pages: {num_pages}")

            full_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                full_text += page.extract_text() + "\n\n"

            full_text = full_text.strip()
            print(f"   ✅ {len(full_text)} caractères extraits")
            print(f"   ✅ ~{len(full_text.split())} mots")

            if len(full_text) < 100:
                raise ValueError("Le prospectus semble vide")
            return full_text

    def _read_docx_complete(self, path: str) -> str:
        """Lit TOUS les paragraphes d'un .docx"""
        doc = docx.Document(path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        print(f"   📄 Nombre de paragraphes: {len(paragraphs)}")

        full_text = '\n'.join(paragraphs)
        print(f"   ✅ {len(full_text)} caractères extraits")
        print(f"   ✅ ~{len(full_text.split())} mots")

        if len(full_text) < 100:
            raise ValueError("Le prospectus semble vide")
        return full_text

    def _chunk_text(self, text: str, max_chars: int = 28000) -> List[str]:
        """Découpe le texte en chunks si nécessaire"""
        if len(text) <= max_chars:
            return [text]

        print(f"   ⚠️  Texte trop long ({len(text)} chars), découpage en chunks...")

        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        print(f"   ✅ {len(chunks)} chunks créés")
        return chunks

    def _normalize_esg_value(self, raw: Optional[str]) -> Optional[str]:
        """Normalise la valeur esg_sfdr_article renvoyée par le LLM"""
        if raw is None:
            return None

        text = str(raw).strip().lower()
        if not text:
            return None

        if "article 9" in text or "art. 9" in text or "art 9" in text:
            return "Article 9"
        if "article 8" in text or "art. 8" in text or "art 8" in text:
            return "Article 8"
        if "article 6" in text or "art. 6" in text or "art 6" in text:
            return "Article 6"

        if text.strip() in ["9", "9.0"]:
            return "Article 9"
        if text.strip() in ["8", "8.0"]:
            return "Article 8"
        if text.strip() in ["6", "6.0"]:
            return "Article 6"

        return None

    def _infer_esg_sfdr_article(self, prospectus_text: str) -> Optional[str]:
        """Fallback basé sur des patterns typiques SFDR dans le texte complet"""
        if not prospectus_text:
            return None

        text = prospectus_text.lower()

        if "article 9" in text or "art. 9" in text or "art 9" in text:
            return "Article 9"
        if "article 8" in text or "art. 8" in text or "art 8" in text:
            return "Article 8"
        if "article 6" in text or "art. 6" in text or "art 6" in text:
            return "Article 6"

        article_9_patterns = [
            "sustainable investment objective",
            "has sustainable investment as its objective",
            "objective is sustainable investment",
            "aims to achieve sustainable investment"
        ]
        for pattern in article_9_patterns:
            if pattern in text:
                return "Article 9"

        article_8_patterns = [
            "promotes environmental and social characteristics",
            "promoting environmental and social characteristics",
            "promote environmental or social characteristics",
            "promotes environmental or social characteristics",
            "environmental and social characteristics are promoted"
        ]
        for pattern in article_8_patterns:
            if pattern in text:
                return "Article 8"

        return None

    def _extract_with_llm(self, prospectus_text: str) -> Dict[str, Any]:
        """Utilise Llama 70B pour extraire les informations du prospectus COMPLET"""
        print("\n🤖 Extraction avec Llama 70B...")

        chunks = self._chunk_text(prospectus_text)
        print(f"   📊 Prospectus divisé en {len(chunks)} chunk(s)")

        prompt_template = """You are a financial document parser. Extract key information from this fund prospectus.

PROSPECTUS EXCERPT (CHUNK {chunk_num}/{total_chunks}):
{prospectus_text}

Extract the following information. If a field is not found, use null.

For esg_sfdr_article, infer the SFDR classification of the fund (Article 6, 8, or 9), based on formulations like:
- "promotes environmental and social characteristics" -> Article 8
- "has sustainable investment as its objective" -> Article 9
- If no specific mention, it may be Article 6 (default)

Return ONLY a valid JSON object with these exact fields:
{{
  "fund_name": "complete fund name (e.g., ODDO BHF US Equity Active UCITS ETF)",
  "inception_date": "date in DD/MM/YYYY or Month YYYY format, or null",
  "sri_risk_level": integer from 1 to 7, or null,
  "esg_sfdr_article": "Article 6" or "Article 8" or "Article 9" or null,
  "benchmark": "benchmark name (e.g., S&P 500 USD Net Total Return) or null",
  "investment_objective": "brief investment objective (max 150 chars) or null",
  "currency": "3-letter currency code (EUR, USD, GBP) or null",
  "fund_type": "UCITS" or "ETF" or "UCITS ETF" or "RAIF" or null,
  "minimum_investment": "minimum investment amount with currency or null",
  "ter": "total expense ratio as percentage string (e.g., '0.35%') or null",
  "distribution_policy": "Accumulation" or "Distribution" or null
}}

Important: Return ONLY the JSON object, no explanations, no markdown.

JSON:"""

        all_results = []

        for i, chunk in enumerate(chunks, 1):
            prompt = prompt_template.format(
                chunk_num=i,
                total_chunks=len(chunks),
                prospectus_text=chunk
            )

            try:
                print(f"   🔄 Traitement chunk {i}/{len(chunks)}...")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise JSON extractor. Return only valid JSON, no additional text."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=800
                )

                content = response.choices[0].message.content.strip()

                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]

                content = content.strip()
                data = json.loads(content)
                all_results.append(data)

                extracted_count = sum(1 for v in data.values() if v is not None)
                print(f"   ✅ Chunk {i}: {extracted_count}/{len(data)} champs extraits")

            except json.JSONDecodeError as e:
                print(f"   ⚠️  Chunk {i}: Erreur parsing JSON: {e}")
                print(f"   Réponse brute: {content[:200]}...")
                all_results.append({})
            except Exception as e:
                print(f"   ❌ Chunk {i}: Erreur API: {e}")
                all_results.append({})

        print("\n   🔄 Fusion des résultats des chunks...")
        merged_data = {}
        for result in all_results:
            for key, value in result.items():
                if value is not None and (key not in merged_data or merged_data[key] is None):
                    merged_data[key] = value

        raw_esg = merged_data.get("esg_sfdr_article")

        if isinstance(raw_esg, str):
            normalized_esg = self._normalize_esg_value(raw_esg)
        else:
            normalized_esg = raw_esg

        if normalized_esg is None:
            print("   🔍 esg_sfdr_article vide, tentative d'inférence depuis le texte...")
            inferred_esg = self._infer_esg_sfdr_article(prospectus_text)
            if inferred_esg:
                print(f"   ℹ️  esg_sfdr_article inféré: {inferred_esg}")
                merged_data["esg_sfdr_article"] = inferred_esg
            else:
                print("   ⚠️  Impossible d'inférer esg_sfdr_article")
                merged_data["esg_sfdr_article"] = None
        else:
            merged_data["esg_sfdr_article"] = normalized_esg
            print(f"   ✅ esg_sfdr_article normalisé: {normalized_esg}")

        print(f"\n   📋 RÉSUMÉ EXTRACTION:")
        for key, value in merged_data.items():
            status = "✅" if value is not None else "⚠️"
            print(f"   {status} {key}: {value}")

        return merged_data


class ComplianceEngine:
    """Moteur de conformité"""

    def __init__(self, reference_loader: ReferenceDataLoader):
        self.ref = reference_loader

    def enrich(self, metadata: Dict[str, Any], prospectus_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit les métadonnées avec toutes les informations de conformité"""
        print("\n⚙️ Calcul des règles de conformité...\n")

        complete_data = self._merge_data(metadata, prospectus_data)
        derived = self._calculate_derived(complete_data)
        applicable_rules = self._get_applicable_rules(complete_data, derived)
        required_disclaimers = self._get_required_disclaimers(complete_data, derived, applicable_rules)
        registration = self._check_registration(complete_data, derived)
        exploitation = self._build_exploitation(complete_data, derived, applicable_rules, required_disclaimers)

        return {
            'metadata': complete_data,
            'prospectus_parsed': prospectus_data,
            'derived': derived,
            'applicable_rules': applicable_rules,
            'required_disclaimers': required_disclaimers,
            'registration': registration,
            'exploitation': exploitation,
            'enrichment_date': datetime.now().isoformat()
        }

    def _merge_data(self, metadata: Dict, prospectus: Dict) -> Dict:
        """Fusionne metadata.json et données extraites du prospectus"""
        merged = metadata.copy()
        if 'fund_info' not in merged:
            merged['fund_info'] = {}

        for key, value in prospectus.items():
            if value is not None:
                if key not in merged['fund_info'] or not merged['fund_info'].get(key):
                    merged['fund_info'][key] = value

        return merged

    def _calculate_derived(self, data: Dict) -> Dict[str, Any]:
        """Calcule les métadonnées dérivées"""
        if 'document_info' in data:
            target_audience = data['document_info'].get('target_audience', 'retail')
            client_type = 'professionnel' if target_audience == 'professional' else 'retail'
        else:
            is_pro = data.get("Le client est-il un professionnel", False)
            client_type = 'professionnel' if is_pro else 'retail'

        if 'document_info' in data:
            language = data['document_info'].get('language', 'FR')
        else:
            language = data.get('language', 'FR')

        if 'management_info' in data:
            management_entity = data['management_info'].get(
                'management_company',
                'ODDO BHF ASSET MANAGEMENT SAS'
            )
            is_sicav_oddo = data['management_info'].get('is_sicav_oddo', False)
        else:
            management_entity = data.get('Société de Gestion', 'ODDO BHF ASSET MANAGEMENT SAS')
            is_sicav_oddo = data.get("Est ce que le produit fait partie de la Sicav d'Oddo", False)

        if 'product_status' in data:
            is_new_product = data['product_status'].get('is_new_product', False)
            is_new_strategy = data['product_status'].get('is_new_strategy', False)
        else:
            is_new_product = data.get('Le document fait-il référence à un nouveau Produit', False)
            is_new_strategy = data.get('Le document fait-il référence à une nouvelle Stratégie', False)

        inception_date = data.get('fund_info', {}).get('inception_date')
        fund_age_months = None
        fund_age_years = None

        if inception_date:
            try:
                if isinstance(inception_date, str) and inception_date.startswith('XX'):
                    inception_date = inception_date.replace('XX', '01')

                for fmt in ['%d/%m/%Y', '%B %Y', '%m/%Y', '%Y-%m-%d']:
                    try:
                        inception = datetime.strptime(str(inception_date), fmt)
                        delta = datetime.now() - inception
                        fund_age_months = delta.days / 30.44
                        fund_age_years = delta.days / 365.25
                        break
                    except Exception:
                        continue
            except Exception as e:
                print(f"⚠️ Erreur parsing date '{inception_date}': {e}")

        performance_constraints = self._get_performance_constraints(
            fund_age_months, fund_age_years, client_type, language
        )

        # NEW: exigences liées au benchmark
        benchmark = data.get('fund_info', {}).get('benchmark')
        benchmark_requirements = {
            'has_benchmark': benchmark is not None,
            'must_compare_performance_to_benchmark': benchmark is not None,
            'must_show_benchmark_on_performance_slide': benchmark is not None,
            'forbid_other_benchmarks_for_comparison': benchmark is not None
        }

        return {
            'client_type': client_type,
            'language': language,
            'management_entity': management_entity,
            'is_sicav_oddo': is_sicav_oddo,
            'is_new_product': is_new_product,
            'is_new_strategy': is_new_strategy,
            'fund_age_months': fund_age_months,
            'fund_age_years': fund_age_years,
            'performance_constraints': performance_constraints,
            'benchmark_requirements': benchmark_requirements
        }

    def _get_performance_constraints(
        self,
        age_months: Optional[float],
        age_years: Optional[float],
        client_type: str,
        language: str
    ) -> Dict[str, Any]:
        """Détermine les contraintes d'affichage des performances"""
        constraints = {
            'can_display_performance': True,
            'minimum_period_years': 10,
            'can_display_ytd': False,
            'special_rules': []
        }

        # 🔻 Nouveau : prudence si on ne connaît pas l'âge du fonds
        if age_months is None:
            constraints['can_display_performance'] = False
            constraints['special_rules'].append(
                "Âge du fonds inconnu : vérifier manuellement avant d'afficher les performances"
            )
            return constraints

        if age_months < 12:
            constraints['can_display_performance'] = False
            constraints['special_rules'].append(
                "Fonds avec moins de 12 mois : affichage performances interdit"
            )
            return constraints

        if age_years and age_years >= 1:
            constraints['can_display_ytd'] = True

        if age_years and age_years < 10:
            constraints['minimum_period_years'] = age_years
            constraints['special_rules'].append(
                f"Afficher performances depuis création ({age_years:.1f} ans)"
            )

        if language == 'DE':
            constraints['special_rules'].append(
                "Allemagne: inclure frais de souscription max (1ère année)"
            )

        return constraints

    def _get_applicable_rules(self, data: Dict, derived: Dict) -> List[str]:
        """Détermine les règles applicables"""
        rules = [
            'REGLE_1_GENERALE',
            'REGLE_2_PAGE_GARDE',
            'REGLE_3_SLIDE_2',
            'REGLE_4_PAGES_SUIVANTES',
            'REGLE_5_PAGE_FIN'
        ]

        if derived['client_type'] == 'retail':
            rules.extend(['REGLE_RETAIL_DISCLAIMERS', 'REGLE_RETAIL_GLOSSAIRE'])
        else:
            rules.extend(['REGLE_PRO_DISCLAIMERS', 'REGLE_PRO_DO_NOT_DISCLOSE'])

        esg_article = data.get('fund_info', {}).get('esg_sfdr_article')
        if esg_article:
            rules.append('REGLE_4.1_ESG')
            if 'Article 8' in str(esg_article) or 'Article 9' in str(esg_article):
                rules.append('REGLE_ESG_COMMUNICATION')

        if derived['performance_constraints']['can_display_performance']:
            rules.append('REGLE_4.3_PERFORMANCES')

        rules.append('REGLE_4.2_MENTIONS_VALEURS')

        if not derived['is_new_product']:
            rules.append('REGLE_REGISTRATION_FUNDS')

        if derived['is_new_strategy']:
            rules.append('REGLE_NEW_OFFER_DISCLAIMER')

        # NEW: règle explicite liée à la présence d'un benchmark
        if data.get('fund_info', {}).get('benchmark'):
            rules.append('REGLE_BENCHMARK_MENTION')

        return list(set(rules))

    def _get_required_disclaimers(
        self,
        data: Dict,
        derived: Dict,
        rules: List[str]
    ) -> List[Dict[str, Any]]:
        """Détermine les disclaimers requis avec textes exacts"""
        disclaimers: List[Dict[str, Any]] = []
        client_type = derived['client_type']

        # 1. OBAM Presentation (toujours)
        disclaimers.append(
            self._create_disclaimer('OBAM_Presentation', client_type, 'slide_2', True, 'always')
        )

        # 2. Commercial documentation selon société de gestion
        mgmt_entity = derived['management_entity']
        if 'GmbH' in mgmt_entity:
            doc_id = 'Commercial_doc_OBAM_GmbH'
        elif 'Lux' in mgmt_entity:
            doc_id = 'Commercial_doc_OBAM_Lux'
        else:
            doc_id = 'Commercial_doc_OBAM_SAS'

        disclaimers.append(
            self._create_disclaimer(doc_id, client_type, 'slide_2', True, 'always')
        )

        # 3. Performance (seulement si autorisé par les contraintes)
        if derived['performance_constraints']['can_display_performance']:
            disclaimers.append(
                self._create_disclaimer(
                    'Performance', client_type, 'near_performance', True, 'if_performance_displayed'
                )
            )

        # 4. ESG + SFDR articles (6 / 8 / 9)
        esg_article = data.get('fund_info', {}).get('esg_sfdr_article')
        if esg_article:
            # ESG Risk générique
            disclaimers.append(
                self._create_disclaimer(
                    'ESG_Risk', client_type, 'near_esg_mention', True, 'if_esg_mentioned'
                )
            )

            # SFDR spécifique
            esg_str = str(esg_article)
            if "Article 8" in esg_str:
                disclaimers.append(
                    self._create_disclaimer(
                        'SFDR_ART_8',
                        client_type,
                        'near_esg_mention',
                        True,
                        'if_esg_mentioned'
                    )
                )
            elif "Article 9" in esg_str:
                disclaimers.append(
                    self._create_disclaimer(
                        'SFDR_ART_9',
                        client_type,
                        'near_esg_mention',
                        True,
                        'if_esg_mentioned'
                    )
                )
            elif "Article 6" in esg_str:
                disclaimers.append(
                    self._create_disclaimer(
                        'SFDR_ART_6',
                        client_type,
                        'near_esg_mention',
                        True,
                        'if_esg_mentioned'
                    )
                )

        # 5. Opinion
        disclaimers.append(
            self._create_disclaimer('Opinion', client_type, 'near_opinion', True, 'if_opinion_expressed')
        )

        # 6. SRI
        disclaimers.append(
            self._create_disclaimer('SRI_marketing', client_type, 'same_slide_as_sri', True, 'if_sri_displayed')
        )

        # 7. Issuers mentioned
        disclaimers.append(
            self._create_disclaimer(
                'Issuers_mentioned',
                client_type,
                'near_company_mention',
                True,
                'if_company_mentioned'
            )
        )

        # 8. New strategy
        if derived['is_new_strategy']:
            disclaimers.append(
                self._create_disclaimer(
                    'New_offer_strategy_only', client_type, 'slide_2', True, 'always'
                )
            )

        # 9. Disclaimers additionnels selon les pays (optionnels)
        distribution = data.get('distribution', {})
        target_countries = distribution.get('target_countries', []) or []
        target_countries_lower = [c.lower() for c in target_countries]

        # Suisse
        if 'switzerland' in target_countries_lower or 'suisse' in target_countries_lower:
            for disc_id in [
                'Additional_DE_LUX_Switzerland',
                'Additional_FR_Switzerland',
                'Additional_SICAV_LUX_Switzerland',
                'Additional_SICAV_FR_Switzerland'
            ]:
                disclaimers.append(
                    self._create_disclaimer(
                        disc_id,
                        client_type,
                        'end_of_document',
                        False,  # optionnel
                        'if_marketed_in_switzerland'
                    )
                )

        # Allemagne
        if 'germany' in target_countries_lower or 'deutschland' in target_countries_lower:
            disclaimers.append(
                self._create_disclaimer(
                    'Additional_German_pro_performance',
                    client_type,
                    'near_performance',
                    False,  # optionnel
                    'if_marketed_in_germany'
                )
            )

        return disclaimers

    def _create_disclaimer(
        self,
        disclaimer_id: str,
        client_type: str,
        location: str,
        mandatory: bool,
        condition: str
    ) -> Dict[str, Any]:
        """Crée un objet disclaimer avec le texte exact"""
        disclaimer_info = self.ref.disclaimers.get(disclaimer_id, {})

        if client_type == 'retail':
            text = disclaimer_info.get('text_retail')
        else:
            text = disclaimer_info.get('text_professional')

        return {
            'id': disclaimer_id,
            'original_name': disclaimer_info.get('original_name', disclaimer_id),
            'client_type': client_type,
            'location': location,
            'mandatory': mandatory,
            'condition': condition,
            'text': text,
            'has_text': text is not None and len(str(text)) > 0
        }

    def _check_registration(self, data: Dict, derived: Dict) -> Dict[str, Any]:
        """Vérifie les pays autorisés dans Registration of Funds"""
        if derived['is_new_product']:
            return {
                'status': 'new_product',
                'message': 'Nouveau produit : pas encore dans Registration of Funds',
                'authorized_countries': []
            }

        fund_name = data.get('fund_info', {}).get('fund_name')

        if not fund_name or self.ref.registration_df.empty:
            return {
                'status': 'not_checked',
                'message': 'Impossible de vérifier Registration',
                'authorized_countries': []
            }

        try:
            fund_row = self.ref.registration_df[
                self.ref.registration_df.iloc[:, 0].astype(str).str.contains(
                    fund_name, case=False, na=False, regex=False
                )
            ]

            if fund_row.empty:
                keywords = fund_name.split()[:4]
                for keyword in keywords:
                    if len(keyword) > 4:
                        fund_row = self.ref.registration_df[
                            self.ref.registration_df.iloc[:, 0].astype(str).str.contains(
                                keyword, case=False, na=False, regex=False
                            )
                        ]
                        if not fund_row.empty:
                            break

            if fund_row.empty:
                return {
                    'status': 'not_found',
                    'message': f"Fonds '{fund_name}' non trouvé dans Registration",
                    'authorized_countries': []
                }

            authorized = []
            for col in self.ref.registration_df.columns[1:]:
                cell_value = fund_row[col].values[0]
                if pd.notna(cell_value) and str(cell_value).upper() in ['TRUE', 'YES', 'X', '1', 'OUI']:
                    authorized.append(col)

            return {
                'status': 'found',
                'fund_name_matched': fund_row.iloc[0, 0],
                'authorized_countries': authorized
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f"Erreur: {str(e)}",
                'authorized_countries': []
            }

    def _build_exploitation(
        self,
        data: Dict[str, Any],
        derived: Dict[str, Any],
        rules: List[str],
        required_disclaimers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Construit un bloc 'exploitation' :
        pour chaque info extraite importante, indique :
        - valeur
        - règles déclenchées
        - disclaimers associés
        - usages attendus dans les slides / docs
        """
        exploitation: Dict[str, Any] = {}
        fund_info = data.get('fund_info', {})

        # Pour faciliter : index des disclaimers par id
        disc_by_id = {d['id']: d for d in required_disclaimers}

        def add_entry(key: str,
                      value: Any,
                      rule_ids: List[str],
                      disclaimer_ids: List[str],
                      usage_texts: List[str]):
            if value is None:
                return
            triggered_rules = [r for r in rules if r in rule_ids]
            triggered_disclaimers = [d_id for d_id in disclaimer_ids if d_id in disc_by_id]
            exploitation[key] = {
                'value': value,
                'rules_triggered': triggered_rules,
                'disclaimers_triggered': triggered_disclaimers,
                'usage': usage_texts
            }

        # fund_name
        add_entry(
            'fund_name',
            fund_info.get('fund_name'),
            ['REGLE_1_GENERALE', 'REGLE_2_PAGE_GARDE', 'REGLE_REGISTRATION_FUNDS'],
            [],
            [
                "Identifier le fonds dans le document",
                "Assurer la cohérence avec Registration of Funds",
                "Afficher le nom complet sur la page de garde et les slides clés"
            ]
        )

        # benchmark
        add_entry(
            'benchmark',
            fund_info.get('benchmark'),
            [
                'REGLE_4.3_PERFORMANCES',
                'REGLE_3_SLIDE_2',
                'REGLE_4_PAGES_SUIVANTES',
                'REGLE_BENCHMARK_MENTION',
                'REGLE_1_GENERALE'
            ],
            [],
            [
                "Comparer les performances du fonds à l'indice de référence officiel",
                "Afficher le benchmark sur la slide performance",
                "Afficher le benchmark sur la slide objectif/stratégie",
                "Interdire la comparaison avec un autre indice non mentionné dans le prospectus",
                "Utiliser la même nature d'indice (NR/PR/GR) et devise que dans le prospectus"
            ]
        )

        # esg_sfdr_article
        add_entry(
            'esg_sfdr_article',
            fund_info.get('esg_sfdr_article'),
            ['REGLE_4.1_ESG', 'REGLE_ESG_COMMUNICATION'],
            ['ESG_Risk'],
            [
                "Déterminer si le fonds est Article 6, 8 ou 9 SFDR",
                "Afficher les mentions ESG appropriées",
                "Positionner le disclaimer ESG_Risk près des mentions ESG",
                "Vérifier la cohérence entre discours marketing et niveau SFDR"
            ]
        )

        # sri_risk_level
        add_entry(
            'sri_risk_level',
            fund_info.get('sri_risk_level'),
            ['REGLE_4_PAGES_SUIVANTES'],
            ['SRI_marketing'],
            [
                "Afficher l'indicateur synthétique de risque (SRI) sur la slide dédiée",
                "Positionner le disclaimer SRI_marketing sur la même slide",
                "Vérifier la cohérence entre le niveau de risque déclaré et le profil du fonds"
            ]
        )

        # inception_date
        add_entry(
            'inception_date',
            fund_info.get('inception_date'),
            ['REGLE_4.3_PERFORMANCES'],
            [],
            [
                "Calculer l'âge du fonds pour déterminer si les performances peuvent être affichées",
                "Interdire l'affichage de performances si l'âge est inférieur à 12 mois",
                "Déterminer la période minimale de performance à afficher"
            ]
        )

        # distribution_policy
        add_entry(
            'distribution_policy',
            fund_info.get('distribution_policy'),
            ['REGLE_4_PAGES_SUIVANTES'],
            [],
            [
                "Indiquer si le fonds est en accumulation ou distribution",
                "Assurer la cohérence avec la classe de parts mentionnée",
                "Afficher la politique de distribution dans les caractéristiques clés"
            ]
        )

        # fund_type
        add_entry(
            'fund_type',
            fund_info.get('fund_type'),
            ['REGLE_2_PAGE_GARDE', 'REGLE_1_GENERALE'],
            [],
            [
                "Identifier le type de produit (UCITS, ETF, RAIF...)",
                "Adapter les mentions obligatoires (UCITS, réglementation locale)",
                "Afficher le type de fonds sur la page de garde et les slides clés"
            ]
        )

        # currency
        add_entry(
            'currency',
            fund_info.get('currency'),
            ['REGLE_4.3_PERFORMANCES', 'REGLE_4_PAGES_SUIVANTES'],
            [],
            [
                "Indiquer la devise de référence du fonds",
                "S'assurer que les performances sont exprimées dans la devise correcte",
                "Vérifier la cohérence devise fonds / benchmark"
            ]
        )

        # ter
        add_entry(
            'ter',
            fund_info.get('ter'),
            ['REGLE_4_PAGES_SUIVANTES'],
            [],
            [
                "Afficher le Total Expense Ratio si requis par la réglementation locale",
                "Permettre la comparaison des frais avec d'autres produits",
                "S'assurer que le TER affiché est cohérent avec la documentation officielle"
            ]
        )

        # minimum_investment
        add_entry(
            'minimum_investment',
            fund_info.get('minimum_investment'),
            ['REGLE_4_PAGES_SUIVANTES'],
            [],
            [
                "Afficher le montant minimum de souscription",
                "Informer correctement l'investisseur sur l'accessibilité du fonds"
            ]
        )

        # investment_objective
        add_entry(
            'investment_objective',
            fund_info.get('investment_objective'),
            ['REGLE_3_SLIDE_2', 'REGLE_1_GENERALE'],
            [],
            [
                "Remplir la slide Objectif/Stratégie avec un texte court et conforme",
                "Vérifier la cohérence entre objectif, benchmark et profil de risque",
                "Éviter un discours trompeur sur les promesses de performance"
            ]
        )

        # Derived: client_type
        add_entry(
            'client_type',
            derived.get('client_type'),
            [
                'REGLE_RETAIL_DISCLAIMERS',
                'REGLE_RETAIL_GLOSSAIRE',
                'REGLE_PRO_DISCLAIMERS',
                'REGLE_PRO_DO_NOT_DISCLOSE'
            ],
            [],
            [
                "Déterminer si le document est destiné à un client retail ou professionnel",
                "Choisir les disclaimers retail ou pro appropriés",
                "Appliquer les restrictions spécifiques aux clients professionnels"
            ]
        )

        # Derived: is_new_product
        add_entry(
            'is_new_product',
            derived.get('is_new_product'),
            ['REGLE_NEW_OFFER_DISCLAIMER'],
            [],
            [
                "Indiquer si le produit est une nouvelle offre",
                "Adapter les mentions sur l'historique de performance (absence possible)",
                "Expliquer pourquoi le fonds peut ne pas apparaître dans Registration of Funds"
            ]
        )

        # Derived: is_new_strategy
        add_entry(
            'is_new_strategy',
            derived.get('is_new_strategy'),
            ['REGLE_NEW_OFFER_DISCLAIMER'],
            ['New_offer_strategy_only'],
            [
                "Identifier les documents présentant une nouvelle stratégie",
                "Ajouter le disclaimer New_offer_strategy_only sur la slide 2",
                "Informer que la stratégie n'a pas encore de track record complet"
            ]
        )

        return exploitation


def run_compliance_check(
    metadata_path: str,
    prospectus_path: str,
    disclaimers_path: str,
    registration_path: str,
    api_key: str,
    output_path: str = "enriched_context.json"
) -> Dict[str, Any]:
    """Fonction principale : vérifie la conformité d'un document"""
    print("=" * 70)
    print("🔍 COMPLIANCE MARKETING - LAYER 1.5 - METADATA ENRICHMENT")
    print("=" * 70 + "\n")

    ref_loader = ReferenceDataLoader(
        disclaimers_path=disclaimers_path,
        registration_path=registration_path
    )

    print(f"📄 Chargement metadata: {Path(metadata_path).name}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print("✅ Metadata chargé\n")

    extractor = ProspectusExtractor(api_key=api_key)
    prospectus_data = extractor.extract(prospectus_path)

    engine = ComplianceEngine(ref_loader)
    enriched_context = engine.enrich(metadata, prospectus_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_context, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"✅ ANALYSE TERMINÉE")
    print("=" * 70)
    print(f"📊 Résumé:")
    print(f"   Fonds: {enriched_context['prospectus_parsed'].get('fund_name', 'N/A')}")
    client_type = enriched_context['derived']['client_type']
    print(f"   Client: {client_type}")

    age_months = enriched_context['derived']['fund_age_months']
    if age_months:
        print(f"   Âge: {age_months:.1f} mois")
    else:
        print(f"   Âge: Non déterminé")

    print(f"   Règles applicables: {len(enriched_context['applicable_rules'])}")

    disclaimers_with_text = sum(
        1 for d in enriched_context['required_disclaimers'] if d['has_text']
    )
    total_disclaimers = len(enriched_context['required_disclaimers'])
    print(f"   Disclaimers requis: {disclaimers_with_text}/{total_disclaimers} avec texte")

    reg_status = enriched_context['registration']['status']
    if reg_status == 'found':
        countries = enriched_context['registration']['authorized_countries']
        if countries:
            print(f"   Pays autorisés: {len(countries)} ({', '.join(countries[:3])}...)")
        else:
            print("   Pays autorisés: 0")
    else:
        print(f"   Registration: {reg_status}")

    print(f"\n💾 Résultat sauvegardé: {output_path}")
    print("=" * 70)

    return enriched_context


if __name__ == "__main__":
    CONFIG = {
        "metadata_path": "example_2/metadata.json",
        "prospectus_path": "example_2/prospectus.pdf" if Path("example_2/prospectus.pdf").exists() else "example_2/prospectus.docx",
        "disclaimers_path": "inputs/disclamer.json",
        "registration_path": "inputs/Registration abroad of Funds_20251008.xlsx",
        # ⚠️ Remplace par ta vraie clé API :
        "api_key": "sk-7c0b80cf494746f580cc5ba555d739b2",
        "output_path": "example_2/outputs/enriched_context_test.json"
    }

    try:
        result = run_compliance_check(**CONFIG)

        print("\n📝 DISCLAIMERS REQUIS:")
        print("-" * 70)
        for disc in result['required_disclaimers']:
            status = "✅" if disc['has_text'] else "⚠️"
            print(f"\n{status} {disc['original_name']}")
            print(f"   ID: {disc['id']}")
            print(f"   Condition: {disc['condition']}")
            print(f"   Location: {disc['location']}")
            if disc['has_text']:
                text_preview = disc['text'][:150] + "..." if len(disc['text']) > 150 else disc['text']
                print(f"   Texte: {text_preview}")
            else:
                print(f"   ⚠️  Texte manquant")

        print("\n" + "=" * 70)
        print("✅ LAYER 1.5 TERMINÉ AVEC SUCCÈS")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()