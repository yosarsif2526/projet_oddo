# ═══════════════════════════════════════════════════════════════════
# FUSION COMPLÈTE OPTIMISÉE : PROSPECTUS + PPTX → metadata_final.json
# Version avec LLM pour analyse des incohérences + organisation claire
# ═══════════════════════════════════════════════════════════════════

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import httpx
from openai import OpenAI


class OptimizedFusion:
    """
    Fusion optimisée avec :
    - Organisation claire des données (pas de duplications)
    - Détection précise des incohérences avec analyse LLM
    - Rapport TXT détaillé et structuré
    """
    
    def __init__(self, 
                 prospectus_json_path: str,
                 pptx_json_path: str,
                 api_key: str = "sk-7c0b80cf494746f580cc5ba555d739b2"):
        self.prospectus_data = self._load_json(prospectus_json_path)
        self.pptx_data = self._load_json(pptx_json_path)
        self.api_key = api_key
        self.incoherences: List[Dict[str, Any]] = []
        
        print("=" * 80)
        print("🔗 FUSION OPTIMISÉE : PROSPECTUS + PPTX")
        print("=" * 80 + "\n")
    
    def _load_json(self, path: str) -> Dict[str, Any]:
        """Charge un fichier JSON."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Erreur chargement {path}: {e}")
            return {}
    
    def fusion_complete(self) -> Dict[str, Any]:
        """
        Fusion complète avec structure claire.
        
        Structure du JSON final (SANS DUPLICATIONS) :
        {
            "fund_info": { ... tous les champs fusionnés ... },
            "document_info": { ... },
            "management_info": { ... },
            "distribution": { ... },
            "product_status": { ... },
            "compliance": {
                "derived": { ... },
                "applicable_rules": [ ... ],
                "required_disclaimers": [ ... ],
                "registration": { ... },
                "exploitation": { ... }
            },
            "extraction_metadata": {
                "prospectus_extraction": { ... },
                "pptx_extraction": { ... },
                "performance_analysis": { ... }
            },
            "fusion_report": {
                "statistics": { ... },
                "incoherences": [ ... ]
            }
        }
        """
        print("🔄 Début de la fusion...\n")
        
        # 1. Extraire fund_info des deux sources
        prospectus_fund_info = self._extract_prospectus_fund_info()
        pptx_fund_info = self.pptx_data.get('fund_info_complete', {})
        
        # 2. Fusion avec détection d'incohérences
        fund_info_merged = self._merge_all_fields(
            prospectus_fund_info, 
            pptx_fund_info
        )
        
        # 3. Analyse LLM des incohérences si détectées
        if self.incoherences:
            print(f"\n⚠️  {len(self.incoherences)} incohérences détectées")
            print("🤖 Analyse des incohérences avec LLM...\n")
            self._analyze_incoherences_with_llm()
        
        # 4. Construction du JSON final STRUCTURÉ
        metadata_final = {
            # ═══════════════════════════════════════════════════════
            # SECTION 1 : INFORMATIONS FONDS (fusionnées)
            # ═══════════════════════════════════════════════════════
            "fund_info": self._clean_fund_info(fund_info_merged),
            
            # ═══════════════════════════════════════════════════════
            # SECTION 2 : MÉTADONNÉES DOCUMENT
            # ═══════════════════════════════════════════════════════
            "document_info": self._extract_document_info(),
            
            # ═══════════════════════════════════════════════════════
            # SECTION 3 : GESTION & DISTRIBUTION
            # ═══════════════════════════════════════════════════════
            "management_info": self._extract_management_info(fund_info_merged),
            "distribution": self._extract_distribution_info(fund_info_merged),
            
            # ═══════════════════════════════════════════════════════
            # SECTION 4 : STATUT PRODUIT
            # ═══════════════════════════════════════════════════════
            "product_status": self._extract_product_status(),
            
            # ═══════════════════════════════════════════════════════
            # SECTION 5 : COMPLIANCE (regroupé)
            # ═══════════════════════════════════════════════════════
            "compliance": {
                "derived": self.prospectus_data.get('derived', {}),
                "applicable_rules": self.prospectus_data.get('applicable_rules', []),
                "required_disclaimers": self.prospectus_data.get('required_disclaimers', []),
                "registration": self.prospectus_data.get('registration', {}),
                "exploitation": self.prospectus_data.get('exploitation', {})
            },
            
            # ═══════════════════════════════════════════════════════
            # SECTION 6 : MÉTADONNÉES EXTRACTION (regroupé)
            # ═══════════════════════════════════════════════════════
            "extraction_metadata": {
                "prospectus_extraction": {
                    "date": self.prospectus_data.get('enrichment_date'),
                    "source": "enriched_context.json"
                },
                "pptx_extraction": {
                    "date": self.pptx_data.get('extraction_metadata', {}).get('extraction_date'),
                    "source_file": self.pptx_data.get('extraction_metadata', {}).get('source_file'),
                    "total_slides": self.pptx_data.get('extraction_metadata', {}).get('total_slides'),
                    "method": self.pptx_data.get('extraction_metadata', {}).get('extraction_method')
                },
                "performance_analysis": self.pptx_data.get('performance_flags', {})
            },
            
            # ═══════════════════════════════════════════════════════
            # SECTION 7 : RAPPORT DE FUSION
            # ═══════════════════════════════════════════════════════
            "fusion_report": self._generate_fusion_report(fund_info_merged)
        }
        
        return metadata_final
    
    def _extract_prospectus_fund_info(self) -> Dict[str, Any]:
        """Extrait fund_info du prospectus."""
        metadata = self.prospectus_data.get('metadata', {})
        
        if 'fund_info' in metadata:
            return metadata['fund_info']
        
        # Fallback : extraire depuis prospectus_parsed
        return self.prospectus_data.get('prospectus_parsed', {})
    
    def _merge_all_fields(self, 
                         prosp_data: Dict[str, Any], 
                         pptx_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fusionne avec détection d'incohérences."""
        merged = {}
        
        # Tous les champs (union)
        all_fields = set(prosp_data.keys()) | set(pptx_data.keys())
        
        # Champs à exclure (métadonnées techniques)
        exclude_fields = {
            'extraction_metadata', 'performance_flags', 'fund_size', 
            'aum', 'minimum_initial_investment_amount'
        }
        
        print("📊 FUSION FIELD-BY-FIELD:")
        print("-" * 80)
        
        for field in sorted(all_fields):
            if field in exclude_fields:
                continue
            
            prosp_value = self._normalize_value(prosp_data.get(field))
            pptx_value = self._normalize_value(pptx_data.get(field))
            
            # CAS 1 : Les deux ont une valeur
            if prosp_value is not None and pptx_value is not None:
                if self._are_values_coherent(prosp_value, pptx_value):
                    merged[field] = prosp_value
                    print(f"✅ {field:<40} COHÉRENT")
                else:
                    # INCOHÉRENT → priorité prospectus
                    merged[field] = prosp_value
                    self._record_incoherence(field, prosp_value, pptx_value)
                    print(f"⚠️  {field:<40} INCOHÉRENT (prospectus gardé)")
            
            # CAS 2 : Seulement prospectus
            elif prosp_value is not None:
                merged[field] = prosp_value
                print(f"🔵 {field:<40} Prospectus seul")
            
            # CAS 3 : Seulement PPTX
            elif pptx_value is not None:
                merged[field] = pptx_value
                print(f"🟢 {field:<40} PPTX seul")
        
        print("-" * 80 + "\n")
        return merged
    
    def _normalize_value(self, value: Any) -> Optional[Any]:
        """Normalise une valeur."""
        if value is None:
            return None
        
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in ['', 'none', 'null', 'n/a', 'na', 'nan']:
                return None
            return value
        
        if isinstance(value, list) and len(value) == 0:
            return None
        
        if isinstance(value, dict) and len(value) == 0:
            return None
        
        return value
    
    def _are_values_coherent(self, val1: Any, val2: Any) -> bool:
        """Vérifie la cohérence entre deux valeurs."""
        if val1 == val2:
            return True
        
        str1 = str(val1).lower().strip()
        str2 = str(val2).lower().strip()
        
        if str1 == str2:
            return True
        
        # Benchmark : "S&P 500 NR" vs "S&P 500 Index (USD, NR)" → similaire
        if 's&p 500' in str1 and 's&p 500' in str2:
            if 'nr' in str1 and 'nr' in str2:
                return True
        
        # Listes
        if isinstance(val1, list) and isinstance(val2, list):
            return set(val1) == set(val2)
        
        return False
    
    def _record_incoherence(self, field: str, prosp_value: Any, pptx_value: Any):
        """Enregistre une incohérence."""
        severity = self._assess_severity(field)
        
        self.incoherences.append({
            'field': field,
            'prospectus_value': prosp_value,
            'pptx_value': pptx_value,
            'resolution': 'prospectus_kept',
            'severity': severity,
            'llm_analysis': None  # Sera rempli par le LLM
        })
    
    def _assess_severity(self, field: str) -> str:
        """Évalue la sévérité d'une incohérence."""
        critical_fields = [
            'fund_name', 'isin_code', 'esg_sfdr_article', 
            'management_company', 'legal_structure'
        ]
        
        important_fields = [
            'benchmark', 'ter', 'inception_date', 'currency',
            'sri_risk_level', 'distribution_policy'
        ]
        
        if field in critical_fields:
            return 'CRITIQUE'
        elif field in important_fields:
            return 'IMPORTANTE'
        else:
            return 'MINEURE'
    
    def _analyze_incoherences_with_llm(self):
        """Analyse les incohérences avec le LLM."""
        if not self.incoherences:
            return
        
        try:
            http_client = httpx.Client(verify=False)
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://tokenfactory.esprit.tn/api",
                http_client=http_client
            )
            
            for inc in self.incoherences:
                prompt = f"""Tu es un expert en analyse de données financières.

INCOHÉRENCE DÉTECTÉE :
Champ : {inc['field']}
Valeur Prospectus : {inc['prospectus_value']}
Valeur PPTX : {inc['pptx_value']}
Sévérité : {inc['severity']}

QUESTIONS :
1. Quelle valeur est probablement correcte ? Pourquoi ?
2. Comment expliquer cette différence ?
3. Quel impact sur la conformité réglementaire ?
4. Action recommandée ?

Réponds en 150 mots maximum, de manière structurée et précise."""

                response = client.chat.completions.create(
                    model="hosted_vllm/Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "system", "content": "Tu es un expert financier précis et concis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                
                analysis = response.choices[0].message.content.strip()
                inc['llm_analysis'] = analysis
                
                print(f"   ✅ Analyse LLM pour '{inc['field']}' terminée")
        
        except Exception as e:
            print(f"   ⚠️  Erreur LLM : {e}")
            for inc in self.incoherences:
                if inc['llm_analysis'] is None:
                    inc['llm_analysis'] = "Analyse LLM non disponible"
    
    def _clean_fund_info(self, fund_info: Dict[str, Any]) -> Dict[str, Any]:
        """Nettoie fund_info en retirant les champs techniques."""
        clean = {}
        exclude = {'extraction_metadata', 'performance_flags'}
        
        for key, value in fund_info.items():
            if key not in exclude:
                clean[key] = value
        
        return clean
    
    def _extract_document_info(self) -> Dict[str, Any]:
        """Extrait document_info."""
        metadata = self.prospectus_data.get('metadata', {})
        
        if 'document_info' in metadata:
            return metadata['document_info']
        
        return {
            'target_audience': 'professional' if metadata.get('Le client est-il un professionnel') else 'retail',
            'language': metadata.get('language', 'FR')
        }
    
    def _extract_management_info(self, fund_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait management_info."""
        metadata = self.prospectus_data.get('metadata', {})
        
        return {
            'management_company': fund_info.get('management_company', 'ODDO BHF ASSET MANAGEMENT SAS'),
            'investment_manager': fund_info.get('investment_manager'),
            'custodian': fund_info.get('custodian'),
            'is_sicav_oddo': metadata.get('Est ce que le produit fait partie de la Sicav d\'Oddo', False)
        }
    
    def _extract_distribution_info(self, fund_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait distribution info."""
        countries = fund_info.get('countries_available_for_sales', [])
        
        if isinstance(countries, str):
            countries = [c.strip() for c in countries.split(',') if c.strip()]
        
        return {
            'target_countries': countries or [],
            'domicile': fund_info.get('domicile'),
            'stock_exchanges': fund_info.get('stock_exchanges'),
            'registration_status': self.prospectus_data.get('registration', {}).get('status', 'not_checked')
        }
    
    def _extract_product_status(self) -> Dict[str, Any]:
        """Extrait product_status."""
        metadata = self.prospectus_data.get('metadata', {})
        
        return {
            'is_new_product': metadata.get('Le document fait-il référence à un nouveau Produit', False),
            'is_new_strategy': metadata.get('Le document fait-il référence à une nouvelle Stratégie', False)
        }
    
    def _generate_fusion_report(self, fund_info: Dict[str, Any]) -> Dict[str, Any]:
        """Génère le rapport de fusion."""
        filled = sum(1 for v in fund_info.values() if v is not None)
        
        return {
            "fusion_date": datetime.now().isoformat(),
            "statistics": {
                "total_fields": len(fund_info),
                "filled_fields": filled,
                "empty_fields": len(fund_info) - filled,
                "incoherent_fields": len(self.incoherences)
            },
            "incoherences_summary": {
                "total": len(self.incoherences),
                "critique": sum(1 for i in self.incoherences if i['severity'] == 'CRITIQUE'),
                "importante": sum(1 for i in self.incoherences if i['severity'] == 'IMPORTANTE'),
                "mineure": sum(1 for i in self.incoherences if i['severity'] == 'MINEURE')
            },
            "incoherences_details": self.incoherences
        }
    
    def generate_txt_report(self, output_path: str = "rapport_incoherences.txt"):
        """Génère le rapport TXT détaillé."""
        if not self.incoherences:
            report = "=" * 80 + "\n"
            report += "RAPPORT D'INCOHÉRENCES - FUSION PROSPECTUS + PPTX\n"
            report += "=" * 80 + "\n\n"
            report += "✅ AUCUNE INCOHÉRENCE DÉTECTÉE\n\n"
            report += f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
            report += "=" * 80 + "\n"
        else:
            report = self._format_detailed_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Rapport TXT généré : {output_path}")
    
    def _format_detailed_report(self) -> str:
        """Formate le rapport détaillé."""
        report = "=" * 80 + "\n"
        report += "RAPPORT D'INCOHÉRENCES - FUSION PROSPECTUS + PPTX\n"
        report += "=" * 80 + "\n\n"
        
        report += f"📅 Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        report += f"📊 Total incohérences : {len(self.incoherences)}\n\n"
        
        # Grouper par sévérité
        by_severity = {
            'CRITIQUE': [i for i in self.incoherences if i['severity'] == 'CRITIQUE'],
            'IMPORTANTE': [i for i in self.incoherences if i['severity'] == 'IMPORTANTE'],
            'MINEURE': [i for i in self.incoherences if i['severity'] == 'MINEURE']
        }
        
        for severity, items in by_severity.items():
            if not items:
                continue
            
            icon = "🔴" if severity == "CRITIQUE" else "🟠" if severity == "IMPORTANTE" else "🟡"
            
            report += f"\n{icon} " + "=" * 76 + "\n"
            report += f"{icon} INCOHÉRENCES {severity}\n"
            report += f"{icon} " + "=" * 76 + "\n\n"
            
            for idx, inc in enumerate(items, 1):
                report += f"[{idx}] CHAMP : {inc['field']}\n"
                report += "-" * 80 + "\n"
                report += f"🔵 Prospectus : {inc['prospectus_value']}\n"
                report += f"🟢 PPTX       : {inc['pptx_value']}\n"
                report += f"✅ Résolution : {inc['resolution']}\n\n"
                
                if inc.get('llm_analysis'):
                    report += "🤖 ANALYSE LLM :\n"
                    report += "-" * 80 + "\n"
                    report += inc['llm_analysis'] + "\n"
                
                report += "\n" + "=" * 80 + "\n\n"
        
        # Recommandations
        report += "\n💡 RECOMMANDATIONS\n"
        report += "=" * 80 + "\n\n"
        report += "1. Vérifier manuellement les incohérences CRITIQUES\n"
        report += "2. Valider la source correcte avec les équipes métier\n"
        report += "3. Mettre à jour la source erronée\n"
        report += "4. Relancer la fusion après correction\n\n"
        report += "=" * 80 + "\n"
        
        return report
    
    def save(self, metadata: Dict[str, Any], 
             json_path: str = "metadata_final.json",
             report_path: str = "rapport_incoherences.txt"):
        """Sauvegarde JSON + rapport TXT."""
        
        # JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 JSON sauvegardé : {json_path}")
        
        # Rapport TXT
        self.generate_txt_report(report_path)
        
        # Résumé
        self._print_summary(metadata)
    
    def _print_summary(self, metadata: Dict[str, Any]):
        """Affiche le résumé."""
        report = metadata['fusion_report']
        stats = report['statistics']
        inc_summary = report['incoherences_summary']
        
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ DE LA FUSION")
        print("=" * 80)
        print(f"✅ Champs remplis : {stats['filled_fields']}/{stats['total_fields']}")
        print(f"❌ Champs vides   : {stats['empty_fields']}")
        
        if inc_summary['total'] > 0:
            print(f"\n⚠️  INCOHÉRENCES : {inc_summary['total']}")
            print(f"   🔴 Critiques   : {inc_summary['critique']}")
            print(f"   🟠 Importantes : {inc_summary['importante']}")
            print(f"   🟡 Mineures    : {inc_summary['mineure']}")
        else:
            print("\n✅ Aucune incohérence détectée")
        
        print("=" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════
def run_optimized_fusion(
    prospectus_json: str = "enriched_context.json",
    pptx_json: str = "enriched_with_pptx_improved.json",
    output_json: str = "metadata_final.json",
    output_report: str = "rapport_incoherences.txt",
    api_key: str = "sk-7c0b80cf494746f580cc5ba555d739b2"
) -> Dict[str, Any]:
    """
    Fusion optimisée avec analyse LLM des incohérences.
    """
    
    fusioner = OptimizedFusion(prospectus_json, pptx_json, api_key)
    
    # Fusion
    metadata_final = fusioner.fusion_complete()
    
    # Sauvegarde
    fusioner.save(metadata_final, output_json, output_report)
    
    return metadata_final


# ═══════════════════════════════════════════════════════════════════
# EXEMPLE D'UTILISATION
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    
    result = run_optimized_fusion(
        #prospectus_json="enriched_context (11).json",
        #pptx_json="enriched_improved_complete.json",
        prospectus_json="example_2/outputs/enriched_context_test.json",
        pptx_json="example_2/outputs/enriched_improved_complete_test.json",
        output_json="example_2/outputs/metadata_final_optimized_test.json",
        output_report="example_2/outputs/rapport_incoherences_detaille.txt"
    )
    
    print("\n✅ FUSION OPTIMISÉE TERMINÉE")
    print(f"📄 JSON final : metadata_final_optimized.json")
    print(f"📄 Rapport    : rapport_incoherences_detaille.txt")