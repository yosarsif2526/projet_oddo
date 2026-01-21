# Generated from: filtrage_des_regles.ipynb
# Converted at: 2025-12-16T19:37:50.182Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

"""
PHASE 1 CORRIGÉ - LOGIQUE DE FILTRAGE CORRECTE
==============================================

PRINCIPE:
- Filtrage metadata → Garde TOUTES les règles applicables
- Classification → Juste pour info/stats (critique/haute/normale)
- Output final → TOUTES les règles applicables (pas de limite à 15)

✅ Si règle passe le filtre metadata → ELLE EST DANS L'OUTPUT
"""

import json
import pickle
from typing import List, Dict, Any
from pathlib import Path
import faiss

# Configuration (simplifiée pour focus sur logique)
http_client = None
llm_client = None

def call_llm(prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
    return '{"categories": ["disclaimers", "slide_2"], "reasoning": {}}'


class Phase1_CORRECT:
    """Phase 1 avec logique de filtrage CORRECTE"""

    # Règles TOUJOURS applicables (aucune condition)
    UNIVERSAL_RULES = {
        '1.3',   # Citation sources
        '1.6',   # Formatage avertissements
        '1.7',   # Opinions atténuées
        '1.8',   # Stratégie conforme prospectus
        '1.10',  # Pas de limites internes
        '1.13',  # Pas de confusion stratégie/fonds
        '1.15',  # Pas d'autres fonds mentionnés
        '2.1',   # Page de garde
        '4.1',   # Ne pas commencer par performance
        '4.4',   # Caractéristiques détaillées en fin
        '4.5',   # Conformité données légales
        '4.6',   # Validation présentation
        '4.7',   # Équipe gestion disclaimer
        '4.2.1', '4.2.2', '4.2.3', '4.2.5', '4.2.6',
        '4.2.7', '4.2.8', '4.2.9', '4.2.10',
        '5.1',   # Page de fin
    }

    # Règles OBLIGATOIRES (jamais éliminées)
    MANDATORY_RULES = {
        '1.1',   # Disclaimers retail
        '1.4',   # SRI
        '1.5',   # Glossaire retail
        '2.1',   # Page de garde
        '3.1',   # Slide 2 disclaimer
        '3.2',   # Slide 2 profil risque
        '5.1',   # Page de fin
    }

    # Variables Phase 2 (ignorées en Phase 1)
    PHASE2_VARS = {
        'slide_number', 'has_charts', 'has_tables',
        'has_images', 'footnote_count', 'element_count',
        'is_bold', 'element_type'
    }

    def __init__(self,
                 rules_json_path: str,
                 metadata_pkl_path: str,
                 faiss_index_path: str,
                 enriched_context_path: str,
                 disclaimers_path: str):

        print("="*70)
        print("📁 CHARGEMENT - VERSION CORRIGÉE")
        print("="*70)

        # 1. Règles JSON
        with open(rules_json_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
            self.all_rules = rules_data.get('rules', rules_data)
            self.total_rules = len(self.all_rules)
        print(f"✅ {self.total_rules} règles chargées")

        # 2. Metadata
        if Path(metadata_pkl_path).exists():
            with open(metadata_pkl_path, 'rb') as f:
                metadata = pickle.load(f)
            self.rule_ids_mapping = metadata['rule_ids']
            print(f"✅ Metadata: {len(self.rule_ids_mapping)} IDs")
        else:
            self.rule_ids_mapping = [r['rule_id'] for r in self.all_rules]

        # 3. FAISS
        if Path(faiss_index_path).exists():
            self.faiss_index = faiss.read_index(faiss_index_path)
            print(f"✅ FAISS: {self.faiss_index.ntotal} vecteurs")
        else:
            self.faiss_index = None
            print("⚠️ FAISS absent")

        # 4. Context
        with open(enriched_context_path, 'r', encoding='utf-8') as f:
            self.context = json.load(f)

        # 5. Disclaimers
        if Path(disclaimers_path).exists():
            with open(disclaimers_path, 'r', encoding='utf-8') as f:
                self.disclaimers = json.load(f)
        else:
            self.disclaimers = {}

        # Extraire variables métier
        self.metadata = self.context.get('metadata', {})
        self.fund_info = self.context.get('fund_info_complete', self.context.get('fund_info', {}))
        self.derived = self.context.get('derived', {})

        self.client_type = self.derived.get('client_type', 'retail')
        self.is_new_product = self.derived.get('is_new_product', False)
        self.is_new_strategy = self.derived.get('is_new_strategy', False)
        self.language = self.derived.get('language', 'FR')

        self.fund_name = self.fund_info.get('fund_name', 'N/A')
        self.fund_type = self.fund_info.get('fund_type', '')
        self.benchmark = self.fund_info.get('benchmark', 'N/A')
        self.has_benchmark = self.benchmark not in ['N/A', '', None]
        self.esg_sfdr_article = self.fund_info.get('esg_sfdr_article', 'Article 6')
        self.sri_risk_level = self.fund_info.get('sri_risk_level')
        self.target_countries = self.metadata.get('target_countries', ['France'])

        perf_constraints = self.derived.get('performance_constraints', {})
        self.fund_age_years = perf_constraints.get('fund_age_years', 10)
        self.can_display_performance = perf_constraints.get('can_display_performance', False)

        print(f"\n📊 Document: {self.client_type.upper()}")
        print(f"📊 Fonds: {self.fund_name}")
        print(f"📊 SFDR: {self.esg_sfdr_article}")

    # ========================================
    # NORMALISATION & ÉVALUATION CONDITIONS
    # ========================================

    def normalize_condition(self, condition: str) -> str:
        """Normalise condition (true→True, AND→and)"""
        if not condition or condition == 'null':
            return 'True'

        normalized = str(condition)
        normalized = normalized.replace(' true', ' True')
        normalized = normalized.replace(' false', ' False')
        normalized = normalized.replace('== true', '== True')
        normalized = normalized.replace('== false', '== False')
        normalized = normalized.replace(' AND ', ' and ')
        normalized = normalized.replace(' OR ', ' or ')

        return normalized

    def contains_phase2_var(self, condition: str) -> bool:
        """Vérifie si condition contient variable Phase 2"""
        if not condition:
            return False
        return any(var in condition for var in self.PHASE2_VARS)

    def evaluate_metadata_condition(self, condition: str, rule_id: str) -> bool:
        """
        Évalue condition metadata

        RÈGLES:
        1. Pas de condition → TRUE (conserve)
        2. Variable Phase 2 → TRUE (évalué plus tard)
        3. Règle universelle → TRUE (toujours applicable)
        4. Règle obligatoire → TRUE (force-keep)
        5. Condition invalide → TRUE (principe précaution)
        6. Sinon → évalue
        """

        # Règle 1: Pas de condition
        if not condition or condition == 'null':
            return True

        # Règle 2: Variable Phase 2
        if self.contains_phase2_var(condition):
            return True

        # Règle 3: Règle universelle
        if rule_id in self.UNIVERSAL_RULES:
            return True

        # Règle 4: Règle obligatoire
        if rule_id in self.MANDATORY_RULES:
            return True

        # Règle 5 & 6: Évaluation
        try:
            normalized = self.normalize_condition(condition)

            variables = {
                'client_type': self.client_type,
                'is_new_product': self.is_new_product,
                'is_new_strategy': self.is_new_strategy,
                'language': self.language,
                'fund_type': self.fund_type,
                'has_benchmark': self.has_benchmark,
                'esg_sfdr_article': self.esg_sfdr_article,
                'benchmark': self.benchmark,
                'sri_risk_level': self.sri_risk_level,
                'target_countries': self.target_countries,
                'fund_age_years': self.fund_age_years,
                'can_display_performance': self.can_display_performance,
                # Valeurs par défaut
                'has_performance_content': True,
                'ter': self.fund_info.get('ter'),
                'distribution_policy': self.fund_info.get('distribution_policy'),
                'esg_approach': self.context.get('esg_approach'),
                'is_precommercialisation': False,
                'is_dated_fund': False,
                'is_client_specific': False,
            }

            condition_eval = normalized

            for var, val in variables.items():
                if var in condition_eval:
                    if val is None:
                        return True  # Variable manquante → conserve
                    elif isinstance(val, bool):
                        condition_eval = condition_eval.replace(var, str(val))
                    elif isinstance(val, str):
                        condition_eval = condition_eval.replace(var, f"'{val}'")
                    elif isinstance(val, list):
                        condition_eval = condition_eval.replace(var, str(val))
                    else:
                        condition_eval = condition_eval.replace(var, str(val))

            result = bool(eval(condition_eval, {"__builtins__": {}}, {}))
            return result

        except Exception as e:
            # Règle 5: Invalide → conserve
            print(f"⚠️ Condition invalide [{rule_id}]: {condition}")
            return True

    # ========================================
    # FILTRAGE METADATA (LOGIQUE CORRECTE)
    # ========================================

    def filter_by_metadata(self) -> List[Dict]:
        """
        Filtrage strict par metadata

        ✅ LOGIQUE CORRECTE:
        - Si règle obligatoire → CONSERVE
        - Si règle universelle → CONSERVE
        - Si applicable_to match → vérifie condition
        - Si condition OK → CONSERVE

        RÉSULTAT: TOUTES les règles qui passent ces critères
        """

        applicable_rules = []
        eliminated_rules = []

        for rule in self.all_rules:
            rule_id = rule.get('rule_id')

            # Force-keep règles obligatoires
            if rule_id in self.MANDATORY_RULES:
                applicable_rules.append(rule)
                continue

            # Force-keep règles universelles
            if rule_id in self.UNIVERSAL_RULES:
                applicable_rules.append(rule)
                continue

            # Vérifier applicable_to
            applicable_to = rule.get('applicable_to', ['retail', 'professional'])
            if self.client_type not in applicable_to:
                eliminated_rules.append({
                    'rule_id': rule_id,
                    'reason': f'Client type: {self.client_type} not in {applicable_to}'
                })
                continue

            # Vérifier metadata_conditions
            condition = rule.get('triggers', {}).get('metadata_conditions')
            if not self.evaluate_metadata_condition(condition, rule_id):
                eliminated_rules.append({
                    'rule_id': rule_id,
                    'reason': f'Condition failed: {condition}'
                })
                continue

            # ✅ Règle APPLICABLE
            applicable_rules.append(rule)

        # Rapport
        print("\n" + "="*70)
        print("🔍 FILTRAGE METADATA (VERSION CORRECTE)")
        print("="*70)
        print(f"📊 Règles initiales: {self.total_rules}")
        print(f"✅ Règles APPLICABLES: {len(applicable_rules)}")
        print(f"❌ Règles éliminées: {len(eliminated_rules)}")

        mandatory_kept = sum(1 for r in applicable_rules if r['rule_id'] in self.MANDATORY_RULES)
        universal_kept = sum(1 for r in applicable_rules if r['rule_id'] in self.UNIVERSAL_RULES)

        print(f"🔒 Obligatoires: {mandatory_kept}/{len(self.MANDATORY_RULES)}")
        print(f"🌍 Universelles: {universal_kept}/{len(self.UNIVERSAL_RULES)}")

        # Distribution
        sections = {}
        for rule in applicable_rules:
            sec = rule.get('section', 'Unknown')
            sections[sec] = sections.get(sec, 0) + 1

        print(f"\n📂 Distribution:")
        for sec, count in sorted(sections.items()):
            print(f"   {sec}: {count} règles")

        return applicable_rules

    # ========================================
    # CLASSIFICATION (JUSTE POUR INFO)
    # ========================================

    def classify_rules_by_priority(self, rules: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Classifie règles par priorité

        ⚠️ IMPORTANT: C'est juste pour STATS/INFO
        Toutes les règles seront dans l'output final !
        """

        classification = {
            'critique': [],
            'haute': [],
            'normale': []
        }

        critical_checks = {
            'disclaimer_verification',
            'performance_disclaimer',
            'prospectus_coherence',
            'registration_verification',
            'performance_restriction'
        }

        high_priority_checks = {
            'sri_verification',
            'linguistic_restrictions',
            'performance_benchmark_comparison',
            'esg_communication',
            'slide_order',
            'citation_verification',
            'page_garde_structure',
            'slide_2_disclaimer',
            'slide_2_risk_profile',
            'page_fin_structure',
        }

        for rule in rules:
            severity = rule.get('severity', 'low')
            check_type = rule.get('check_type', '')

            if severity == 'high' and check_type in critical_checks:
                classification['critique'].append(rule)
            elif severity in ['high', 'medium'] and check_type in high_priority_checks:
                classification['haute'].append(rule)
            else:
                classification['normale'].append(rule)

        print("\n" + "="*70)
        print("🎯 CLASSIFICATION (INFO UNIQUEMENT)")
        print("="*70)
        print(f"🔴 Critique: {len(classification['critique'])}")
        print(f"🟠 Haute: {len(classification['haute'])}")
        print(f"🟢 Normale: {len(classification['normale'])}")
        print(f"\n✅ TOUTES seront dans l'output final")

        return classification

    # ========================================
    # PIPELINE COMPLET (LOGIQUE CORRECTE)
    # ========================================

    def run_phase1(self) -> Dict:
        """
        Pipeline Phase 1 CORRECT

        ✅ LOGIQUE:
        1. Filtrer par metadata → garde TOUTES applicables
        2. Classifier pour stats → info seulement
        3. Output final = TOUTES les règles applicables
        """

        print("\n" + "="*70)
        print("🚀 PHASE 1 - VERSION CORRECTE")
        print("="*70)

        # Étape 1: Filtrage metadata
        applicable_rules = self.filter_by_metadata()

        # Étape 2: Classification (juste info)
        classified = self.classify_rules_by_priority(applicable_rules)

        # Étape 3: ✅ OUTPUT FINAL = TOUTES LES RÈGLES APPLICABLES
        final_rules = applicable_rules  # ← CHANGEMENT CRUCIAL !

        # PAS DE LIMITE ARBITRAIRE !
        # Toutes les règles qui ont passé le filtre metadata sont conservées

        # Statistiques
        stats = {
            'total_initial': self.total_rules,
            'after_metadata_filter': len(applicable_rules),
            'final_count': len(final_rules),  # = len(applicable_rules)
            'reduction_rate': round((1 - len(final_rules)/self.total_rules) * 100, 1),
            'by_priority': {
                'critique': len(classified['critique']),
                'haute': len(classified['haute']),
                'normale': len(classified['normale'])
            },
            'mandatory_kept': sum(1 for r in final_rules if r['rule_id'] in self.MANDATORY_RULES),
            'universal_kept': sum(1 for r in final_rules if r['rule_id'] in self.UNIVERSAL_RULES)
        }

        result = {
            'applicable_rules': final_rules,  # ← TOUTES les règles applicables
            'classified_rules': classified,    # ← Pour info
            'statistics': stats
        }

        print("\n" + "="*70)
        print("✅ PHASE 1 TERMINÉE - VERSION CORRECTE")
        print("="*70)
        print(f"📊 Règles finales: {stats['final_count']} / {stats['total_initial']}")
        print(f"📉 Réduction: {stats['reduction_rate']}%")
        print(f"\n🎯 Détail:")
        print(f"   🔴 Critique: {stats['by_priority']['critique']}")
        print(f"   🟠 Haute: {stats['by_priority']['haute']}")
        print(f"   🟢 Normale: {stats['by_priority']['normale']}")
        print(f"\n✅ TOUTES les {stats['final_count']} règles sont dans l'output")
        print(f"🔒 Obligatoires: {stats['mandatory_kept']}/{len(self.MANDATORY_RULES)}")
        print(f"🌍 Universelles: {stats['universal_kept']}/{len(self.UNIVERSAL_RULES)}")

        return result

    def export_results(self, result: Dict, output_path: str = "example_2/outputs/phase1_correct_output.json"):
        """Exporte résultats"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Exporté: {output_path}")


# ============================================================
# UTILISATION
# ============================================================

def main():
    """Exemple complet"""

    analyzer = Phase1_CORRECT(
        rules_json_path="inputs/vector_db_rules_merged_fixed.json",
        metadata_pkl_path="inputs/vectordb_rules.pkl",
        faiss_index_path="inputs/vectordb_rules.index",
        enriched_context_path="example_2/outputs/metadata_final_optimized_test.json",
        disclaimers_path="inputs/disclamer.json"
    )

    result = analyzer.run_phase1()
    analyzer.export_results(result)

    # Vérifications
    print("\n" + "="*70)
    print("🔍 VÉRIFICATION RÈGLES OBLIGATOIRES")
    print("="*70)

    final_ids = {r['rule_id'] for r in result['applicable_rules']}
    missing_mandatory = analyzer.MANDATORY_RULES - final_ids
    missing_universal = analyzer.UNIVERSAL_RULES - final_ids

    if missing_mandatory:
        print(f"❌ OBLIGATOIRES MANQUANTES: {missing_mandatory}")
    else:
        print(f"✅ Toutes obligatoires présentes ({len(analyzer.MANDATORY_RULES)})")

    if missing_universal:
        print(f"⚠️ UNIVERSELLES MANQUANTES: {missing_universal}")
    else:
        print(f"✅ Toutes universelles présentes ({len(analyzer.UNIVERSAL_RULES)})")

    # Afficher règles par priorité
    print("\n" + "="*70)
    print("📋 RÈGLES PAR PRIORITÉ")
    print("="*70)

    for priority in ['critique', 'haute', 'normale']:
        rules = result['classified_rules'][priority]
        print(f"\n{'🔴' if priority=='critique' else '🟠' if priority=='haute' else '🟢'} {priority.upper()} ({len(rules)}):")
        for rule in rules[:5]:  # Top 5
            print(f"   {rule['rule_id']}: {rule['rule_text'][:60]}...")

    return result


if __name__ == "__main__":
    result = main()