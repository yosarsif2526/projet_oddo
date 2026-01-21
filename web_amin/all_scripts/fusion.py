# Generated from: fusion_output_rapport_final_.ipynb
# Converted at: 2025-12-16T23:17:30.263Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import json
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime

class ComplianceReportMerger:
    """
    Classe pour fusionner dynamiquement deux rapports de compliance JSON
    en un seul rapport unifié avec sections violations et recommandations par slide.
    """

    def __init__(self):
        self.merged_report = {
            "metadata": {},
            "global_violations": [],  # NOUVEAU: Section pour violations globales
            "slides_analysis": [],
            "global_statistics": {},
            "global_rules_status": {}
        }

    def load_json_file(self, filepath: str) -> Dict:
        """Charge un fichier JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement de {filepath}: {e}")
            return {}

    def merge_metadata(self, report1: Dict, report2: Dict) -> Dict:
        """Fusionne les métadonnées des deux rapports"""
        metadata = {}

        # Prendre les métadonnées du rapport le plus complet
        if "metadata" in report2:
            metadata = report2["metadata"].copy()
        elif "document_name" in report1:
            metadata = {
                "document_name": report1.get("document_name", ""),
                "analysis_date": report1.get("analysis_date", datetime.now().isoformat()),
            }

        # Ajouter les sources
        metadata["sources"] = {
            "report1": "global_compliance_report",
            "report2": "phase2_3_violations_optimized"
        }

        return metadata

    def extract_violations_by_slide(self, report1: Dict) -> tuple[Dict[int, List[Dict]], List[Dict]]:
        """
        Extrait les violations du premier rapport (format global)
        et les organise par numéro de slide

        Returns:
            tuple: (violations_by_slide, global_violations_list)
        """
        violations_by_slide = defaultdict(list)
        global_violations_list = []  # NOUVEAU: Liste des violations globales

        # Violations principales
        if "violations" in report1:
            for violation in report1["violations"]:
                slides = violation.get("slides_involved", [])

                violation_obj = {
                    "rule_id": violation.get("rule_id", ""),
                    "violation_type": violation.get("violation_type", ""),
                    "severity": violation.get("severity", "medium"),
                    "explanation": violation.get("explanation", ""),
                    "violating_content": violation.get("violating_content", ""),
                    "correction": violation.get("correction", ""),
                    "confidence": violation.get("confidence", 0.8),
                    "source": "global_report"
                }

                if not slides:
                    # NOUVEAU: Violation globale (sans slide spécifique)
                    violation_obj["scope"] = "document"
                    global_violations_list.append(violation_obj)
                else:
                    # Violation spécifique à une ou plusieurs slides
                    for slide_num in slides:
                        violations_by_slide[slide_num].append(violation_obj.copy())

        # Recommandations (traitées comme violations de sévérité plus faible)
        if "recommendations" in report1:
            for recommendation in report1["recommendations"]:
                slides = recommendation.get("slides_involved", [])

                recommendation_obj = {
                    "rule_id": recommendation.get("rule_id", ""),
                    "violation_type": recommendation.get("violation_type", ""),
                    "severity": recommendation.get("severity", "low"),
                    "explanation": recommendation.get("explanation", ""),
                    "violating_content": recommendation.get("violating_content", ""),
                    "correction": recommendation.get("correction", ""),
                    "confidence": recommendation.get("confidence", 0.8),
                    "source": "global_report",
                    "is_recommendation": True
                }

                if not slides:
                    # Recommandation globale (rare mais possible)
                    recommendation_obj["scope"] = "document"
                    global_violations_list.append(recommendation_obj)
                else:
                    for slide_num in slides:
                        violations_by_slide[slide_num].append(recommendation_obj.copy())

        return violations_by_slide, global_violations_list

    def merge_slide_data(self, slide_data: Dict, global_violations: List[Dict]) -> Dict:
        """
        Fusionne les données d'une slide du rapport 2 avec les violations du rapport 1
        """
        slide_num = slide_data.get("slide_number", 0)

        # Récupérer les violations existantes du rapport 2
        existing_violations = slide_data.get("violations", [])
        existing_recommendations = slide_data.get("recommendations", [])

        # Ajouter les violations du rapport 1
        all_violations = existing_violations.copy()
        all_recommendations = existing_recommendations.copy()

        for violation in global_violations:
            if violation.get("is_recommendation", False):
                all_recommendations.append(violation)
            else:
                all_violations.append(violation)

        # Créer la structure fusionnée
        merged_slide = {
            "slide_number": slide_num,
            "slide_title": slide_data.get("slide_title", ""),
            "rules_checked": slide_data.get("rules_checked", 0),
            "violations_count": len(all_violations),
            "recommendations_count": len(all_recommendations),
            "violations": all_violations,
            "recommendations": all_recommendations
        }

        # Ajouter les champs optionnels s'ils existent
        if "range_rules_cleanup" in slide_data:
            merged_slide["range_rules_cleanup"] = slide_data["range_rules_cleanup"]

        return merged_slide

    def calculate_statistics(self, slides_analysis: List[Dict],
                            global_violations_list: List[Dict]) -> Dict:
        """Calcule les statistiques globales du rapport fusionné"""
        stats = {
            "total_slides": len(slides_analysis),
            "total_violations": 0,
            "total_recommendations": 0,
            "total_global_violations": len([v for v in global_violations_list if not v.get("is_recommendation", False)]),
            "total_global_recommendations": len([v for v in global_violations_list if v.get("is_recommendation", False)]),
            "slides_with_violations": 0,
            "slides_with_recommendations": 0,
            "slides_clean": 0,
            "violations_by_severity": defaultdict(int),
            "violations_by_type": defaultdict(int),
            "top_violated_rules": defaultdict(int)
        }

        # Analyser les violations globales d'abord
        for violation in global_violations_list:
            severity = violation.get("severity", "medium")
            v_type = violation.get("check_type", violation.get("violation_type", "unknown"))
            rule_id = violation.get("rule_id", "unknown")

            stats["violations_by_severity"][severity] += 1
            stats["violations_by_type"][v_type] += 1
            stats["top_violated_rules"][rule_id] += 1

        # Analyser les slides
        for slide in slides_analysis:
            violations_count = slide.get("violations_count", 0)
            recommendations_count = slide.get("recommendations_count", 0)

            stats["total_violations"] += violations_count
            stats["total_recommendations"] += recommendations_count

            if violations_count > 0:
                stats["slides_with_violations"] += 1
            if recommendations_count > 0:
                stats["slides_with_recommendations"] += 1
            if violations_count == 0 and recommendations_count == 0:
                stats["slides_clean"] += 1

            # Analyser les violations
            for violation in slide.get("violations", []):
                severity = violation.get("severity", "medium")
                v_type = violation.get("check_type", violation.get("violation_type", "unknown"))
                rule_id = violation.get("rule_id", "unknown")

                stats["violations_by_severity"][severity] += 1
                stats["violations_by_type"][v_type] += 1
                stats["top_violated_rules"][rule_id] += 1

            # Analyser les recommandations
            for rec in slide.get("recommendations", []):
                severity = rec.get("severity", "low")
                v_type = rec.get("check_type", rec.get("violation_type", "unknown"))
                rule_id = rec.get("rule_id", "unknown")

                stats["violations_by_severity"][severity] += 1
                stats["violations_by_type"][v_type] += 1
                stats["top_violated_rules"][rule_id] += 1

        # Ajouter les totaux globaux aux totaux généraux
        stats["total_violations"] += stats["total_global_violations"]
        stats["total_recommendations"] += stats["total_global_recommendations"]

        # Convertir les defaultdict en dict et trier les règles les plus violées
        stats["violations_by_severity"] = dict(stats["violations_by_severity"])
        stats["violations_by_type"] = dict(stats["violations_by_type"])

        top_rules = sorted(
            stats["top_violated_rules"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        stats["top_violated_rules"] = [
            {"rule_id": rule_id, "count": count}
            for rule_id, count in top_rules
        ]

        return stats

    def merge_reports(self, report1: Dict, report2: Dict) -> Dict:
        """
        Fonction principale pour fusionner les deux rapports

        Args:
            report1: Rapport au format global_compliance_report
            report2: Rapport au format phase2_3_violations_optimized

        Returns:
            Rapport fusionné avec toutes les violations et recommandations
        """
        print("🔄 Début de la fusion des rapports...")

        # 1. Fusionner les métadonnées
        print("📋 Fusion des métadonnées...")
        self.merged_report["metadata"] = self.merge_metadata(report1, report2)

        # 2. Extraire les violations du rapport 1 par slide ET les violations globales
        print("🔍 Extraction des violations du rapport global...")
        violations_by_slide, global_violations_list = self.extract_violations_by_slide(report1)

        # NOUVEAU: Stocker les violations globales
        self.merged_report["global_violations"] = global_violations_list
        if global_violations_list:
            print(f"   ⚠️  {len(global_violations_list)} violation(s) globale(s) détectée(s)")

        # 3. Fusionner les données par slide
        print("🔀 Fusion des données par slide...")
        slides_analysis = []

        if "slides_analysis" in report2:
            for slide_data in report2["slides_analysis"]:
                slide_num = slide_data.get("slide_number", 0)
                violations_for_slide = violations_by_slide.get(slide_num, [])

                merged_slide = self.merge_slide_data(slide_data, violations_for_slide)
                slides_analysis.append(merged_slide)

        self.merged_report["slides_analysis"] = slides_analysis

        # 4. Calculer les statistiques (incluant les violations globales)
        print("📊 Calcul des statistiques globales...")
        self.merged_report["global_statistics"] = self.calculate_statistics(
            slides_analysis,
            global_violations_list
        )

        # 5. Ajouter le statut des règles globales
        if "global_rules_status" in report2.get("global_statistics", {}):
            self.merged_report["global_rules_status"] = report2["global_statistics"]["global_rules_status"]

        print("✅ Fusion terminée avec succès!")
        return self.merged_report

    def save_merged_report(self, output_path: str):
        """Sauvegarde le rapport fusionné dans un fichier JSON"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.merged_report, f, ensure_ascii=False, indent=2)
            print(f"💾 Rapport fusionné sauvegardé: {output_path}")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")


# ============================================================================
# UTILISATION
# ============================================================================

if __name__ == "__main__":
    # Initialiser le merger
    merger = ComplianceReportMerger()

    # Charger les deux rapports
    print("📂 Chargement des rapports...")
    report1 = merger.load_json_file("example_2/outputs/global_compliance_report.json")
    report2 = merger.load_json_file("example_2/outputs/phase2_3_violations_optimized.json")

    # Fusionner les rapports
    merged_report = merger.merge_reports(report1, report2)

    # Sauvegarder le résultat
    merger.save_merged_report("example_2/outputs/merged_compliance_report.json")

    # Afficher un résumé
    print("\n" + "="*80)
    print("📈 RÉSUMÉ DU RAPPORT FUSIONNÉ")
    print("="*80)
    stats = merged_report.get("global_statistics", {})
    print(f"Total slides analysées: {stats.get('total_slides', 0)}")
    print(f"Total violations globales: {stats.get('total_global_violations', 0)}")
    print(f"Total violations par slide: {stats.get('total_violations', 0) - stats.get('total_global_violations', 0)}")
    print(f"Total violations: {stats.get('total_violations', 0)}")
    print(f"Total recommandations: {stats.get('total_recommendations', 0)}")
    print(f"Slides avec violations: {stats.get('slides_with_violations', 0)}")
    print(f"Slides propres: {stats.get('slides_clean', 0)}")

    # Afficher les violations globales
    global_viols = merged_report.get("global_violations", [])
    if global_viols:
        print(f"\n🚨 VIOLATIONS GLOBALES DU DOCUMENT:")
        for i, v in enumerate(global_viols, 1):
            print(f"   {i}. [{v['rule_id']}] {v['violation_type']} - {v['severity']}")
            print(f"      {v['explanation']}")

    print("\n💡 Top 5 règles les plus violées:")
    for i, rule in enumerate(stats.get('top_violated_rules', [])[:5], 1):
        print(f"  {i}. Règle {rule['rule_id']}: {rule['count']} violations")
    print("="*80)