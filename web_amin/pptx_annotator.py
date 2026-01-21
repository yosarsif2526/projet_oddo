# pptx_annotator.py - Ajoute des commentaires PowerPoint avec surlignage (Version Modifiée)
import os
import win32com.client
import pythoncom  # Nécessaire pour Flask/Multi-threading
from datetime import datetime
import json

class PPTXAnnotator:
    """
    Classe pour ajouter des commentaires de conformité dans une présentation PowerPoint
    Utilise COM pour ajouter de vrais commentaires PowerPoint
    """
    
    def __init__(self, pptx_path, results, output_path=None):
        """
        Args:
            pptx_path: Chemin vers la présentation originale
            results: Liste des résultats d'analyse (format JSON)
            output_path: Chemin de sortie (optionnel, génère automatiquement si None)
        """
        self.pptx_path = os.path.abspath(pptx_path)
        self.results = results
        self.output_path = output_path or self._generate_output_path()
        
        # Variables COM
        self.powerpoint = None
        self.presentation = None
        
    def _generate_output_path(self):
        """Génère un nom de fichier pour la présentation annotée"""
        base_dir = os.path.dirname(self.pptx_path)
        base_name = os.path.basename(self.pptx_path)
        name_without_ext = os.path.splitext(base_name)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"{name_without_ext}_ANNOTATED_{timestamp}.pptx")
    
    def _open_presentation(self):
        """Ouvre la présentation avec COM"""
        try:
            # Initialisation COM pour les threads Flask
            pythoncom.CoInitialize()
            
            print(f"\n📂 Ouverture de la présentation...")
            print(f"   Fichier: {self.pptx_path}")
            
            # Créer l'application PowerPoint
            self.powerpoint = win32com.client.Dispatch("PowerPoint.Application")
            
            # Tenter de rendre visible (parfois nécessaire pour que COM fonctionne bien)
            try:
                self.powerpoint.Visible = 1
            except:
                pass
            
            # Ouvrir la présentation
            self.presentation = self.powerpoint.Presentations.Open(self.pptx_path)
            
            print(f"   ✅ Présentation ouverte ({self.presentation.Slides.Count} slides)")
            return True
            
        except Exception as e:
            print(f"   ❌ Erreur lors de l'ouverture: {e}")
            return False
    
    def _close_presentation(self, save=True):
        """Ferme la présentation"""
        try:
            if self.presentation:
                if save:
                    print(f"\n💾 Sauvegarde vers: {self.output_path}")
                    self.presentation.SaveAs(os.path.abspath(self.output_path))
                    print(f"   ✅ Présentation sauvegardée")
                self.presentation.Close()
            
            if self.powerpoint:
                self.powerpoint.Quit()
                
        except Exception as e:
            print(f"   ⚠️ Erreur lors de la fermeture: {e}")
    
    def _find_text_in_slide(self, slide, search_text):
        """
        Trouve un texte dans une slide et retourne la shape et le range de texte
        Returns: (shape, text_range) ou (None, None) si non trouvé
        """
        if not search_text or len(search_text) < 3:
            return None, None
        
        search_lower = search_text.lower().strip()
        
        try:
            # Parcourir toutes les shapes de la slide
            for shape in slide.Shapes:
                # Vérifier si la shape a un TextFrame
                if shape.HasTextFrame:
                    text_frame = shape.TextFrame
                    if text_frame.HasText:
                        text_range = text_frame.TextRange
                        full_text = text_range.Text.lower()
                        
                        # Chercher le texte
                        if search_lower in full_text:
                            # Trouver la position exacte
                            start_pos = full_text.find(search_lower)
                            if start_pos >= 0:
                                # Créer un range pour le texte trouvé
                                # Note: Les positions COM sont 1-indexed
                                found_range = text_range.Characters(start_pos + 1, len(search_text))
                                return shape, found_range
        
        except Exception as e:
            print(f"      ⚠️ Erreur lors de la recherche de texte: {e}")
        
        return None, None
    
    def _highlight_text_range(self, text_range):
        """
        Applique un surlignage jaune au texte SANS changer la couleur du texte
        """
        try:
            # Méthode 1: Utiliser Fill pour surligner (comme un stabylo)
            text_range.Font.Fill.ForeColor.RGB = 0x00FFFF  # Jaune (BGR format)
            text_range.Font.Fill.Visible = -1  # True en COM
            text_range.Font.Fill.Solid()
            
            # MODIFICATION : On ne change PLUS la couleur du texte en rouge
            # text_range.Font.Color.RGB = 0x0000FF 
            
            return True
            
        except Exception as e:
            print(f"      ⚠️ Impossible de surligner (méthode 1): {e}")
            
            # Méthode 2: Utiliser BackColor si disponible
            try:
                text_range.Font.BackColor.RGB = 0x00FFFF  # Jaune
                return True
            except Exception as e2:
                print(f"      ⚠️ Impossible de surligner (méthode 2): {e2}")
                return False
    
    def _add_comment_to_slide(self, slide, violation, slide_num):
        """
        Ajoute un vrai commentaire PowerPoint à une slide
        """
        try:
            # Formater le texte du commentaire
            comment_text = self._format_violation_comment(violation)
            author = "Audit de Conformité"
            
            # Essayer de trouver et surligner le texte problématique
            evidence = violation.get('evidence', '')
            shape, text_range = self._find_text_in_slide(slide, evidence)
            
            # Position par défaut du commentaire
            left = 50
            top = 50
            
            if text_range:
                # Si on a trouvé le texte, placer le commentaire à sa position
                try:
                    left = shape.Left + 10
                    top = shape.Top + 10
                    
                    # Surligner le texte
                    if self._highlight_text_range(text_range):
                        print(f"      ✅ Texte surligné: '{evidence[:50]}...'")
                    else:
                        print(f"      ⚠️ Texte trouvé mais non surligné")
                        
                except Exception as e:
                    print(f"      ⚠️ Erreur de positionnement: {e}")
            
            # Ajouter le commentaire
            comment = slide.Comments.Add(
                Left=left,
                Top=top,
                Author=author,
                AuthorInitials="AC",
                Text=comment_text
            )
            
            print(f"      ✅ Commentaire ajouté: {violation.get('rule_id')}")
            return True
            
        except Exception as e:
            print(f"      ❌ Erreur lors de l'ajout du commentaire: {e}")
            return False
    
    def _format_violation_comment(self, violation):
        """Formate une violation en texte de commentaire"""
        rule_id = violation.get('rule_id', 'N/A')
        issue = violation.get('issue', 'Problème non spécifié')
        suggested_fix = violation.get('suggested_fix', 'Aucune solution proposée')
        evidence = violation.get('evidence', '')
        
        comment = f"🚨 {rule_id}\n\n"
        if evidence:
            comment += f"Texte concerné:\n\"{evidence[:100]}{'...' if len(evidence) > 100 else ''}\"\n\n"
        comment += f"Problème:\n{issue}\n\n"
        comment += f"💡 Solution:\n{suggested_fix}"
        
        return comment
    
    # MODIFICATION : La méthode _add_summary_slide est conservée mais ne sera pas appelée
    def _add_summary_slide(self):
        """(Désactivé) Ajoute une slide de résumé au début"""
        pass 
    
    def annotate(self):
        """
        Ajoute tous les commentaires de conformité à la présentation
        """
        print(f"\n🎨 Annotation de la présentation...")
        print(f"   Fichier source: {self.pptx_path}")
        print(f"   Fichier sortie: {self.output_path}")
        
        # Ouvrir la présentation
        if not self._open_presentation():
            return None
        
        try:
            violations_added = 0
            
            # Parcourir tous les résultats
            for result in self.results:
                slide_id = result.get('slide_id')
                violations = result.get('violations', [])
                
                if not violations:
                    continue
                
                # Les slides COM sont indexées à partir de 1
                if slide_id > self.presentation.Slides.Count:
                    print(f"   ⚠️ Slide {slide_id} introuvable dans la présentation")
                    continue
                
                slide = self.presentation.Slides(slide_id)
                
                print(f"\n   📌 Slide {slide_id} - {len(violations)} violation(s)")
                
                # Ajouter un commentaire pour chaque violation
                for violation in violations:
                    if self._add_comment_to_slide(slide, violation, slide_id):
                        violations_added += 1
            
            # MODIFICATION : Suppression de l'appel à la slide de résumé
            # self._add_summary_slide()
            
            print(f"\n✅ Annotation terminée!")
            print(f"   ⚠️ {violations_added} commentaire(s) ajouté(s)")
            
            # Fermer et sauvegarder
            self._close_presentation(save=True)
            
            return self.output_path
            
        except Exception as e:
            print(f"\n❌ Erreur lors de l'annotation: {e}")
            self._close_presentation(save=False)
            return None

def annotate_presentation(pptx_path, results, output_path=None):
    annotator = PPTXAnnotator(pptx_path, results, output_path)
    return annotator.annotate()