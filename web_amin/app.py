from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import json
import shutil
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid

# Import de votre module d'analyse
from all_scripts.run_all import run_pipeline
from pptx_annotator import annotate_presentation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['ALLOWED_EXTENSIONS'] = {'pptx', 'pdf', 'docx'}
app.secret_key = 'sk-5312125f52c6491488218c98a934862a'

# Créer les dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

def allowed_file(filename, extensions=None):
    if extensions is None:
        extensions = app.config['ALLOWED_EXTENSIONS']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # Vérifier que tous les fichiers requis sont présents
        if 'presentation' not in request.files:
            return jsonify({'error': 'Présentation (.pptx) manquante'}), 400
        if 'prospectus' not in request.files:
            return jsonify({'error': 'Prospectus (.pdf ou .docx) manquant'}), 400
        
        presentation = request.files['presentation']
        prospectus = request.files['prospectus']
        
        # Vérifier que les fichiers ont des noms
        if presentation.filename == '' or prospectus.filename == '':
            return jsonify({'error': 'Fichiers non sélectionnés'}), 400
        
        # Vérifier les extensions
        if not allowed_file(presentation.filename, {'pptx'}):
            return jsonify({'error': 'La présentation doit être au format .pptx'}), 400
        
        if not allowed_file(prospectus.filename, {'pdf', 'docx'}):
            return jsonify({'error': 'Le prospectus doit être au format .pdf ou .docx'}), 400
        
        # Récupérer les métadonnées du formulaire
        metadata = {
            "Société de Gestion": request.form.get('societe_gestion', ''),
            "Est ce que le produit fait partie de la Sicav d'Oddo": request.form.get('sicav_oddo') == 'true',
            "Le client est-il un professionnel": request.form.get('client_professionnel') == 'true',
            "Le document fait-il référence à une nouvelle Stratégie": request.form.get('nouvelle_strategie') == 'true',
            "Le document fait-il référence à un nouveau Produit": request.form.get('nouveau_produit') == 'true',
            
        }
        
        # Créer un dossier unique pour cette analyse
        analysis_id = str(uuid.uuid4())
        analysis_folder = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
        os.makedirs(analysis_folder, exist_ok=True)
        
        # Sauvegarder les fichiers
        presentation_path = os.path.join(analysis_folder, 'presentation.pptx')
        presentation.save(presentation_path)
        
        # Sauvegarder le prospectus avec le bon nom
        prospectus_ext = prospectus.filename.rsplit('.', 1)[1].lower()
        prospectus_path = os.path.join(analysis_folder, f'prospectus.{prospectus_ext}')
        prospectus.save(prospectus_path)
        
        # Sauvegarder les métadonnées
        metadata_path = os.path.join(analysis_folder, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Lancer l'analyse
        print(f"🚀 Démarrage de l'analyse pour {analysis_id}")
        results = run_pipeline(presentation_path, prospectus_path, metadata)

        global_violations = []
        try:
            merged_report_path = os.path.join(os.path.dirname(__file__), 'all_scripts', 'example_2', 'outputs', 'merged_compliance_report.json')
            if os.path.exists(merged_report_path):
                with open(merged_report_path, 'r', encoding='utf-8') as f:
                    merged_report = json.load(f)
                global_violations = merged_report.get('global_violations', []) or []
        except Exception as e_global:
            print(f"❌ Erreur lors du chargement des violations globales : {e_global}")

        # Générer le PPTX annoté (même comportement que l'ancien compliance_engine)
        annotated_path = os.path.join(analysis_folder, 'presentation_annotated.pptx')
        try:
            annotate_presentation(presentation_path, results, output_path=annotated_path)
        except Exception as e_annot:
            print(f"❌ Erreur lors de l'annotation PowerPoint : {e_annot}")
        
        # Vérifier si le fichier annoté a été créé
        annotated_file_exists = os.path.exists(os.path.join(analysis_folder, 'presentation_annotated.pptx'))
        
        # Sauvegarder les résultats
        results_path = os.path.join('results', f'{analysis_id}.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_id': analysis_id,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata,
                'has_annotated_pptx': annotated_file_exists, # Nouveau flag
                'global_violations': global_violations,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'redirect': url_for('view_results', analysis_id=analysis_id)
        })
        
    except Exception as e:
        print(f"❌ Erreur lors de l'upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erreur lors de l\'analyse: {str(e)}'}), 500

@app.route('/results/<analysis_id>')
def view_results(analysis_id):
    try:
        results_path = os.path.join('results', f'{analysis_id}.json')
        if not os.path.exists(results_path):
            return "Résultats non trouvés", 404
        
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return render_template('results.html', data=data)
    except Exception as e:
        return f"Erreur lors du chargement des résultats: {str(e)}", 500
    
    
    

# --- NOUVELLE ROUTE POUR TÉLÉCHARGER LE PPTX ANNOTÉ ---
@app.route('/download-pptx/<analysis_id>')
def download_annotated_pptx(analysis_id):
    try:
        # Chemin vers le dossier d'upload d'origine
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
        file_path = os.path.join(folder_path, 'presentation_annotated.pptx')
        
        if not os.path.exists(file_path):
            return "Fichier annoté non trouvé (peut-être une erreur lors de la génération)", 404
            
        return send_file(file_path, 
                        as_attachment=True,
                        download_name=f'Resultats_Annotés_{analysis_id}.pptx',
                        mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation')
    except Exception as e:
        return f"Erreur: {str(e)}", 500



@app.route('/download/<analysis_id>')
def download_results(analysis_id):
    try:
        results_path = os.path.join('results', f'{analysis_id}.json')
        if not os.path.exists(results_path):
            return "Résultats non trouvés", 404
        
        return send_file(results_path, 
                        as_attachment=True,
                        download_name=f'compliance_analysis_{analysis_id}.json',
                        mimetype='application/json')
    except Exception as e:
        return f"Erreur lors du téléchargement: {str(e)}", 500

@app.route('/api/status/<analysis_id>')
def get_status(analysis_id):
    """API endpoint pour vérifier le statut d'une analyse"""
    results_path = os.path.join('results', f'{analysis_id}.json')
    if os.path.exists(results_path):
        return jsonify({'status': 'completed'})
    else:
        return jsonify({'status': 'processing'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)