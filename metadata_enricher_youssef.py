import os
import json
import httpx
from openai import OpenAI
from docx import Document

# ============================
# LLM Configuration
# ============================

def get_llm_client():
    """Initialize OpenAI client for TokenFactory"""
    http_client = httpx.Client(verify=False)
    client = OpenAI(
        api_key="sk-721b5920df174c10a8993002a07b452f",  # REPLACE WITH YOUR API KEY
        base_url="https://tokenfactory.esprit.tn/api",
        http_client=http_client
    )
    return client

def call_llm(prompt, system_message="Tu es un assistant d'extraction de métadonnées précis et concis."):
    """Call LLM and return response"""
    client = get_llm_client()
    
    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000,
        top_p=0.9,
    )
    
    return response.choices[0].message.content


# ============================
# 1. PARSE EXISTING METADATA.JSON
# ============================

def parse_existing_metadata(folder_path):
    """Load existing metadata.json"""
    metadata_path = os.path.join(folder_path, "metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}


# ============================
# 2. ANALYZE FOLDER STRUCTURE
# ============================

def detect_language_from_content(slides_content):
    """Use langdetect library to detect language from slides content"""
    try:
        from langdetect import detect
        text_sample = slides_content[:3000]
        detected = detect(text_sample).upper()
        return detected
    except:
        text_lower = slides_content[:3000].lower()
        if any(word in text_lower for word in ['le ', 'la ', 'les ', 'des ', 'une ', 'est ']):
            return "FR"
        elif any(word in text_lower for word in ['the ', 'and ', 'is ', 'are ', 'this ']):
            return "EN"
        elif any(word in text_lower for word in ['der ', 'die ', 'das ', 'und ', 'ist ']):
            return "DE"
        else:
            return "Unknown"


def analyze_folder(folder_path):
    """Analyze folder: count PPTX files, detect languages from ALL extracted txt files"""
    pptx_files = [f for f in os.listdir(folder_path) if f.endswith('.pptx') and not f.startswith('~$')]
    nombre_pptx = len(pptx_files)
    
    languages = []
    txt_files = [f for f in os.listdir(folder_path) if f.startswith('slides_extracted_') and f.endswith('.txt')]
    
    for txt_file in txt_files:
        txt_path = os.path.join(folder_path, txt_file)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
                detected_lang = detect_language_from_content(content)
                languages.append(detected_lang)
                print(f"   ✓ Detected language for {txt_file}: {detected_lang}")
        except Exception as e:
            print(f"   ⚠️ Error reading {txt_file}: {e}")
    
    return {
        "nombre_documents": nombre_pptx,
        "documents_multilingues": nombre_pptx > 1,
        "langues": list(set(languages)) if languages else ["Unknown"]
    }


# ============================
# 3. EXTRACT FROM PPTX
# ============================

def extract_from_pptx_txt(folder_path):
    """Extract metadata from slides_extracted.txt using LLM - ONLY PROCESS ONE FILE"""
    txt_files = [f for f in os.listdir(folder_path) if f.startswith('slides_extracted_') and f.endswith('.txt')]
    
    if not txt_files:
        print(f"⚠️ No slides_extracted_*.txt files found in {folder_path}")
        return {}
    
    txt_path = os.path.join(folder_path, txt_files[0])
    print(f"   📄 Using {txt_files[0]} for metadata extraction")
    
    with open(txt_path, "r", encoding="utf-8") as f:
        slides_content = f.read()
    
    nombre_pages = slides_content.count("=== Slide")
    
    prompt = f"""Analyse ce document PowerPoint extrait et retourne UNIQUEMENT un JSON valide avec ces informations :

{{
  "type_presentation": "standard" ou "court" (standard si ~30 pages, court si 5-6 pages),
  "client_specifique": true/false (cherche dans les disclaimers, slide 2 généralement),
  "client_belge": true/false (si client spécifique mentionné, est-il belge?),
  "mention_nombre_lignes": true/false (le document mentionne-t-il un nombre de lignes dans le portefeuille?),
  "nombre_lignes_portefeuille": nombre ou null (extraire le nombre si mentionné),
  "presentation_equipe_gestion": true/false (y a-t-il une présentation de l'équipe de gestion?),
  "allemagne_exclusivement": true/false (document exclusivement pour l'Allemagne?)
}}

Document (nombre de slides: {nombre_pages}):
{slides_content}

Retourne UNIQUEMENT le JSON, sans texte additionnel."""

    llm_response = call_llm(prompt)
    
    try:
        json_start = llm_response.find('{')
        json_end = llm_response.rfind('}') + 1
        json_str = llm_response[json_start:json_end]
        
        extracted_data = json.loads(json_str)
        extracted_data["nombre_pages"] = nombre_pages
        
        return extracted_data
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parsing LLM response: {e}")
        print(f"LLM Response: {llm_response}")
        return {"nombre_pages": nombre_pages}


# ============================
# 4. EXTRACT FROM PROSPECTUS.DOCX
# ============================

def extract_from_prospectus(folder_path):
    """Extract ESG approach from prospectus using LLM"""
    prospectus_path = os.path.join(folder_path, "prospectus.docx")
    
    if not os.path.exists(prospectus_path):
        print(f"⚠️ prospectus.docx not found in {folder_path}")
        return {}
    
    doc = Document(prospectus_path)
    text_content = "\n".join([para.text for para in doc.paragraphs])
    
    prompt = f"""Analyse ce prospectus et détermine le type d'approche ESG , il faut comprendre le prospectus pour connaitre l'approche , ca va pas etre donné directement. Retourne UNIQUEMENT un JSON valide:

{{
  "approche_esg": "engageante" ou "réduite" ou "limitée au prospectus" ou "non spécifiée"
}}

Critères:
- **Approche engageante**: ≥ 20% d'exclusion ET ≥ 90% du portefeuille couvert
- **Approche réduite**: Communication limitée à moins de 10% du volume
- **Approche limitée au prospectus**: Pas de mention ESG sauf pour investisseurs institutionnels professionnels

Prospectus (extrait):
{text_content}

Retourne UNIQUEMENT le JSON, sans texte additionnel."""

    llm_response = call_llm(prompt)
    
    try:
        json_start = llm_response.find('{')
        json_end = llm_response.rfind('}') + 1
        json_str = llm_response[json_start:json_end]
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parsing LLM response: {e}")
        print(f"LLM Response: {llm_response}")
        return {"approche_esg": "non spécifiée"}


# ============================
# 5. CHECK PRE-COMMERCIALISATION
# ============================

def check_pre_commercialisation(existing_metadata, folder_path):
    """Check if document is in pre-commercialization based on dates and new product/strategy"""
    pptx_files = [f for f in os.listdir(folder_path) if f.endswith('.pptx') and not f.startswith('~$')]
    
    if not pptx_files:
        return False
    
    pptx_path = os.path.join(folder_path, pptx_files[0])
    pptx_date = os.path.getmtime(pptx_path)
    
    is_new_strategy = existing_metadata.get("Le document fait-il référence à une nouvelle Stratégie", False)
    is_new_product = existing_metadata.get("Le document fait-il référence à un nouveau Produit", False)
    
    return is_new_strategy or is_new_product


# ============================
# 6. COMBINE ALL METADATA
# ============================

def extract_all_metadata(folder_path):
    """Main function: extract all metadata and save to new metadata.json"""
    
    print(f"\n📂 Processing folder: {folder_path}")
    print("="*60)
    
    print("\n1️⃣ Loading existing metadata.json...")
    existing_metadata = parse_existing_metadata(folder_path)
    print(f"   ✓ Loaded: {list(existing_metadata.keys())}")
    
    print("\n2️⃣ Analyzing folder structure...")
    folder_metadata = analyze_folder(folder_path)
    print(f"   ✓ Documents: {folder_metadata['nombre_documents']}")
    print(f"   ✓ Languages: {folder_metadata['langues']}")
    
    print("\n3️⃣ Extracting from slides_extracted_*.txt (via LLM)...")
    pptx_metadata = extract_from_pptx_txt(folder_path)
    print(f"   ✓ Extracted: {list(pptx_metadata.keys())}")
    
    print("\n4️⃣ Extracting from prospectus.docx (via LLM)...")
    prospectus_metadata = extract_from_prospectus(folder_path)
    print(f"   ✓ ESG Approach: {prospectus_metadata.get('approche_esg', 'N/A')}")
    
    print("\n5️⃣ Checking pre-commercialization status...")
    is_pre_commercialisation = check_pre_commercialisation(existing_metadata, folder_path)
    print(f"   ✓ Pre-commercialization: {is_pre_commercialisation}")
    
    print("\n6️⃣ Combining all metadata...")
    enriched_metadata = {
        **existing_metadata,
        **folder_metadata,
        **pptx_metadata,
        **prospectus_metadata,
        "pre_commercialisation": is_pre_commercialisation
    }
    
    output_path = os.path.join(folder_path, "metadata_enriched.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Enriched metadata saved to: {output_path}")
    print("="*60)
    
    return enriched_metadata


# ============================
# MAIN EXECUTION
# ============================

if __name__ == "__main__":
    folder = "example_3"
    
    metadata = extract_all_metadata(folder)
    
    print("\n📊 FINAL METADATA:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))