import re
import json
import httpx
from openai import OpenAI

print("\n🚀 STARTING SLIDE ANALYSIS SCRIPT (NO RULES MODE)")

# =============================
# 1. INIT CLIENT
# =============================
print("🔹 Initializing LLM client...")
http_client = httpx.Client(verify=False)

client = OpenAI(
    api_key="sk-721b5920df174c10a8993002a07b452f",
    base_url="https://tokenfactory.esprit.tn/api",
    http_client=http_client
)

# =============================
# 2. LOAD SLIDES TXT
# =============================
print("🔹 Loading slides from TXT file...")

with open("example_2/slides_extracted_charts.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Split slides
slides = re.split(r"=== Slide (\d+) ===", content)

# This returns: ['', '1', 'content', '2', 'content', ...]
slide_pairs = [(slides[i], slides[i+1]) for i in range(1, len(slides), 2)]

print(f"✅ {len(slide_pairs)} slides found")

# =============================
# 3. PROCESS EACH SLIDE
# =============================
all_slide_analyses = []

for slide_number, slide_content in slide_pairs:
    print(f"\n📄 Processing Slide {slide_number}...")
    print("🔹 Sending to LLM...")

    prompt = f"""
Contexte IMPORTANT :
Cette slide fait partie d’un document PowerPoint appartenant à une présentation commerciale d’un fonds d’investissement.

Voici le contenu d'une slide :

[Slide {slide_number}]
{slide_content}

ANALYSE CETTE SLIDE ET RÉPONDS STRICTEMENT EN JSON AU FORMAT SUIVANT :

{{
  "summary": "",
  "themes": [],
  "visual_elements": [],
  "data_type": [],
  "disclaimers": []
}}

INTERPRÉTATION DES CHAMPS :

1. "summary" :
   - Court paragraphe (1 à 2 phrases max)
   - Doit expliquer globalement de quelle slide il s’agit et de quoi elle parle

2. "themes" :
   - 3 à 5 mots-clés sémantiques principaux

3. "visual_elements" :
   - Graphiques, tableaux, logos, titres, images, blocs de texte, etc.

4. "data_type" :
   - Exemple : description du fonds, données de performance, texte marketing, avertissements légaux, dates, informations réglementaires, etc.

5. "disclaimers" :
   - Liste des avertissements / mentions légales réellement présentes sur la slide (s’il y en a)

RÈGLES STRICTES :
- AUCUN texte en dehors du JSON
- PAS de markdown
- PAS d’explication hors structure
- JSON valide uniquement
"""

    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": "Tu es un assistant précis, technique et structuré."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1000
    )

    result = response.choices[0].message.content
    print("✅ LLM response received")

    try:
        slide_analysis = json.loads(result)

        all_slide_analyses.append({
            "slide_number": int(slide_number),
            "analysis": slide_analysis
        })

        print(f"✅ Slide {slide_number} successfully parsed & saved")

    except Exception as e:
        print(f"❌ ERROR parsing Slide {slide_number}")
        print("Raw response was:\n", result)
        print("Error:", e)

# =============================
# 4. SAVE FINAL OUTPUT
# =============================
output_path = "example_2/outputs/slides_analysis.json"
print(f"\n🔹 Saving final analysis to {output_path}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_slide_analyses, f, indent=2, ensure_ascii=False)

print("\n🎉 ALL DONE — Slides analysis successfully generated!")
