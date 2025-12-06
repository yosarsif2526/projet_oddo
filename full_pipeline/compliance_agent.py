import re
import json
import httpx
from openai import OpenAI

print("\n🚀 STARTING SLIDE COMPLIANCE CHECK SCRIPT (STRICT MODE)")

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
# 2. LOAD METADATA
# =============================
print("🔹 Loading document metadata...")

with open("example_2/outputs/metadata_enriched.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("✅ Metadata loaded")


# =============================
# 3. LOAD SLIDES RAW CONTENT
# =============================
print("🔹 Loading slides content...")

with open("example_2/slides_extracted_charts.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Split into slides
slides = re.split(r"=== Slide (\d+) ===", content)
slide_pairs = [(slides[i], slides[i + 1].strip()) for i in range(1, len(slides), 2)]

slides_content = {
    int(number): txt
    for number, txt in slide_pairs
}

print(f"✅ {len(slides_content)} slides loaded")


# =============================
# 4. LOAD RULES PER SLIDE
# =============================
print("🔹 Loading rules linked to each slide...")

with open("example_2/outputs/slides_with_applicable_rules.json", "r", encoding="utf-8") as f:
    slides_rules = json.load(f)

print(f"✅ {len(slides_rules)} slide rule-sets loaded")


# =============================
# 5. PROCESS EACH SLIDE
# =============================
final_results = []

for slide in slides_rules:

    slide_number = slide.get("slide_number")
    applicable_rules = slide.get("applicable_rules", [])

    # Skip if no rules
    if not applicable_rules:
        continue

    slide_text = slides_content.get(slide_number, "")

    print(f"\n📄 Checking Slide {slide_number}...")
    print("🔹 Sending to LLM in strict mode...")

    prompt = f"""
Tu es un expert senior compliance en conformité réglementaire pour les documents commerciaux de fonds d’investissement.

===================================================
CONTEXTE GLOBAL — DOCUMENT
===================================================

Cette slide fait partie d’une PRÉSENTATION POWERPOINT COMMERCIALE d’un fonds d’investissement.

Voici les métadonnées du document (fiables et à utiliser comme vérité) :

{json.dumps(metadata, ensure_ascii=False, indent=2)}




===================================================
SLIDE À ANALYSER
===================================================

NUMÉRO DE SLIDE : {slide_number}

CONTENU BRUT DE LA SLIDE :

{slide_text}


===================================================
RÈGLES À VÉRIFIER (déjà filtrées pour cette slide)
===================================================

{json.dumps(applicable_rules, ensure_ascii=False, indent=2)}


===================================================
MISSION — MODE STRICT (ANTIFAUX POSITIFS)
===================================================

Pour CHAQUE règle fournie :

✅ Tu NE DOIS inclure la règle dans la réponse QUE si :
   - tu trouves une NON-CONFORMITÉ FACTUELLE visible dans la slide
   - OU il existe un risque réel, logique et justifiable qu’elle ne soit pas respectée

❌ Tu N’INCLUES PAS la règle si :
   - elle est respectée
   - elle est hors contexte selon la metadata
   - elle concerne une autre slide
   - tu n’as aucune preuve visible dans la slide

👉 Tu DONNES OBLIGATOIREMENT une justification basée UNIQUEMENT
    sur le contenu réel de la slide et la metadata.

===================================================
FORMAT DE SORTIE — STRICT JSON UNIQUEMENT
===================================================

{{
  "slide_number": {slide_number},
  "non_respected_rules": [
    {{
      "rule_id": "",
      "section": "",
      "rule_text": "",
      "reason": ""
    }}
  ],
  "uncertain_rules": [
    {{
      "rule_id": "",
      "section": "",
      "rule_text": "",
      "reason": ""
    }}
  ]
}}

RÈGLES STRICTES :
- AUCUN texte en dehors du JSON
- PAS de markdown
- PAS d’explications supplémentaires
- Si tout est conforme → tableaux vides
- Raisons basées uniquement sur la slide {slide_number}
- Sois conservateur : en cas de doute faible → ne pas inclure
- Certaines règles ne s'appliquent que selon le type de client, le marché, le pays, le contexte ou le type de document.
- Evite les faux positifs , c'est tres important 
- Si une règle ne s'applique PAS → IGNORE-LA totalement.

    """

    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": "Tu es un auditeur de conformité ultra strict, conservateur et factuel."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1600
    )

    result = response.choices[0].message.content
    print("✅ LLM response received")

    try:
        parsed = json.loads(result)
        final_results.append(parsed)
        print(f"✅ Slide {slide_number} compliance check saved")

    except Exception as e:
        print(f"❌ ERROR parsing Slide {slide_number}")
        print("Raw response:\n", result)
        print("Error:", e)


# =============================
# 6. SAVE FINAL OUTPUT
# =============================
output_path = "example_2/outputs/slides_compliance_report.json"

print(f"\n🔹 Saving compliance report to {output_path}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print("\n🎉 ALL DONE — STRICT COMPLIANCE ANALYSIS FINISHED!")
