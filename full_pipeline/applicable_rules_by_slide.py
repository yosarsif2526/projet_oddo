import re
import json
import httpx
from openai import OpenAI

print("\n🚀 STARTING LLM RULE-MATCHING SCRIPT")

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
# 2. LOAD SLIDES ANALYSIS
# =============================
print("🔹 Loading slides analysis...")

with open("example_2/outputs/slides_analysis.json", "r", encoding="utf-8") as f:
    slides = json.load(f)

print(f"✅ {len(slides)} slides loaded")

# =============================
# 3. LOAD RULES
# =============================
print("🔹 Loading rules...")

with open("example_2/outputs/rules_cleaned.json", "r", encoding="utf-8") as f:
    rules = json.load(f)

print(f"✅ {len(rules)} rules loaded")


# =============================
# 4. PROCESS EACH SLIDE
# =============================
final_output = []

for slide in slides:

    slide_number = slide.get("slide_number")
    analysis = slide.get("analysis", {})

    print(f"\n📄 Processing Slide {slide_number}...")
    print("🔹 Sending to LLM...")

    prompt = f"""
Tu es un expert en conformité pour des présentations commerciales de fonds d’investissement.

IMPORTANT CONTEXTE :
Ces slides font partie d’un document PowerPoint utilisé dans une présentation commerciale de fonds.

Voici l’analyse de la slide {slide_number} :

{json.dumps(analysis, ensure_ascii=False)}

Voici la liste complète des règles de conformité potentielles :

{json.dumps(rules, ensure_ascii=False)}

OBJECTIF :
Pour cette slide uniquement :
- Sélectionne UNIQUEMENT les règles qui concernent ou peuvent concerner cette slide
- Base-toi sur le contenu, le type de données, les visuels, les mentions légales et le contexte
- Ne sélectionne PAS les règles totalement hors sujet

FORMAT DE RÉPONSE — STRICTEMENT en JSON :

{{
  "slide_number": {slide_number},
  "applicable_rules": [
    {{
      "rule_id": "",
      "section": "",
      "rule_text": "",
      "severity": ""
    }}
  ]
}}

RÈGLES IMPORTANTES :
- Pas de texte explicatif
- Pas de markdown
- Pas de phrases
- Seulement du JSON valide
- Ne retourne QUE les règles applicables à CETTE slide
    """

    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": "Tu es un assistant juridique très précis et strict."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1200
    )

    result = response.choices[0].message.content
    print("✅ LLM response received")

    try:
        slide_rules = json.loads(result)
        final_output.append(slide_rules)
        print(f"✅ Slide {slide_number} rules matched & saved")

    except Exception as e:
        print(f"❌ ERROR parsing Slide {slide_number}")
        print("Raw response was:\n", result)
        print("Error:", e)

# =============================
# 5. SAVE FINAL FILE
# =============================
output_path = "example_2/outputs/slides_with_applicable_rules.json"
print(f"\n🔹 Saving output to {output_path}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print("\n🎉 ALL DONE — Rules matching completed!")
