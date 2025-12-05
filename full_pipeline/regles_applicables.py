import json
import httpx
from openai import OpenAI

print("🔹 Début du script...")

# Désactive la vérification TLS/SSL si nécessaire
print("🔹 Initialisation du client OpenAI...")
http_client = httpx.Client(verify=False)

client = OpenAI(
    api_key="sk-721b5920df174c10a8993002a07b452f",  # <-- Mets ta clé ici
    base_url="https://tokenfactory.esprit.tn/api",
    http_client=http_client
)

# Charge les metadata et les règles
print("🔹 Chargement des fichiers metadata et rules...")
with open("example_2/outputs/metadata_enriched.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open("rules_parsed1.json", "r", encoding="utf-8") as f:
    rules = json.load(f)

print(f"✅ Metadata chargée : {len(metadata)} champs")
print(f"✅ Nombre total de règles : {len(rules)}")

# Construire le prompt pour le LLM
print("🔹 Construction du prompt pour le LLM...")
prompt = f"""
Tu es un assistant expert en conformité de documents financiers.

Voici les métadonnées du document (en JSON) :
{json.dumps(metadata, ensure_ascii=False)}

Voici la liste des règles (en JSON) :
{json.dumps(rules, ensure_ascii=False)}

Ta tâche : filtrer les règles pour ne conserver que celles **applicables à ce document**, en utilisant la metadata pour déterminer l'applicabilité. 

- Ne modifie pas les champs des règles.
- Retourne **uniquement un JSON contenant la liste filtrée des règles**, dans le même format que les règles originales.
- Ne mets aucune explication ou texte supplémentaire.
"""

# Appel au LLM
print("🔹 Envoi de la requête au LLM... Cela peut prendre quelques secondes.")
response = client.chat.completions.create(
    model="hosted_vllm/Llama-3.1-70B-Instruct",
    messages=[
        {"role": "system", "content": "Tu es un assistant utile et concis."},
        {"role": "user", "content": prompt}
    ],
    temperature=0,
)

print("✅ Réponse du LLM reçue.")

# Récupère la réponse
rules_cleaned_json = response.choices[0].message.content

# Sauvegarde dans un fichier
print("🔹 Sauvegarde des règles filtrées dans example_2/outputs/rules_cleaned.json...")
with open("example_2/outputs/rules_cleaned.json", "w", encoding="utf-8") as f:
    f.write(rules_cleaned_json)

print("🎉 Liste filtrée des règles enregistrée avec succès !")
