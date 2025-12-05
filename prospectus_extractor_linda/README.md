# 📄 README — VectorDB_Prospectus (Étape 3B)

## 🧠 Objectif du projet

Cette pipeline permet d’extraire automatiquement le contenu d’un **prospectus UCITS au format .docx**, de le découper en sections intelligentes, de classer chaque section selon un **type canonique UCITS**, et de construire une **VectorDB (FAISS)** permettant :

- 🔍 recherche sémantique précise,
- 🧪 recherche hybride (semantic + BM25),
- 🎯 filtrage par type de section,
- 📌 reranking par cross-encoder,
- 🧩 expansion de query,
- 📖 ajout de contexte adjacents.

Elle constitue la base d’un module RAG pour l’agent de vérification réglementaire.

---

## ⚙️ Architecture du pipeline

```text
prospectus.docx
       │
       ▼
[Parsing & Extraction]
       │
       ├── Paragraphs
       ├── Tables (flatten)
       ▼
Sections (title + text)

       │
       ▼
Chunking (max 450 words)
       │
       │  + Metadata:
       │      - section_type (heuristique)
       │      - fund_name
       │      - original_title
       │      - part_index
       ▼
Refinement LLM (optionnel)
TokenFactory → UCITS section_type canonique

       │
       ▼
Embedding (SBERT)
       │
       ├─ FAISS Index (encoded chunks)
       └─ JSON Metadata (chunk list, model info)

OUTPUTS:
  • outputs/parsed/prospectus_parsed.json
  • outputs/index/prospectus_faiss.index
  • outputs/index/prospectus_metadata.json
