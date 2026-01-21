import json
import pickle
import faiss
import numpy as np
from openai import OpenAI
import httpx
from tqdm import tqdm
import re
import time
import subprocess
import sys
import os

# ========================
# CONFIG
# ========================

os.makedirs("example_2/outputs", exist_ok=True)

PPTX_FILE = "example_2/expl2.pptx"  # Change this to your PPTX file path
TXT_FILE = "example_2/outputs/slides_extracted_charts.txt"
EXTRACTOR_SCRIPT = "pptx_extractor_charts.py"

API_KEY = "sk-721b5920df174c10a8993002a07b452f"
BASE_URL = "https://tokenfactory.esprit.tn/api"
MODEL = "hosted_vllm/Llama-3.1-70B-Instruct"

# Output files for each chunk type
OUTPUT_FILES = {
    "slide_full": {
        "pkl": "example_2/outputs/pptx_slide_full.pkl",
        "index": "example_2/outputs/pptx_slide_full.index",
        "json": "example_2/outputs/pptx_slide_full.json"
    },
    "slide_title": {
        "pkl": "example_2/outputs/pptx_slide_title.pkl",
        "index": "example_2/outputs/pptx_slide_title.index",
        "json": "example_2/outputs/pptx_slide_title.json"
    },
    "slide_element": {
        "pkl": "example_2/outputs/pptx_slide_element.pkl",
        "index": "example_2/outputs/pptx_slide_element.index",
        "json": "example_2/outputs/pptx_slide_element.json"
    },
    "footnote": {
        "pkl": "example_2/outputs/pptx_footnote.pkl",
        "index": "example_2/outputs/pptx_footnote.index",
        "json": "example_2/outputs/pptx_footnote.json"
    }
}

# ========================
# PPTX EXTRACTION
# ========================

def run_pptx_extractor(pptx_file):
    """Run the pptx_extractor_charts.py script to generate the text file"""
    
    # Check if PPTX file exists
    if not os.path.exists(pptx_file):
        raise FileNotFoundError(f"PPTX file not found: {pptx_file}")
    
    # Check if extractor script exists
    if not os.path.exists(EXTRACTOR_SCRIPT):
        raise FileNotFoundError(f"Extractor script not found: {EXTRACTOR_SCRIPT}")
    
    print(f"\n📊 Running PPTX extractor on: {pptx_file}")
    print(f"   Using script: {EXTRACTOR_SCRIPT}")
    
    try:
        # Run the extractor script with the PPTX file as argument
        result = subprocess.run(
            [sys.executable, EXTRACTOR_SCRIPT, pptx_file],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print extractor output
        if result.stdout:
            print("   Extractor output:")
            for line in result.stdout.strip().split('\n'):
                print(f"     {line}")
        
        print(f"✓ Extraction complete! Output saved to: {TXT_FILE}")
        
        # Verify the output file was created
        if not os.path.exists(TXT_FILE):
            raise FileNotFoundError(f"Expected output file not found: {TXT_FILE}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running extractor script:")
        print(f"   Return code: {e.returncode}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        raise

# ========================
# LLM CLIENT
# ========================

http_client = httpx.Client(verify=False, timeout=120.0)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)

# ========================
# VALIDATION
# ========================

def validate_chunks(chunks, slide_number):
    """Validate chunk structure matches requirements"""
    chunk_types = [c["chunk_type"] for c in chunks]
    
    # Must have exactly 1 slide_full and 1 slide_title
    if chunk_types.count("slide_full") != 1:
        raise ValueError(f"Slide {slide_number}: Must have exactly 1 slide_full chunk")
    if chunk_types.count("slide_title") != 1:
        raise ValueError(f"Slide {slide_number}: Must have exactly 1 slide_title chunk")
    
    # Check required fields
    required_fields = ["chunk_id", "chunk_type", "content", "metadata"]
    for idx, chunk in enumerate(chunks):
        for field in required_fields:
            if field not in chunk:
                raise ValueError(f"Slide {slide_number}, chunk {idx}: Missing field '{field}'")
    
    print(f"  ✓ Validated: {len(chunks)} chunks ({chunk_types.count('slide_element')} elements, {chunk_types.count('footnote')} footnotes)")

# ========================
# JSON EXTRACTION
# ========================

def extract_json_robustly(text, slide_number):
    """Extract JSON from LLM response with multiple fallback strategies"""
    
    # Save raw response for debugging
    debug_file = f"debug_slide_{slide_number}.txt"
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(text)
        
    
    # Strategy 8: Use json.JSONDecoder with strict=False
    try:
        import json
        decoder = json.JSONDecoder(strict=False)
        result = decoder.decode(text.strip())
        print(f"  ✓ Parsed with strategy 8 (non-strict decoder)")
        return result
    except:
        pass
    
    
    
    # Strategy 1: Direct parsing
    try:
        result = json.loads(text)
        print(f"  ✓ Parsed with strategy 1 (direct)")
        return result
    except Exception as e:
        pass
    
    # Strategy 2: Strip whitespace and try again
    try:
        result = json.loads(text.strip())
        print(f"  ✓ Parsed with strategy 2 (stripped)")
        return result
    except Exception as e:
        pass
    
    # Strategy 3: Extract from markdown code blocks
    code_block_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]
    
    for idx, pattern in enumerate(code_block_patterns):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1).strip())
                print(f"  ✓ Parsed with strategy 3.{idx+1} (markdown)")
                return result
            except:
                continue
    
    # Strategy 4: Find array boundaries
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            result = json.loads(json_str)
            print(f"  ✓ Parsed with strategy 4 (array extraction)")
            return result
    except Exception as e:
        pass
    
    # Strategy 5: Remove common LLM artifacts
    cleaned = text.replace('```json', '').replace('```', '').strip()
    try:
        result = json.loads(cleaned)
        print(f"  ✓ Parsed with strategy 5 (cleaned)")
        return result
    except:
        pass
    
    # Strategy 6: Fix common JSON errors (trailing commas)
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        result = json.loads(fixed)
        print(f"  ✓ Parsed with strategy 6 (fixed commas)")
        return result
    except:
        pass
    
    # Strategy 7: Try fixing quotes and escaping
    try:
        # Fix unescaped newlines in strings
        fixed = text.strip()
        result = json.loads(fixed)
        print(f"  ✓ Parsed with strategy 7 (quote fixing)")
        return result
    except:
        pass
    
    
    
    # Strategy 9: Read the saved debug file and try parsing it
    # (sometimes encoding issues occur during string handling)
    try:
        with open(debug_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        result = json.loads(file_content)
        print(f"  ✓ Parsed with strategy 9 (from file)")
        return result
    except:
        pass
    
    print(f"  ❌ All parsing strategies failed. Check {debug_file}")
    print(f"  First 200 chars of text: {text[:200]}")
    print(f"  Text length: {len(text)} characters")
    raise json.JSONDecodeError(f"Could not extract valid JSON for slide {slide_number}. Check {debug_file}", text, 0)

# ========================
# LLM CHUNKING
# ========================

def call_llm_for_chunking(slide_number, slide_text, total_slides, retry=3):
    """Appelle le LLM avec le texte brut de la slide"""
    
    system_prompt = f"""You are a JSON generator for PowerPoint slide chunking. Output ONLY valid JSON, nothing else.

TASK: Analyze slide {slide_number}/{total_slides} and create 4 types of chunks.

IDENTIFICATION RULES:
1. TITLE: Text in [top-center], [top-left], or [middle-center] with position marker - usually short (1-5 words)
2. BODY ELEMENTS: Text in [middle-*] positions, paragraphs, bullets, tables, charts
3. FOOTNOTES:
ONLY include:
- Legal disclaimers
- Risk warnings
- Sources
- Regulatory text
- Page numbers / dates

DO NOT classify as FOOTNOTE:
- Product explanations
- ESG descriptions
- Strategy descriptions
- Marketing or educational content
Even if located at the bottom of the slide.

FOOTNOTES MUST MATCH ONE OF THESE:
- Risk disclaimer
- Legal / regulatory text
- Source citation (starts with "Source:")
- Page number or date ONLY

If none match → NOT a FOOTNOTE.

CHUNK CREATION:
- SLIDE_FULL (1 chunk): Combine title + all body + all footnotes in structured format
- SLIDE_TITLE (1 chunk): Extract only the main title
- SLIDE_ELEMENT (multiple): ONE chunk per distinct body element (paragraph, bullet, table, chart)
- FOOTNOTE (multiple): ONE chunk per sentence/block in [bottom-*] positions

FOR TABLES/CHARTS:
- In SLIDE_FULL: "[TABLE: description of content]" or "[CHART: description]"
- As SLIDE_ELEMENT: Create separate chunk with description
- Set has_tables=true or has_charts=true in metadata

FORMAT (return ONLY this JSON, no markdown):
[
  {{"chunk_id": "document_slide_{slide_number}_full", "chunk_type": "slide_full", "content": "Slide {slide_number}: [TITLE]\\n\\n[Body content with tables/charts]\\n\\nNotes:\\n- [footnote 1]\\n- [footnote 2]", "metadata": {{"slide_number": {slide_number}, "slide_title": "[title text]", "has_charts": false, "has_tables": false, "has_images": false, "element_count": 2, "footnote_count": 2, "total_slides": {total_slides}}}}},
  {{"chunk_id": "document_slide_{slide_number}_title", "chunk_type": "slide_title", "content": "Slide {slide_number} - Titre: [TITLE]", "metadata": {{"slide_number": {slide_number}, "element_type": "title", "parent_chunk": "document_slide_{slide_number}_full"}}}},
  {{"chunk_id": "document_slide_{slide_number}_elem_1", "chunk_type": "slide_element", "content": "Slide {slide_number} - [first body element verbatim]", "metadata": {{"slide_number": {slide_number}, "element_index": 1, "element_type": "body_text", "font_size": 14, "is_bold": false, "parent_chunk": "document_slide_{slide_number}_full"}}}},
  {{"chunk_id": "document_slide_{slide_number}_elem_2", "chunk_type": "slide_element", "content": "Slide {slide_number} - [TABLE: description of table content]", "metadata": {{"slide_number": {slide_number}, "element_index": 2, "element_type": "body_text", "font_size": 12, "is_bold": false, "parent_chunk": "document_slide_{slide_number}_full"}}}},
  {{"chunk_id": "document_slide_{slide_number}_footnote_1", "chunk_type": "footnote", "content": "Slide {slide_number} - Note: [first bottom text verbatim]", "metadata": {{"slide_number": {slide_number}, "footnote_index": 1, "element_type": "footnote", "parent_chunk": "document_slide_{slide_number}_full"}}}},
  {{"chunk_id": "document_slide_{slide_number}_footnote_2", "chunk_type": "footnote", "content": "Slide {slide_number} - Note: [second bottom text verbatim]", "metadata": {{"slide_number": {slide_number}, "footnote_index": 2, "element_type": "footnote", "parent_chunk": "document_slide_{slide_number}_full"}}}}
]

CRITICAL RULES:
- First character MUST be [
- Last character MUST be ]
- NO ```json``` markdown
- NO explanations
- Keep ALL text verbatim from slide
- Split long [bottom-*] text into separate FOOTNOTE chunks by sentence or logical breaks
- Count elements: element_count = number of SLIDE_ELEMENT chunks, footnote_count = number of FOOTNOTE chunks
- If a text explains a concept (ESG, Liquidity, Strategy, Performance), it MUST be a SLIDE_ELEMENT, never a FOOTNOTE.
- A FOOTNOTE must NEVER explain a concept or product feature.
- Any text containing explanations, definitions, advantages, or strategy descriptions MUST be classified as SLIDE_ELEMENT, even if located at the bottom of the slide.
"""


    user_prompt = f"""Analyze this slide and return the JSON array:

SLIDE {slide_number} TEXT:
{slide_text}

Return ONLY the JSON array. No markdown, no explanation."""

    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=4000
            )

            raw_response = response.choices[0].message.content.strip()
            
            # Additional debug: check if response is empty or has unexpected content
            if not raw_response:
                raise ValueError(f"Empty response from LLM for slide {slide_number}")
            
            # Check if response starts with [ (should be JSON array)
            if not raw_response.lstrip().startswith('['):
                print(f"  ⚠️  Warning: Response doesn't start with '['. First 100 chars: {raw_response[:100]}")
            
            parsed_json = extract_json_robustly(raw_response, slide_number)
            
            if not isinstance(parsed_json, list) or len(parsed_json) == 0:
                raise ValueError(f"Invalid response structure for slide {slide_number}: got {type(parsed_json).__name__} with {len(parsed_json) if isinstance(parsed_json, list) else 0} items")
            
            # VALIDATE chunk structure
            validate_chunks(parsed_json, slide_number)
            
            return parsed_json
            
        except Exception as e:
            error_type = type(e).__name__
            print(f"\n⚠️  Attempt {attempt + 1}/{retry} failed for slide {slide_number}")
            print(f"    Error type: {error_type}")
            print(f"    Error message: {str(e)}")
            
            if attempt < retry - 1:
                time.sleep(2)
            else:
                error_file = f"failed_slide_{slide_number}.txt"
                with open(error_file, "w", encoding="utf-8") as f:
                    f.write(f"Error Type: {error_type}\n")
                    f.write(f"Error: {str(e)}\n\n")
                    f.write(f"Slide text:\n{slide_text[:500]}\n\n")
                    f.write(f"Raw response:\n{raw_response if 'raw_response' in locals() else 'No response received - API call failed'}")
                print(f"❌ Failed to process slide {slide_number}. Debug info saved to {error_file}")
                raise

# ========================
# PARSE TXT FILE
# ========================

def parse_slides_from_txt(txt_file):
    """Parse your slides_extracted_charts.txt file"""
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by slide markers
    slide_pattern = r'=== Slide (\d+) ===\n(.*?)(?=\n=== Slide \d+|$)'
    matches = re.findall(slide_pattern, content, re.DOTALL)
    
    slides = []
    for slide_num, slide_text in matches:
        slides.append({
            "slide_number": int(slide_num),
            "text": slide_text.strip()
        })
    
    total_slides = len(slides)
    for slide in slides:
        slide["total_slides"] = total_slides
    
    return slides

# ========================
# EMBEDDING (Placeholder)
# ========================

def text_to_embedding(text, dim=384):
    """Simple embedding - REPLACE with sentence-transformers in production"""
    vector = np.zeros(dim)
    for i, c in enumerate(text.encode("utf-8")[:dim]):
        vector[i] += c
    vector += np.random.random(dim) * 0.1
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

# ========================
# NEW: HELPER TO REMOVE EMBEDDINGS
# ========================

def remove_embeddings(chunks):
    """Create a copy of chunks without embeddings for JSON export"""
    clean_chunks = []
    for chunk in chunks:
        clean_chunk = {k: v for k, v in chunk.items() if k != "embedding"}
        clean_chunks.append(clean_chunk)
    return clean_chunks

# ========================
# MAIN PIPELINE
# ========================

def main():
    print("\n" + "="*60)
    print("🚀 STARTING PPTX CHUNKING PIPELINE")
    print("="*60)
    
    # Step 1: Run PPTX extractor
    print("\n" + "="*60)
    print("STEP 1: EXTRACT SLIDES FROM PPTX")
    print("="*60)
    
    try:
        run_pptx_extractor(PPTX_FILE)
    except Exception as e:
        print(f"\n❌ Failed to extract PPTX file: {str(e)}")
        print("\n💡 Make sure:")
        print(f"   1. The PPTX file exists: {PPTX_FILE}")
        print(f"   2. The extractor script exists: {EXTRACTOR_SCRIPT}")
        print(f"   3. Required dependencies are installed")
        return
    
    # Step 2: Parse extracted slides
    print("\n" + "="*60)
    print("STEP 2: PARSE EXTRACTED SLIDES")
    print("="*60)
    
    print(f"\n📄 Reading slides from: {TXT_FILE}")
    slides = parse_slides_from_txt(TXT_FILE)
    print(f"✓ Parsed {len(slides)} slides")
    
    # Show sample of first slide
    print(f"\n📝 Sample (Slide 1 preview):")
    print(slides[0]["text"][:200] + "..." if len(slides[0]["text"]) > 200 else slides[0]["text"])
    
    # Step 3: Process with LLM
    print("\n" + "="*60)
    print("STEP 3: CHUNK SLIDES WITH LLM")
    print("="*60)
    
    print("\n🤖 Processing slides with LLM...")
    print("💡 Each slide takes ~10-20 seconds")
    
    # Separate storage for each chunk type
    chunks_by_type = {
        "slide_full": [],
        "slide_title": [],
        "slide_element": [],
        "footnote": []
    }
    
    vectors_by_type = {
        "slide_full": [],
        "slide_title": [],
        "slide_element": [],
        "footnote": []
    }
    
    failed_slides = []

    for slide in tqdm(slides, desc="Processing slides"):
        try:
            chunks = call_llm_for_chunking(
                slide_number=slide["slide_number"],
                slide_text=slide["text"],
                total_slides=slide["total_slides"]
            )

            # Separate chunks by type
            for chunk in chunks:
                chunk_type = chunk["chunk_type"]
                content = chunk["content"]
                
                # Generate embedding
                emb = text_to_embedding(content)
                chunk["embedding"] = emb.tolist()
                
                # Store in appropriate type-specific list
                chunks_by_type[chunk_type].append(chunk)
                vectors_by_type[chunk_type].append(emb)
                
        except Exception as e:
            failed_slides.append(slide["slide_number"])
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"\n❌ Skipping slide {slide['slide_number']}: {error_msg}")
            continue

    # Check if we have any chunks
    total_chunks = sum(len(chunks) for chunks in chunks_by_type.values())
    if total_chunks == 0:
        print("\n❌ ERROR: No chunks were created. Check your API connection and model.")
        print("\n💡 Debug tips:")
        print("   1. Check if debug_slide_1.txt was created")
        print("   2. Verify API_KEY and BASE_URL are correct")
        print("   3. Test API manually with curl or Postman")
        return

    print(f"\n✓ Successfully processed {len(slides) - len(failed_slides)}/{len(slides)} slides")
    if failed_slides:
        print(f"⚠️  Failed slides: {failed_slides}")

    # Step 4: Build indices and save files
    print("\n" + "="*60)
    print("STEP 4: BUILD FAISS INDICES AND SAVE FILES")
    print("="*60)
    
    # Build separate FAISS index and save pickle + JSON for each chunk type
    print("\n💾 Building FAISS indices and saving chunks by type...")
    
    for chunk_type in chunks_by_type.keys():
        chunks = chunks_by_type[chunk_type]
        vectors = vectors_by_type[chunk_type]
        
        if len(chunks) == 0:
            print(f"   ⚠️  No chunks of type '{chunk_type}' - skipping")
            continue
        
        # Convert to numpy array
        vectors_array = np.array(vectors).astype("float32")
        
        # Build FAISS index
        index = faiss.IndexFlatL2(vectors_array.shape[1])
        index.add(vectors_array)
        
        # Save index
        index_file = OUTPUT_FILES[chunk_type]["index"]
        faiss.write_index(index, index_file)
        
        # Save chunks with embeddings (pickle)
        pkl_file = OUTPUT_FILES[chunk_type]["pkl"]
        with open(pkl_file, "wb") as f:
            pickle.dump(chunks, f)
        
        # NEW: Save chunks without embeddings (JSON)
        json_file = OUTPUT_FILES[chunk_type]["json"]
        clean_chunks = remove_embeddings(chunks)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(clean_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ {chunk_type}: {len(chunks)} chunks")
        print(f"     - Index: {index_file}")
        print(f"     - Pickle: {pkl_file}")
        print(f"     - JSON: {json_file}")

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"📊 Statistics:")
    print(f"   • Total slides processed: {len(slides) - len(failed_slides)}/{len(slides)}")
    print(f"   • Total chunks created: {total_chunks}")
    
    print(f"\n📋 Chunk breakdown by type:")
    for chunk_type, chunks in chunks_by_type.items():
        print(f"   • {chunk_type}: {len(chunks)} chunks")
    
    print(f"\n📁 Output files created:")
    for chunk_type in chunks_by_type.keys():
        if len(chunks_by_type[chunk_type]) > 0:
            print(f"   • {chunk_type}:")
            print(f"     - Index: {OUTPUT_FILES[chunk_type]['index']}")
            print(f"     - Pickle: {OUTPUT_FILES[chunk_type]['pkl']}")
            print(f"     - JSON: {OUTPUT_FILES[chunk_type]['json']}")
    
    # Save a sample output for inspection
    print("\n💾 Saving sample chunks for inspection...")
    sample_chunks = {}
    for chunk_type, chunks in chunks_by_type.items():
        sample_chunks[chunk_type] = [c for c in chunks if c["metadata"]["slide_number"] <= 2]
    
    # Remove embeddings from sample for readability
    sample_clean = {}
    for chunk_type, chunks in sample_chunks.items():
        sample_clean[chunk_type] = remove_embeddings(chunks)
    
    with open("sample_chunks_by_type.json", "w", encoding="utf-8") as f:
        json.dump(sample_clean, f, indent=2, ensure_ascii=False)
    print("   • Sample output: sample_chunks_by_type.json")
    
    print("="*60)

if __name__ == "__main__":
    # Allow PPTX file to be passed as command-line argument
    if len(sys.argv) > 1:
        PPTX_FILE = sys.argv[1]
        print(f"\n📋 Using PPTX file from command line: {PPTX_FILE}")
    
    main()