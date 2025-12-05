import json
from datetime import datetime
import docx


class ProspectusExtractor:
    def extract(self, prospectus_path: str):
        text = self._read_prospectus(prospectus_path)
        return self._simple_extract(text)

    def _read_prospectus(self, path: str) -> str:
        try:
            doc = docx.Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            joined = "\n".join(paragraphs[:20])
            return joined[:2000]
        except Exception:
            return ""

    def _simple_extract(self, text: str):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        fund_name = lines[0] if lines else None

        currency = None
        for c in ["EUR", "USD", "GBP"]:
            if c in text:
                currency = c
                break

        return {
            "currency": currency,
        }


class ComplianceEngine:
    def build_context(self, metadata, prospectus_data):
        basic = self._get_basic(metadata)
        fund_summary = self._get_fund_summary(prospectus_data)
        disclaimer_ids = self._get_disclaimer_ids(basic, fund_summary)

        return {
            "basic_metadata": basic,
            "disclaimer_ids": disclaimer_ids,
            "generated_at": datetime.now().isoformat(),
        }

    def _get_basic(self, data):
        doc_info = data.get("document_info", {})
        language = doc_info.get("language", "FR")
        audience = doc_info.get("target_audience", "retail")
        if audience == "professional":
            client_type = "professionnel"
        else:
            client_type = "retail"
        return {
            "language": language,
            "client_type": client_type,
        }

    def _get_fund_summary(self, prospectus_data):
        return {
            "currency": prospectus_data.get("currency"),
        }

    def _get_disclaimer_ids(self, basic, fund_summary):
        client_type = basic["client_type"]
        ids = []

        ids.append({
            "id": "OBAM_Presentation",
            "client_type": client_type,
            "location": "slide_2",
        })

        ids.append({
            "id": "Performance",
            "client_type": client_type,
            "location": "near_performance",
        })

        ids.append({
            "id": "SRI_marketing",
            "client_type": client_type,
            "location": "same_slide_as_sri",
        })

        return ids


def run_compliance_check(
    metadata_path: str,
    prospectus_path: str,
    disclaimers_path: str,
    api_key: str,
    output_path: str = "enriched_context.json",
):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    extractor = ProspectusExtractor()
    prospectus_data = extractor.extract(prospectus_path)

    engine = ComplianceEngine()
    context = engine.build_context(metadata, prospectus_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=False)

    return context


if _name_ == "_main_":
    CONFIG = {
        "metadata_path": "/content/metadata.json",
        "prospectus_path": "/content/prospectus.docx",
        "disclaimers_path": "/content/disclamer.json",
        "api_key": "PLACEHOLDER",
        "output_path": "/content/enriched_context.json",
    }

    result = run_compliance_check(**CONFIG)