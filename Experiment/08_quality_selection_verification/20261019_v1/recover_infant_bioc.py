"""Read the public NCBI BioC copy when the Europe PMC JATS transport fails."""
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parent / "literature"
URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC7510882/unicode"


def main():
    result = {"id": "infant2020", "public_source_url": URL, "prior_failures_preserved": True}
    try:
        response = requests.get(URL, timeout=(10, 25))
        response.raise_for_status()
        tree = ET.fromstring(response.content)
        passages = tree.findall(".//passage")
        if not passages:
            raise ValueError("No BioC passages")
        (ROOT / "infant2020_bioc.xml").write_bytes(response.content)
        lines = [" ".join(passage.find("text").itertext()) for passage in passages if passage.find("text") is not None]
        target = ROOT / "infant2020.txt"
        with target.open("x", encoding="utf-8") as stream:
            stream.write("\n\n".join(lines))
        result.update(status="full_text_public_bioc", characters=len("\n".join(lines)), passages=len(passages),
                      text_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    except Exception as error:
        result.update(status="retrieval_failed", error=str(error))
    with (ROOT / "infant_bioc_retrieval.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(result)


if __name__ == "__main__":
    main()
