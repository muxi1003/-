"""One bounded retry for a transient public-API failure; normalize author metadata."""
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

from collect_literature import words

ROOT = Path(__file__).resolve().parent / "literature"


def main():
    if (ROOT / "verified_bibliography.json").exists():
        raise FileExistsError("Preserve bibliography snapshot")
    records = json.loads((ROOT / "retrieval_manifest.json").read_text(encoding="utf-8"))
    supplement = []
    for record in records:
        if record["status"] != "retrieval_failed":
            continue
        event = {"id": record["id"], "original_failure_preserved": record["error"]}
        try:
            response = requests.get(record["retrieval_url"], timeout=(10, 35))
            response.raise_for_status()
            article = ET.fromstring(response.content)
            if article.tag != "article":
                raise ValueError("Not an article")
            (ROOT / f"{record['id']}.xml").write_bytes(response.content)
            meta = article.find("front/article-meta")
            text = "TITLE: " + words(meta.find("title-group/article-title")) + "\nABSTRACT: " + words(meta.find("abstract"))
            text += "\n\n" + "\n\n".join("SECTION: " + words(section.find("title")) + "\n" + words(section)
                                            for section in article.find("body").findall("sec"))
            target = ROOT / f"{record['id']}.txt"
            target.write_text(text, encoding="utf-8")
            event.update(status="full_text_open_jats", text_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         xml_sha256=hashlib.sha256(response.content).hexdigest())
            record["status"] = event["status"]
        except Exception as error:
            event.update(status="retrieval_failed", error=str(error))
        supplement.append(event)
    for record in records:
        path = ROOT / f"{record['id']}.xml"
        if path.is_file():
            article = ET.fromstring(path.read_bytes())
            meta = article.find("front/article-meta")
            record["title"] = words(meta.find("title-group/article-title"))
            record["doi"] = words(meta.find("article-id[@pub-id-type='doi']"))
            record["authors"] = [words(node.find("name/given-names")) + " " + words(node.find("name/surname"))
                                 for node in meta.findall("contrib-group/contrib")
                                 if node.get("contrib-type", "author") == "author" and node.find("name") is not None]
            record["metadata_note"] = "Author contributors only; editors excluded. Raw first retrieval manifest preserved."
        print(record["id"], record["status"], record.get("doi", ""), record.get("authors", []), flush=True)
    (ROOT / "retrieval_supplement.json").write_text(json.dumps(supplement, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "verified_bibliography.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
