"""Snapshot six bounded primary sources; report retrieval failures explicitly."""
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import fitz
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent / "literature"
SOURCES = [
    {"id": "chen2025", "local_pdf": "E:/real/学习/呼吸文献/1-s2.0-S0306456525001111-main.pdf",
     "url": "https://doi.org/10.1016/j.jtherbio.2025.104154", "doi": "10.1016/j.jtherbio.2025.104154"},
    {"id": "zhao2023", "local_pdf": "E:/real/学习/呼吸文献/agriculture-13-01939-v2.pdf",
     "url": "https://www.mdpi.com/2077-0472/13/10/1939", "doi": "10.3390/agriculture13101939"},
    {"id": "negishi2020", "pmcid": "PMC7218727", "url": "https://www.mdpi.com/1424-8220/20/8/2171"},
    {"id": "takahashi2021", "pmcid": "PMC8271612", "url": "https://www.mdpi.com/1424-8220/21/13/4406"},
    {"id": "infant2020", "pmcid": "PMC7510882", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7510882/"},
    {"id": "analia2026", "pmcid": "PMC12788367", "url": "https://www.mdpi.com/1424-8220/26/1/278"},
]


def words(element):
    return " ".join(element.itertext()).strip() if element is not None else ""


def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "retrieval_manifest.json").exists():
        raise FileExistsError("Preserve published-source retrieval snapshot")
    records = []
    for source in SOURCES:
        record = dict(source)
        try:
            if "local_pdf" in source:
                document = fitz.open(source["local_pdf"])
                text = "\n".join(f"\nPAGE {i + 1}\n" + page.get_text() for i, page in enumerate(document))
                record.update(status="full_text_local_user_supplied", pages=len(document),
                              pdf_sha256=hashlib.sha256(Path(source["local_pdf"]).read_bytes()).hexdigest())
            else:
                url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{source['pmcid']}/fullTextXML"
                record["retrieval_url"] = url
                response = requests.get(url, timeout=(10, 35), headers={"User-Agent": "Cow-RR-method-verification/1.0"})
                response.raise_for_status()
                tree = ET.fromstring(response.content)
                if tree.tag != "article":
                    raise ValueError(f"Expected JATS article, got {tree.tag}")
                (ROOT / f"{source['id']}.xml").write_bytes(response.content)
                meta = tree.find("front/article-meta")
                record.update(status="full_text_open_jats", title=words(meta.find("title-group/article-title")),
                              journal=words(tree.find("front/journal-meta/journal-title-group/journal-title")),
                              authors=[words(author.find("name/given-names")) + " " + words(author.find("name/surname"))
                                       for author in meta.findall("contrib-group/contrib")
                                       if author.find("name") is not None and author.get("contrib-type", "author") == "author"],
                              doi=words(meta.find("article-id[@pub-id-type='doi']")),
                              license=words(meta.find("permissions/license")))
                sections = ["TITLE: " + record["title"], "ABSTRACT: " + words(meta.find("abstract"))]
                body = tree.find("body")
                for section in body.findall("sec"):
                    sections.append("SECTION: " + words(section.find("title")) + "\n" + words(section))
                text = "\n\n".join(sections)
                record["xml_sha256"] = hashlib.sha256(response.content).hexdigest()
            target = ROOT / f"{source['id']}.txt"
            target.write_text(text, encoding="utf-8")
            record["text_path"] = str(target)
            record["text_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            record["characters"] = len(text)
        except Exception as error:
            record.update(status="retrieval_failed", error=f"{type(error).__name__}: {error}")
        records.append(record)
        print(source["id"], record["status"], record.get("doi", ""), flush=True)
    (ROOT / "retrieval_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(ROOT / "retrieval_manifest.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
