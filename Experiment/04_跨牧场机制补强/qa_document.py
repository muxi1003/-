"""Render the Word-exported PDF and check the manuscript's factual anchors."""
from pathlib import Path
import json
import argparse
import zipfile
import xml.etree.ElementTree as ET
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image,ImageDraw

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'20260921_v1')
parser.add_argument('--tables',type=int,default=7)
parser.add_argument('--figures',type=int,default=5)
args=parser.parse_args()
ROOT=args.root
pdf=ROOT/'奶牛热红外呼吸检测_中文初稿_20260921.pdf'
docx=ROOT/'奶牛热红外呼吸检测_中文初稿_20260921.docx'
out=ROOT/'document_qa/release_20260921';out.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(docx) as z:
    for n in z.namelist():
        if n.endswith('.xml') or n.endswith('.rels'):ET.fromstring(z.read(n))
    body=ET.fromstring(z.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    texts=''.join(x.text or '' for x in body.findall('.//w:t',ns))
    for required in ['0.890902','0.419629','0.492010','0.401696','−1.4491','−0.1734','Abstract','待补A1']:
        assert required in texts,required
    assert len(body.findall('.//w:tbl',ns))==args.tables
    assert len(body.findall('.//w:drawing',ns))==args.figures
reader=PdfReader(pdf);pages=[];counts=[]
render=pdfium.PdfDocument(str(pdf))
for i,page in enumerate(reader.pages):
    t=page.extract_text() or '';counts.append(len(t));assert len(t)>35,f'Blank page {i+1}'
    im=render[i].render(scale=1).to_pil().convert('RGB')
    im.save(out/f'page_{i+1:02d}.png');im.thumbnail((298,422))
    thumb=Image.new('RGB',(310,450),'#d9e0e3');thumb.paste(im,((310-im.width)//2,20));ImageDraw.Draw(thumb).text((8,3),str(i+1),fill='black');pages.append(thumb)
rows=(len(pages)+3)//4;sheet=Image.new('RGB',(4*310,rows*450),'white')
for i,im in enumerate(pages):sheet.paste(im,((i%4)*310,(i//4)*450))
sheet.save(out/'contact_sheet.png')
(out/'verification.json').write_text(json.dumps(dict(status='PASS_STRUCTURAL_AND_RENDER',pages=len(reader.pages),tables=args.tables,figures=args.figures,
    text_chars_by_page=counts,renderer='Microsoft Word COM to PDF; pypdfium2 raster',
    checks='XML well-formed, key numerical anchors, nonblank PDF pages; visual QA recorded separately'),indent=2),encoding='utf-8')
print('Document QA:',len(reader.pages),'pages,',args.tables,'tables,',args.figures,'figures')
