const fs = require('fs');
const path = require('path');
const libs = 'C:/Users/muxi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/';
const {Document,Packer,Paragraph,TextRun,Table,TableRow,TableCell,ImageRun,
  HeadingLevel,Footer,Header,PageNumber,AlignmentType,WidthType,BorderStyle,
  ShadingType,ExternalHyperlink} = require(libs+'docx');
const root = process.argv[2] ? path.resolve(process.argv[2]) : path.join(__dirname,'20260921_v1');
const input = path.join(root,'中文论文初稿.md');
const output = path.join(root,'奶牛热红外呼吸检测_中文初稿_20260921.docx');
const contentWidth=9306;
const strip = x=>x.replace(/\*\*(.*?)\*\*/g,'$1').replace(/`([^`]*)`/g,'$1');
function runs(line) {
  const out=[];let from=0;
  for(const m of line.matchAll(/https?:\/\/[^\s]+/g)) {
    out.push(new TextRun(strip(line.slice(from,m.index))));
    out.push(new ExternalHyperlink({link:m[0],children:[new TextRun({text:m[0],style:'Hyperlink',size:19})]}));
    from=m.index+m[0].length;
  }
  out.push(new TextRun(strip(line.slice(from))));return out;
}
function paragraph(text, opts={}) {return new Paragraph({children:runs(text),...opts});}
const lines=fs.readFileSync(input,'utf8').split(/\r?\n/);const children=[];
let figures=0,tables=0;
for(let i=0;i<lines.length;i++) {
  const line=lines[i].trim();if(!line)continue;
  const heading=line.match(/^(#{1,3}) (.*)$/);
  if(heading){const level=heading[1].length;children.push(paragraph(heading[2],{
    heading:level===1?HeadingLevel.TITLE:level===2?HeadingLevel.HEADING_1:HeadingLevel.HEADING_2,
    pageBreakBefore:level===2 && /^(1 |参考文献)/.test(heading[2]),
    keepNext:true}));continue;}
  if(line.startsWith('|')) {
    const rows=[];
    while(i<lines.length && lines[i].trim().startsWith('|')) {
      const cells=lines[i].trim().slice(1,-1).split('|').map(x=>strip(x.trim()));
      if(!cells.every(x=>/^:?-+:?$/.test(x)))rows.push(cells);
      i++;
    }
    i--;const n=rows[0].length;
    const widths=Array(n).fill(Math.floor(contentWidth/n));widths[n-1]+=contentWidth-widths.reduce((a,b)=>a+b,0);
    const border={style:BorderStyle.SINGLE,size:4,color:'BCC9CE'};
    children.push(new Table({width:{size:contentWidth,type:WidthType.DXA},columnWidths:widths,
      rows:rows.map((r,ri)=>new TableRow({tableHeader:ri===0,cantSplit:true,children:r.map((c,ci)=>new TableCell({
        width:{size:widths[ci],type:WidthType.DXA},borders:{top:border,bottom:border,left:border,right:border},
        margins:{top:90,bottom:90,left:110,right:110},
        shading:{type:ShadingType.CLEAR,fill:ri===0?'E7F0F2':'FFFFFF'},
        children:[new Paragraph({keepNext:ri<rows.length-1,spacing:{after:0,line:260},children:[new TextRun({text:c,size:20,bold:ri===0})]})]
      }))}))}));tables++;continue;
  }
  const image=line.match(/^!\[(.*?)\]\((.*?)\)$/);
  if(image){const file=path.resolve(root,image[2]);const data=fs.readFileSync(file);
    if(data.toString('ascii',1,4)!=='PNG')throw Error('Only generated PNG expected');
    const w=data.readUInt32BE(16),h=data.readUInt32BE(20),scale=Math.min(600/w,380/h);
    children.push(new Paragraph({alignment:AlignmentType.CENTER,keepNext:true,children:[new ImageRun({
      type:'png',data,transformation:{width:Math.round(w*scale),height:Math.round(h*scale)},
      altText:{name:image[1],title:image[1],description:'Derived from archived experimental data; see caption.'}})]}));figures++;continue;
  }
  const caption=/^图\d|^表\d/.test(line);
  children.push(paragraph(line,caption?{style:'Caption',keepNext:false}:{spacing:{after:130,line:340}}));
}
const doc=new Document({creator:'Research draft',title:'双鼻孔质量选择的奶牛热红外呼吸检测及跨牧场失效机制分析',
  description:'Evidence-bounded Chinese manuscript draft with English abstract, 2026-09-21',
  styles:{default:{document:{run:{font:{ascii:'Times New Roman',hAnsi:'Times New Roman',eastAsia:'宋体'},size:23},
    paragraph:{spacing:{after:130,line:340}}}},paragraphStyles:[
    {id:'Title',name:'Title',basedOn:'Normal',run:{font:{eastAsia:'黑体'},size:36,bold:true},paragraph:{spacing:{after:260},keepNext:true}},
    {id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:{eastAsia:'黑体'},size:30,bold:true},paragraph:{outlineLevel:0,spacing:{before:240,after:180},keepNext:true}},
    {id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:{eastAsia:'黑体'},size:25,bold:true},paragraph:{outlineLevel:1,spacing:{before:180,after:100},keepNext:true}},
    {id:'Caption',name:'Caption',basedOn:'Normal',run:{size:20,color:'374C55'},paragraph:{spacing:{after:180,line:270}}}
  ]},sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:1250,bottom:1200,left:1300,right:1300}}},
    headers:{default:new Header({children:[paragraph('奶牛热红外呼吸检测 | 研究初稿 · 2026-09-21',{alignment:AlignmentType.RIGHT})]})},
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({children:[PageNumber.CURRENT],size:18})]})]})},children}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync(output,b);fs.writeFileSync(path.join(root,'docx_build.json'),JSON.stringify({input,output,figures,tables,bytes:b.length},null,2));console.log(output,figures,tables)});
