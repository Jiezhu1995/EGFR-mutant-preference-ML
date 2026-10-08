"""Redraw S7, preserving original identities and separating evidence from selection."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]
from pathlib import Path
import importlib.util
import sys
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[2]
from PIL import Image

def crop_structure(panel):
    image = Image.open(ROOT / 'figures/consensus_16_structures/consensus_16_molecular_structures.png').convert('RGB')
    idx = panel - 1
    x = 36 + (idx % 4) * 590 + 26
    y = 110 + (idx // 4) * 475 + 20
    return image.crop((x, y, x + 538, y + 300))
OUT = ROOT / 'figures/Supplementary_Figures/Supplementary_Figure7/Supplementary_Figure7_evidence_prioritization.pdf'
FORMAL = True
OUT.parent.mkdir(parents=True, exist_ok=True)
from matplotlib import font_manager
pdfmetrics.registerFont(TTFont('Arial', font_manager.findfont('Arial')))
pdfmetrics.registerFont(TTFont('ArialBold', font_manager.findfont(font_manager.FontProperties(family='Arial', weight='bold'))))
W,H=A4
c=canvas.Canvas(str(OUT), pagesize=A4)
c.setTitle('Supplementary Figure 7 | Evidence annotation and experimental prioritization')
INK='#20272B'; MUTED='#59646B'; LINE='#9AA4AA'; TEAL='#287C78'; BLUE='#46759B'

def txt(x,y,s,size=9,color=INK,bold=False):
    if FORMAL:
        size, color, bold = 9, '#000000', False
    c.setFillColor(HexColor(color)); c.setFont('ArialBold' if bold else 'Arial',size)
    c.drawString(x,H-y,s)

def box(x,y,w,h,fill='#F3F4F5',stroke=LINE):
    c.setFillColor(HexColor(fill)); c.setStrokeColor(HexColor(stroke)); c.setLineWidth(.7)
    c.roundRect(x,H-y-h,w,h,5,fill=1,stroke=1)

def para(x,y,s,width,size=8.5,color=MUTED,leading=11):
    line=''; lines=[]
    for word in s.split():
        test=(line+' '+word).strip()
        if pdfmetrics.stringWidth(test,'Arial',size)>width and line:
            lines.append(line); line=word
        else: line=test
    if line: lines.append(line)
    for i,line in enumerate(lines):txt(x,y+i*leading,line,size,color)
    return y+len(lines)*leading

def arrow(x,y1,y2):
    c.setStrokeColor(HexColor(LINE)); c.setLineWidth(.9)
    c.line(x,H-y1,x,H-y2)
    c.line(x,H-y2,x-3,H-y2+5); c.line(x,H-y2,x+3,H-y2+5)

txt(28,31,'Supplementary Figure 7',12,bold=True)
txt(28,49,'Evidence annotation and experimental prioritization of screening hits',10)
if not FORMAL: txt(28,65,'All 16 consensus hits are accounted for; prior activity does not imply a negative prediction.',8,MUTED)

box(183,80,229,36,'#EBF1F5',BLUE)
txt(235,102,'16 consensus hits',10)
# Audit separates overlaps without deleting them from the displayed hit set.
c.setStrokeColor(HexColor(LINE)); c.setLineWidth(.9)
c.line(297.5,H-116,297.5,H-126); c.line(153,H-126,426,H-126)
arrow(153,126,139);arrow(426,126,139)
box(28,140,249,110)
txt(40,159,'Training-set connectivity overlap (n = 4)',9)
for i,s in enumerate(['Tigozertinib; (Rac)-JBJ-04-125-02','Staurosporine; staurosporine stereoisomer']):txt(40,177+i*13,s,8.4)
if FORMAL: txt(40,218,'Not advanced for experimental evaluation',9,MUTED)
else: para(40,214,'Not advanced for experiments; not evidence of external predictive performance.',224)
box(292,140,275,46,'#EBF1F5',BLUE)
txt(304,159,'No exact training-set overlap (n = 12)',9)
txt(304,174,'Literature annotation and prioritization',8.5,MUTED)
arrow(430,186,200)
box(292,201,275,105,'#EBF1F5',BLUE)
txt(304,221,'Literature-supported positive hits (n = 4)',9)
txt(304,239,'TQB3804; brigatinib/OBX02-011;',9)
txt(304,252,'CZC-54252; NVP-TAE684',9)
if FORMAL:
    txt(304,273,'Prior biochemical and cellular evidence',9,MUTED)
    txt(304,292,'Not prioritized for additional testing',9,BLUE)
else:
    para(304,257,'Prior biochemical and cellular evidence in relevant C797S-containing or triple-mutant models.',250,8.3)
    txt(304,293,'Retained in hit summary; not prioritized for retesting.',8.2,BLUE)
# Second non-overlap branch to eight candidates; routed outside cards.
c.setStrokeColor(HexColor(LINE));c.line(430,H-186,578,H-186);c.line(578,H-186,578,H-323);c.line(578,H-323,298,H-323)
arrow(298,323,337)
box(28,338,539,55)
txt(40,359,'Eight candidates considered for further characterization',9)
if FORMAL: txt(40,377,'Characterization gaps, structural redundancy and experimental tractability',9,MUTED)
else: txt(40,377,'Selection considered characterization gaps, structural redundancy and experimental tractability.',8.4,MUTED)
arrow(298,393,410)
box(28,411,539,52)
txt(40,431,'Candidates removed: structural redundancy or commercial unavailability (n = 3)' if FORMAL else 'Three candidates not advanced',9)
txt(40,449,'UCN epimer; 7-oxo staurosporine; compound 15',8.5,MUTED)
# Selection continuation down right, avoiding the not-advanced card.
c.setStrokeColor(HexColor(LINE));c.line(298,H-399,578,H-399);c.line(578,H-399,578,H-475);c.line(578,H-475,298,H-475)
arrow(298,475,488)
box(28,489,539,145 if FORMAL else 167,'#EDF5F3',TEAL)
txt(40,509,'Five compounds selected for initial biochemical evaluation',10,color=TEAL)
panels=[6,8,11,13,16]
for i,panel in enumerate(panels):
    x=39+i*105
    box(x,521,99,99,'#FFFFFF','#B7C7C5')
    c.drawImage(ImageReader(crop_structure(panel)),x+2,H-521-78,95,74,preserveAspectRatio=True,anchor='c',mask='auto')
    names={6:['UCN-02'],8:['GSK1904529A'],11:['(S)-Sunvozertinib'],13:['HG-14-10-04','(T4015)'],16:['Mps1-IN-2']}
    for j,s in enumerate(names[panel]):
        font=9 if FORMAL else 7.7;tw=pdfmetrics.stringWidth(s,'Arial',font)
        txt(x+(99-tw)/2,604+j*10,s,font)
if FORMAL:
    arrow(298,634,654)
    box(96,655,403,53,'#EDF5F3',TEAL)
    txt(111,676,'HG-14-10-04 and UCN-02',10,color=TEAL)
    txt(111,694,'Detailed biochemical, cellular and structural characterization',9)
else:
    txt(40,641,'HG-14-10-04: previously reported triple-mutant biochemical activity (Chen et al., 2022).',8.1)
    arrow(298,656,672)
    box(96,673,403,53,'#EDF5F3',TEAL)
    txt(111,694,'HG-14-10-04 and UCN-02',10,color=TEAL)
    txt(111,712,'Detailed biochemical, cellular and structural characterization',8.7)
    para(28,750,'Known activity is supporting literature evidence, not a reason to label a screening hit as negative. Training-set overlap and experimental selection are reported separately.',539,8,leading=10)
    para(28,780,'Prior evidence refers to the EGFR variants and assay conditions reported in the cited studies, not necessarily the exact L858R/T790M/C797S setting. Compound identities and references are provided in Supplementary Table 5 and the text.',539,7.7,leading=10)
c.save()
print(OUT)
