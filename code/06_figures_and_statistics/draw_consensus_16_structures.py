#!/usr/bin/env python3
"""Draw the 16 regression/classification consensus molecules as a 4x4 plate."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

from pathlib import Path
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/figure1_screening/corrected_full_library/taosu_corrected_regression_ECFP6_consensus_16.csv"
OUT = ROOT / "figures/consensus_16_structures"
PDF_OUT = ROOT / "output/pdf"

NAMES = {
    "ZYSKXRAGBGLELB-UHFFFAOYSA-N": "TQB3804",
    "LIMFPAAAIVQRRD-BCGVJQADSA-N": "Tigozertinib",
    "VHQVOTINPRYDAO-UHFFFAOYSA-N": "(Rac)-JBJ-04-125-02",
    "HKSZLNNOFSGOKW-FYTWVXJKSA-N": "Staurosporine",
    "OZRNZXPMNJNRPE-UHFFFAOYSA-N": "No confirmed short name",
    "PBCZSGKMGDDXIJ-KRUBCLEUSA-N": "UCN-02 (project annotation)",
    "PBCZSGKMGDDXIJ-HQCWYSJUSA-N": "7-Hydroxystaurosporine / UCN epimer",
    "MOSKATHMXWSZTQ-UHFFFAOYSA-N": "No confirmed short name",
    "CLGWUCNXOBLWFM-UHFFFAOYSA-N": "No confirmed short name",
    "POTTVLREWUNNRO-UGZRAAABSA-N": "7-Oxo staurosporine",
    "BTMKEDDEMKKSEF-KRWDZBQOSA-N": "(S)-Sunvozertinib",
    "QQWUGDVOUVUTOY-UHFFFAOYSA-N": "No confirmed short name",
    "HRYNCLKDKCPYBF-UHFFFAOYSA-N": "T4015 / HG-14-10-04",
    "HKSZLNNOFSGOKW-UAXGJYLHSA-N": "Staurosporine stereoisomer",
    "YGSZHLWTCPKMLA-UHFFFAOYSA-N": "No confirmed short name",
    "WELBJLUKWAJOQV-UHFFFAOYSA-N": "Mps1-IN-2",
}


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size, index=0)
    return ImageFont.load_default()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    cell_w, cell_h = 590, 475
    margin, header = 36, 110
    canvas = Image.new("RGB", (cell_w*4 + margin*2, cell_h*4 + header + margin), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 28), "Regression–classification consensus molecules (n = 16)",
              font=font(36, True), fill=(25, 32, 38))
    draw.text((margin, 74), "Formula and public/common name are shown where a reliable match was found",
              font=font(21), fill=(83, 94, 103))
    metadata = []

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        mol = Chem.MolFromSmiles(row.smiles)
        rdDepictor.Compute2DCoords(mol)
        key = Chem.MolToInchiKey(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        name = NAMES.get(key, "No confirmed short name")
        x = margin + ((idx-1) % 4)*cell_w
        y = header + ((idx-1) // 4)*cell_h
        is_lead = key in {"HRYNCLKDKCPYBF-UHFFFAOYSA-N", "PBCZSGKMGDDXIJ-KRUBCLEUSA-N"}
        border = (35, 118, 128) if is_lead else (213, 220, 224)
        draw.rounded_rectangle((x+7, y+7, x+cell_w-14, y+cell_h-14), radius=12,
                               outline=border, width=4 if is_lead else 2, fill=(255,255,255))
        image = Draw.MolToImage(mol, size=(cell_w-52, 300), kekulize=True,
                                options=Draw.MolDrawOptions())
        canvas.paste(image, (x+26, y+20))
        short_id = str(row.hit_id)
        if short_id.startswith("NOHIT::"):
            short_id = short_id.replace("NOHIT::", "")
        draw.text((x+25, y+320), f"{idx}. {name}", font=font(22, True), fill=(25,32,38))
        draw.text((x+25, y+354), formula, font=font(20), fill=(42,78,84))
        wrapped = textwrap.wrap(short_id, width=43)[:2]
        for line_i, line in enumerate(wrapped):
            draw.text((x+25, y+386+line_i*24), line, font=font(17), fill=(92,101,108))
        metadata.append({"panel": idx, "identifier": row.hit_id, "public_or_common_name": name,
                         "molecular_formula": formula, "inchikey": key, "smiles": row.smiles,
                         "mut_ic50_nM": row.mut_ic50_nM, "wt_ic50_nM": row.wt_ic50_nM,
                         "direct_selectivity_fold": row.direct_selectivity_fold,
                         "derived_selectivity_fold": row.derived_selectivity_fold,
                         "historical_mutant_ECFP6_score": row.historical_mutant_ECFP6_score})

    png = OUT / "consensus_16_molecular_structures.png"
    pdf = PDF_OUT / "consensus_16_molecular_structures.pdf"
    canvas.save(png, dpi=(300,300), optimize=True)
    canvas.save(pdf, "PDF", resolution=300.0)
    pd.DataFrame(metadata).to_csv(OUT / "consensus_16_structure_metadata.csv", index=False)
    print(png); print(pdf)


if __name__ == "__main__":
    main()
