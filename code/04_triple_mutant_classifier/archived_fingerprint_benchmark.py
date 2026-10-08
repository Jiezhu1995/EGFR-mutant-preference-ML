"""Load and audit the supplied triple-mutant fingerprint matrices."""

import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

FP_SPECS = {
    "CDK_Standard": ("Standard", "standard"),
    "ECFP6": ("ECFP6", "ECFP6"),
    "CDK_Extended": ("Extended", "extended"),
    "CDK_Graph": ("Graph", "graph"),
    "MACCS": ("MACCS", "MACCS"),
    "PubChem": ("Pubchem", "pubchem"),
    "Substructure": ("Substructure", "substructure"),
}


def scaffold(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return value or Chem.MolToSmiles(mol, canonical=True)


def file_path(root, fp, endpoint, split):
    folder, suffix = FP_SPECS[fp]
    ep = "mutation" if endpoint == "mutant" else "wildtype"
    base = root / folder
    if endpoint == "mutant" and folder != "Graph":
        base = base / "mutation"
    return base / f"final_{ep}_dataset_{split}set_{suffix}.csv"


def vcols(df):
    cols = [c for c in df.columns if re.fullmatch(r"V\d+", str(c))]
    return sorted(cols, key=lambda x: int(str(x)[1:]))


def load_archived(root, fp, endpoint):
    parts = []
    for split in ("training", "test"):
        path = file_path(root, fp, endpoint, split)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        features = vcols(df)
        if not features:
            raise ValueError(f"No archived V* fingerprint columns: {path}")
        part = pd.DataFrame({
            "smiles": df["Standard_Smiles"].astype(str),
            "historical_split": split,
            "historical_row": np.arange(len(df)),
        })
        if "label" in df:
            part["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
        else:
            part["label"] = np.nan
        parts.append((part, df[features].to_numpy(dtype=np.uint8), features, path))
    if parts[0][2] != parts[1][2]:
        raise ValueError(f"Feature columns differ between archived splits: {fp}/{endpoint}")
    meta = pd.concat([parts[0][0], parts[1][0]], ignore_index=True)
    X = np.vstack([parts[0][1], parts[1][1]])
    return meta, X, parts[0][2], [str(parts[0][3]), str(parts[1][3])]


def audit_endpoint(root, endpoint):
    loaded = {fp: load_archived(root, fp, endpoint) for fp in FP_SPECS}
    ref = loaded["CDK_Standard"][0]
    for fp, (meta, _, _, _) in loaded.items():
        if len(meta) != len(ref) or not meta[["smiles", "historical_split"]].equals(
            ref[["smiles", "historical_split"]]
        ):
            raise ValueError(f"Archived row/order mismatch: {endpoint}/{fp}")
        if meta.label.isna().all():
            meta["label"] = ref.label.to_numpy()
        elif not np.array_equal(meta.label.to_numpy(), ref.label.to_numpy()):
            raise ValueError(f"Archived label mismatch: {endpoint}/{fp}")
    scaffolds = ref.smiles.map(scaffold)
    valid = scaffolds.notna().to_numpy()
    if not valid.all():
        print(f"WARNING {endpoint}: excluding {(~valid).sum()} invalid SMILES consistently across all fingerprints")
    for fp in loaded:
        meta, X, features, paths = loaded[fp]
        loaded[fp] = (meta.loc[valid].reset_index(drop=True), X[valid], features, paths)
    scaffolds = scaffolds.loc[valid].reset_index(drop=True)
    return loaded, scaffolds
