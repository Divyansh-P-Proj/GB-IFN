import torch
import numpy as np
from collections import Counter
from itertools import product
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import esm
import xgboost as xgb

# ---- Load ESM model ----
print("Loading ESM model...")
esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
device = "cuda" if torch.cuda.is_available() else "cpu"
esm_model.to(device)
esm_model.eval()

AA = list("ACDEFGHIKLMNPQRSTVWY")

# ---- Feature functions ----
def embed_single(seq):
    seq = seq.strip().upper()
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        output = esm_model(tokens, repr_layers=list(range(esm_model.num_layers - 4, esm_model.num_layers)))
        reps = [output["representations"][l][0, 1:len(seq)+1] for l in range(esm_model.num_layers - 4, esm_model.num_layers)]
        residue_reps = torch.stack(reps).mean(0)
        pooled = residue_reps.mean(0)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.cpu().numpy()

def compute_aac(seq):
    seq = seq.upper()
    counts = Counter(seq)
    total = len(seq)
    return np.array([counts.get(aa, 0) / total for aa in AA])

def compute_dpc(seq):
    seq = seq.upper()
    pairs = [''.join(p) for p in product(AA, repeat=2)]
    counts = Counter([seq[i:i+2] for i in range(len(seq)-1)])
    total = len(seq) - 1
    return np.array([counts.get(p, 0) / total for p in pairs])

def biopy_features(seq):
    seq = seq.upper()
    prot = ProteinAnalysis(seq)
    aa_frac = list(prot.get_amino_acids_percent().values())
    gravy = prot.gravy()
    charge = prot.charge_at_pH(7.0)
    aromaticity = prot.aromaticity()
    instability = prot.instability_index()
    return np.array(aa_frac + [gravy, charge, aromaticity, instability])

# ---- Load trained model ----
print("Loading trained model GB-IFN.xgb...")
model = xgb.XGBClassifier()
model.load_model("GB-IFN.xgb")

# ---- Prediction ----
def predict_ifn_gamma(epitope):
    print(f"\nPredicting for epitope: {epitope}")
    emb = embed_single(epitope)
    aac = compute_aac(epitope)
    dpc = compute_dpc(epitope)
    bio = biopy_features(epitope)

    X = np.hstack([emb, bio, dpc])  # same feature order used in training
    X = X.reshape(1, -1)

    prob = model.predict_proba(X)[0, 1]
    print(f"\nPredicted IFN-γ induction probability: {prob:.4f}")
    return prob
epitope = input("Enter peptide sequence: ").strip()
if len(epitope) == 0:
    print("No sequence entered.")
else:
    predict_ifn_gamma(epitope)
