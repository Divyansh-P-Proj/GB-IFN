!pip install fair-esm
!pip install torch-geometric
!pip install biopython
!pip install propy3
import esm
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def embed_batch(seqs, batch_size=48):
    all_embs = []
    last_layers = list(range(model.num_layers - 4, model.num_layers))

    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size), desc="Embedding batches"):
            batch = seqs[i:i+batch_size]
            data = [(f"seq{i+j}", s.strip().upper()) for j, s in enumerate(batch)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            output = model(tokens, repr_layers=last_layers)
            batch_embs = []
            for b, seq in enumerate(batch):
                seq_len = len(seq)
                reps = [output["representations"][l][b, 1:seq_len+1] for l in last_layers]
                residue_reps = torch.stack(reps).mean(0)
                pooled = residue_reps.mean(0)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                batch_embs.append(pooled.cpu().numpy())

            all_embs.extend(batch_embs)

    return np.stack(all_embs)

DATA = pd.read_csv("ifnneg.csv")
DATA1 = pd.read_csv("ifnpos.csv")

DATA["Measure"] = 0
DATA1["Measure"] = 1

AA = list("ACDEFGHIKLMNPQRSTVWY")

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
    return aa_frac + [gravy, charge, aromaticity, instability]

df = pd.concat([DATA, DATA1], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
y = df["Measure"].astype(int).values
emb_path = "epitope_embeddings.npy"
feat_path = "biopy_features.npy"
aac_path = "aac.npy"
dpc_path = "dpc.npy"

if os.path.exists(emb_path):
    print(f"Loading cached embeddings from {emb_path}")
    X_emb = np.load(emb_path)
else:
    print("Generating embeddings...")
    X_emb = embed_batch(df["Epitope"].tolist())
    np.save(emb_path, X_emb)


if os.path.exists(feat_path):
    print(f"Loading cached Biopython features from {feat_path}")
    X_feat = np.load(feat_path)
else:
    print("Generating Biopython features...")
    X_feat = np.array([biopy_features(seq) for seq in tqdm(df["Epitope"], desc="BioPy features")])
    np.save(feat_path, X_feat)

if os.path.exists(aac_path):
    print(f"Loading cached AAC features from {aac_path}")
    X_aac = np.load(aac_path)
else:
    print("Generating AAC features...")
    X_aac = np.array([compute_aac(seq) for seq in tqdm(df["Epitope"], desc="AAC features")])
    np.save(aac_path, X_aac)


if os.path.exists(dpc_path):
    print(f"Loading cached DPC features from {dpc_path}")
    X_dpc = np.load(dpc_path)
else:
    print("Generating DPC features...")
    X_dpc = np.array([compute_dpc(seq) for seq in tqdm(df["Epitope"], desc="DPC features")])
    np.save(dpc_path, X_dpc)


X = np.hstack([X_emb, X_feat, X_dpc])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "device": "cuda",
    "n_jobs": -1,
    "n_estimators": 956,
    "learning_rate": 0.06791432575549611,
    "max_depth": 9,
    "subsample": 0.951314532949814,
    "colsample_bytree": 0.828664727299987,
    "gamma": 0.30031493663519104,
    "reg_lambda": 3.416223882946928,
    "reg_alpha": 0.035127400047873694,
    "min_child_weight": 1,
    "random_state": 42,
    "early_stopping_rounds":50
}

model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print("Final AUC-ROC:", auc)

