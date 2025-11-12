# GB-IFN: Interferon-Gamma Epitope Prediction Model

GB-IFN is an **XGBoost-based classifier** designed to predict whether a peptide (epitope) can induce **IFN-γ (Interferon-Gamma)** responses.  
It combines **protein language model embeddings (ESM-2)** with **physicochemical** and **sequence-based** features such as AAC, DPC, and Biopython descriptors.

---

## Features Used
- **ESM-2 (T12-35M) Embeddings** — captures high-level sequence representations.  
- **Amino Acid Composition (AAC)** — frequency of each amino acid.  
- **Dipeptide Composition (DPC)** — frequency of amino acid pairs.  
- **Biopython Features** — charge, aromaticity, GRAVY, instability index, amino acid fractions.

---

## Usage

The model can be used directly provided all dependencies are installed and the main model file is installed
usemodel.py is the file required for usage
