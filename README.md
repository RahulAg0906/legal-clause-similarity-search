# Legal Clause Semantic Search Engine

**COMP8420 Assignment 3 — NLP Application Project**  
**Author:** Rahul Aggarwal  
**Student ID:** 47757612

---

## Overview

This project implements a **semantic search system for legal contract clauses** using transformer-based embeddings and unsupervised clustering.

Traditional keyword search fails when clause wording varies. This system enables **meaning-based retrieval**, where users can find semantically relevant clauses — even if wording differs — using **dense transformer embeddings** and **cosine similarity**.

We further explore the embedding space using **clustering (KMeans, DBSCAN)** and visualize clause structure with **t-SNE**.

---

## Use Case

**Input:** _"Termination of agreement"_  
**Goal:** Return top-5 most semantically similar clauses from a contract corpus, regardless of keyword overlap.

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_data_loading_and_preprocessing.ipynb
│   ├── 02_clause_embedding_and_search.ipynb
│   ├── 03_clause_clustering.ipynb
│   └── 04_retrieval_evaluation.ipynb
├── results/
│   ├── clause_embeddings.npy
│   ├── clause_metadata.pkl
│   ├── clause_nn_model.pkl
│   ├── clause_clusters.pkl
│   ├── clause_embeddings_pca50.npy
│   ├── pca_model.pkl
│   └── cuad_normalized_clauses.csv
├── dataset/CUAD_v1/
│   ├── full_contract_pdf/
│   ├── full_contract_txt/
│   ├── CUAD_v1.json
│   ├── master_clauses.csv
│   └── label_group_xlsx/
├── README.md
└── requirements.txt
```

---

## Setup Instructions

1. **Python Environment Setup**

Ensure Python 3.10 or newer is installed. Then:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

2. **Required Python Packages**

pandas>=1.5.3
numpy>=1.23.5
matplotlib>=3.6.2
scikit-learn>=1.2.2
sentence-transformers>=2.2.2
torch>=1.13.1
tqdm>=4.64.1
umap-learn>=0.5.3
ipython>=8.10.0


3. **Data**

Expected input file format (CSV):

| clause_text | clause_type | context |
|-------------|-------------|---------|

---

## How to Run

### 1. Preprocess Clauses

Open: `01_data_loading_and_preprocessing.ipynb`

- Loads raw CUAD JSON
- Extracts & normalizes clause types
- Outputs `cuad_normalized_clauses.csv`

---

### 2. Generate Clause Embeddings + Search

Run: `02_clause_embedding_and_search.ipynb`

- Embeds each clause using:  
  `sentence-transformers/msmarco-MiniLM-L6-cos-v5`
- Builds cosine-based NearestNeighbors index
- Defines `semantic_search_df()` function

---

### 3. Cluster Clauses

Run: `03_clause_clustering.ipynb`

- Reduces embeddings via PCA
- Applies KMeans and DBSCAN
- Saves cluster assignments
- Visualizes in 2D with t-SNE

---

### 4. Evaluate Search Performance

Run: `04_retrieval_evaluation.ipynb`

- Uses a curated test set of queries + expected clause types
- Evaluates using:
  - **Precision@5**
  - **MRR**
  - **Max Cosine Similarity**
  - **Cluster Cohesion**
- Includes fuzzy type matching and ablation study

---

## Example Output

> **Query:** `"Force majeure clause"`  
> **Top Clause Types:** `["Expiration Date", "Cap on Liability", "Force Majeure"]`  
> **Precision@5:** `0.8`  
> **MRR:** `1.0`  
> **Max Cosine Similarity:** `0.78`

---

## Key Techniques

- Transformer embeddings via `msmarco-MiniLM-L6-cos-v5`
- Cosine similarity + NearestNeighbors index
- Fuzzy label matching (soft evaluation)
- KMeans + DBSCAN for clause clustering
- t-SNE for 2D visualization
- PCA for dimensionality reduction
- Cluster cohesion analysis

---

## Ablation Study Summary

| Phase | Change Introduced          | Precision@5 | MRR   |
|-------|----------------------------|-------------|-------|
| 1     | Fuzzy clause type matching | ↑           | ↑     |
| 2     | Swapped in legal model     | ↑↑          | ↑↑    |
| 3     | Expanded query set         | ~           | ↓     |
| 4     | Cosine sim scoring added   | ↑           | ↑     |
| 5     | Cluster-based evaluation   | Qualitative | ↑     |

---

## Author

- **Rahul Aggarwal**  
  Student ID: 47757612  
  COMP8420 – NLP Applications  

---

## License

For academic use only (ANU COMP8420 Assignment 3).