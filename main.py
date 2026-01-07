import os
import shutil
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdFingerprintGenerator
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve
import deepchem as dc


# --- CONFIGURARE ---
TARGET_ASSAY = 'NR-AR'
DATA_DIR = './tox21_data'
N_BITS = 2048
RADIUS = 2

#Initializare generator de fingerprint-uri
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
def clean_and_load_data(data_dir):
    """Încarcă datele și curăță automat cache-ul în caz de eroare."""
    try:
        print(f"[*] Se încarcă datele Tox21 în: {data_dir}...")
        return dc.molnet.load_tox21(data_dir=data_dir)
    except Exception as e:
        print(f"[!] Eroare detectată: {e}")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print("[+] Cache curățat. Reîncercare...")
            return dc.molnet.load_tox21(data_dir=data_dir)
        raise e

def smiles_to_fp(smiles):
    """Conversie sigură SMILES -> Morgan Fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
           
            return morgan_gen.GetFingerprintAsNumPy(mol)
    except:
        return None
    return None

def preprocess_dataset(dataset, target_index):
    """Curățare NaN și generare Fingerprints."""
    print(f"[*] Procesare set date (inițial: {len(dataset)})...")
    y_all = dataset.y[:, target_index]
    smiles_all = dataset.ids
    
    valid_x, valid_y = [], []
    for smiles, y in zip(smiles_all, y_all):
        if not np.isnan(y):
            fp = smiles_to_fp(smiles)
            if fp is not None:
                valid_x.append(fp)
                valid_y.append(int(y))
                
    return np.array(valid_x), np.array(valid_y)

def evaluate_metrics(model, X, y, threshold=0.5):
    """Calculare metrici și returnare probabilități."""
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        'AUC-ROC': roc_auc_score(y, probs),
        'F1': f1_score(y, preds),
        'Acc': accuracy_score(y, preds),
        'Probs': probs
    }

# --- FLUX PRINCIPAL ---

# 1. Încărcare Date
tasks, datasets, transformers = clean_and_load_data(DATA_DIR)
target_idx = tasks.index(TARGET_ASSAY)

# 2. Pregătire Date
X_train, y_train = preprocess_dataset(datasets[0], target_idx)
X_test, y_test = preprocess_dataset(datasets[2], target_idx)

# 3. Model 1: Bernoulli Naive Bayes 
print("\n[+] Antrenare BernoulliNB (GridSearch)...")
nb_grid = GridSearchCV(BernoulliNB(), {'alpha': [0.01, 0.1, 1.0]}, cv=5, scoring='roc_auc', n_jobs=-1)
nb_grid.fit(X_train, y_train)
best_nb = nb_grid.best_estimator_

# 4. Model 2: K-Nearest Neighbors (KNN)

print("[+] Antrenare KNN (GridSearch pentru K)...")
knn_grid = GridSearchCV(KNeighborsClassifier(n_jobs=1), {'n_neighbors': [3, 5, 7, 11]}, cv=5, scoring='roc_auc')
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_
print(f"    -> Cel mai bun K găsit: {best_knn.n_neighbors}")

# 5. Evaluare și Optimizare Prag pentru KNN
print("\n--- REZULTATE FINALE ---")
res_nb = evaluate_metrics(best_nb, X_test, y_test)
res_knn = evaluate_metrics(best_knn, X_test, y_test)

# Optimizare Prag pentru KNN (pentru a îmbunătăți F1-Score)
precision, recall, thresholds = precision_recall_curve(y_test, res_knn['Probs'])
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_thr = thresholds[np.argmax(f1_scores)]
res_knn_opt = evaluate_metrics(best_knn, X_test, y_test, threshold=best_thr)

# 6. Afișare Tabelară
results_df = pd.DataFrame({
    'Metrică': ['AUC-ROC', 'F1-Score', 'Acuratețe'],
    'BernoulliNB': [res_nb['AUC-ROC'], res_nb['F1'], res_nb['Acc']],
    'KNN (Standard 0.5)': [res_knn['AUC-ROC'], res_knn['F1'], res_knn['Acc']],
    'KNN (Optimizat)': [res_knn_opt['AUC-ROC'], res_knn_opt['F1'], res_knn_opt['Acc']]
})

print(results_df.to_markdown(index=False))
print(f"\n[*] Prag optim pentru KNN: {best_thr:.4f}")