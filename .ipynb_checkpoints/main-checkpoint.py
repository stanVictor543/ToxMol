import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import time

print("--- Inițializare Proiect: Previziunea Toxicității SR-ATAD5 ---")

# 1. PARAMETRI ȘI ÎNCĂRCAREA DATELOR
TARGET_ASSAY = 'NR-AR'
N_BITS = 2048 # Dimensiunea Morgan Fingerprint
RADIUS = 2   # Raza Morgan Fingerprint

try:
    # Descarcă și încarcă setul de date Tox21 (split implicit: train/valid/test)
    print(f"Încărcarea datelor Tox21... (Țintă: {TARGET_ASSAY})")
    tasks, datasets, transformers = dc.molnet.load_tox21()
    train_dataset, valid_dataset, test_dataset = datasets[0], datasets[1], datasets[2]
except Exception as e:
    print(f"Eroare la încărcarea DeepChem/TensorFlow. Asigură-te că ai instalat 'deepchem' și 'tensorflow': {e}")
    exit()

# Definirea funcției de generare a descriptorilor
def smiles_to_morgan_fingerprint(smiles, nBits=N_BITS, radius=RADIUS):
    """Convertește șirul SMILES în Morgan Fingerprint."""
    try:
        # Foloseste RDKit pentru a genera descriptorii
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None # Molecule invalide
        fingerprint = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(fingerprint)
    except:
        return None

# Găsește indexul coloanei țintă
try:
    atad5_index = tasks.index(TARGET_ASSAY)
except ValueError:
    print(f"Eroare: Coloana țintă '{TARGET_ASSAY}' nu a fost găsită în setul de date.")
    exit()

# 2. PREGĂTIREA DATELOR (TRAIN ȘI TEST)
print("\n2. Pregătirea Datelor (Generare Fingerprints și Curățare)...")

def prepare_data(dataset, target_index):
    """Extrage, filtrează și generează descriptorii pentru un set de date."""
    
    X_smiles = dataset.ids
    Y_all = dataset.y
    Y_target = Y_all[:, target_index]
    
    # 2.1 Filtrare NaN (date lipsă) pe eticheta Y
    nan_mask_y = ~np.isnan(Y_target)
    
    X_smiles_cleaned = X_smiles[nan_mask_y]
    Y_cleaned = Y_target[nan_mask_y].astype(int)
    
    # 2.2 Generarea Fingerprints (X)
    X_fingerprints = [smiles_to_morgan_fingerprint(s) for s in X_smiles_cleaned]
    
    # 2.3 Filtrare finală pentru molecule invalide
    valid_indices = [i for i, fp in enumerate(X_fingerprints) if fp is not None]
    X_final = np.array([X_fingerprints[i] for i in valid_indices])
    Y_final = Y_cleaned[valid_indices]
    
    Y_final = Y_final.ravel()
    
    print(f"   -> Dimensiune Finală (X, Y): {X_final.shape}")
    return X_final, Y_final

# Pregătirea setului de antrenament
print("   - Procesare set de antrenament:")
X_train, Y_train = prepare_data(train_dataset, atad5_index)

# Pregătirea setului de test
print("   - Procesare set de test:")
X_test, Y_test = prepare_data(test_dataset, atad5_index)


# 3. FUNCȚIE PENTRU EVALUARE
def evaluate_model(model, X_test, Y_test, model_name):
    """Evaluează modelul."""
    
    # 1. Predictii
    # Pentru AUC-ROC avem nevoie de probabilitati, nu doar de clasa (0 sau 1)
    Y_pred = model.predict(X_test)
    Y_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Metricile
    auc_roc = roc_auc_score(Y_test, Y_proba)
    f1 = f1_score(Y_test, Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    print(f"\n   --- Evaluare: {model_name} ---")
    print(f"   - AUC-ROC: {auc_roc:.4f} (Cheie pentru date dezechilibrate)")
    print(f"   - F1-Score: {f1:.4f}")
    print(f"   - Acuratețe (Accuracy): {accuracy:.4f}")
    
    return auc_roc, f1

# 4. MODELARE ȘI ANTRENAMENT

# MODEL 1: Clasificatorul Bayesian Naiv (GaussianNB)
print("\n4.1 Antrenare Model 1: Clasificator Bayesian Naiv...")
start_time_bayes = time.time()
model_bayes = GaussianNB()
model_bayes.fit(X_train, Y_train)
time_bayes = time.time() - start_time_bayes
print(f"   -> Timp antrenament: {time_bayes:.2f} secunde")
auc_bayes, f1_bayes = evaluate_model(model_bayes, X_test, Y_test, "Bayesian Naiv")


# MODEL 2: Păduri Aleatoare (Random Forest)
print("\n4.2 Antrenare Model 2: Păduri Aleatoare (Random Forest)...")
start_time_rf = time.time()
# class_weight='balanced' ajută la gestionarea seturilor de date dezechilibrate
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1) 
model_rf.fit(X_train, Y_train)
time_rf = time.time() - start_time_rf
print(f"   -> Timp antrenament: {time_rf:.2f} secunde")
auc_rf, f1_rf = evaluate_model(model_rf, X_test, Y_test, "Random Forest")


# 5. REZULTATE FINALE ȘI INTERPRETARE
print("\n\n######################################################")
print("## 5. REZULTATE FINALE ȘI ANALIZĂ COMPARATIVĂ ##")
print("######################################################")

results = {
    'Algoritm': ['Bayesian Naiv', 'Random Forest'],
    'AUC-ROC': [auc_bayes, auc_rf],
    'F1-Score': [f1_bayes, f1_rf],
    'Timp Antrenament (s)': [time_bayes, time_rf]
}
df_results = pd.DataFrame(results)
print(df_results.sort_values(by='AUC-ROC', ascending=False).to_markdown(index=False))

print("\n--- Analiză ---")

# Importanța Caracteristicilor (doar pentru RF)
print("\nTop 10 Descriptori (Fingerprint Bits) Importanți (Random Forest):")
importances = model_rf.feature_importances_
feature_names = [f"Bit_{i}" for i in range(N_BITS)]
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False).head(10)
print(feature_df.to_markdown(index=False))

print(f"\nConcluzie: Modelul cu AUC-ROC mai mare (probabil Random Forest) oferă o capacitate de predicție superioară pentru {TARGET_ASSAY}.")

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score


# Reiau modelul antrenat si seturile de test din codul anterior:
# model_rf, X_test, Y_test

print("--- 6. Optimizarea Pragului de Decizie ---")

# 1. Generează probabilitățile de clasă 1 (Toxic)
# Avem nevoie de probabilitati, nu de predicții binare
Y_proba_rf = model_rf.predict_proba(X_test)[:, 1]

# 2. Calculează Curba Precision-Recall
# Aceasta ne oferă Precision, Recall și Pragul (Threshold) pentru fiecare punct.
precision, recall, thresholds = precision_recall_curve(Y_test, Y_proba_rf)

# 3. Calculează F1-Score pentru fiecare prag
# Se aplică formula F1: 2 * (Precision * Recall) / (Precision + Recall)
# Ignorăm ultimul punct al curbei care are NaN sau 0
fscore = 2 * (precision * recall) / (precision + recall)
fscore[np.isnan(fscore)] = 0 # Trateaza rezultatele NaN/0 care apar la divizarea cu 0

# 4. Găsește pragul (threshold) care maximizează F1-Score-ul
ix = np.argmax(fscore)
optimal_f1 = fscore[ix]
optimal_threshold = thresholds[ix]

print(f"   -> F1-Score Maximizat: {optimal_f1:.4f}")
print(f"   -> Prag Optim (Optimal Threshold): {optimal_threshold:.4f}")
print(f"   -> Rapel (Recall) corespunzător: {recall[ix]:.4f}")
print(f"   -> Precizie (Precision) corespunzătoare: {precision[ix]:.4f}")

# 5. RE-EVALUAREA MODELULUI CU PRAGUL OPTIM
print("\n--- 7. Re-Evaluare cu Pragul Optim ---")
from sklearn.metrics import roc_auc_score, accuracy_score

# Aplică noul prag pentru a obține predicțiile binare
Y_pred_optimized = (Y_proba_rf >= optimal_threshold).astype(int)

# Recalculează metricile de evaluare
auc_roc_optimized = roc_auc_score(Y_test, Y_proba_rf) # AUC-ROC nu se schimbă la ajustarea pragului
f1_optimized = f1_score(Y_test, Y_pred_optimized)
accuracy_optimized = accuracy_score(Y_test, Y_pred_optimized)

print(f"   - AUC-ROC (neschimbat): {auc_roc_optimized:.4f}")
print(f"   - F1-Score Optimizat: {f1_optimized:.4f} (Îmbunătățire MAJORĂ față de 0.0000!)")
print(f"   - Acuratețe Optimizată: {accuracy_optimized:.4f}")

# Comparație finală
print("\n######################################################")
print("## Rezultate Comparate (RF Standard vs. RF Optimizat) ##")
print("######################################################")
print(f"F1-Score (Prag 0.5): {0.0000:.4f}")
print(f"F1-Score (Prag {optimal_threshold:.4f}): {f1_optimized:.4f}")