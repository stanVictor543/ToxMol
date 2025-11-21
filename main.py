import pandas as pd

df = pd.read_csv("tox21_data.xlsx")

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Function to convert SMILES to Morgan fingerprint (vector)
def smiles_to_features(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
    else:
        return [0]*nBits


X = np.array([smiles_to_features(s) for s in df['smiles']])
y = df['label'].values  

