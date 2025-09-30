# datasets.py
from tdc.single_pred import ADME
from rdkit import Chem
import features
import numpy as np
from experiments import Method
import pandas as pd
import os

class Dataset():
    def __init__(self, method: Method):
        """
        Initialize dataset with appropriate input representation based on method
        
        Args:
            method: Method object from MethodRegistry
        """

        if method.dataset.startswith('Solubility_'):
            # Extract percentage from dataset name (e.g., 'Solubility_010' -> 10%)
            parts = method.dataset.split('_')
            if len(parts) > 1:
                percentage_str = parts[-1]
                percentage = int(percentage_str)
            else:
                percentage = 100
            
            data = ADME(name='Solubility_AqSolDB')
            df = data.get_data()
            
            # Sample the specified percentage of the dataset
            if percentage < 100:
                df = df.sample(frac=percentage/100, random_state=42)

            smiles = df['Drug']
            mols = [Chem.MolFromSmiles(x) for x in smiles]
            self.X = features.getFeature(mols, method.feature)
            labels = df['Y']
        else:
            try:
                data = ADME(name=method.dataset)
                df = data.get_data()
                smiles = df['Drug']
                mols = [Chem.MolFromSmiles(x) for x in smiles]
                self.X = features.getFeature(mols, method.feature)
                labels = df['Y']
            except:
                featurized_path = "./data/featurized.csv"
                
                # Check if featurized data already exists
                if os.path.exists(featurized_path):
                    print(f"Loading pre-computed features from {featurized_path}")
                    df_feat = pd.read_csv(featurized_path)
                    
                    # Extract X (all columns except 'Dock') and Y (Dock column)
                    self.X = df_feat.drop(columns=['Dock']).values
                    self.feature_names = df_feat.drop(columns=['Dock']).columns.tolist()
                    self.Y = df_feat['Dock'].values
                    
                else:
                    print("Computing features from scratch...")
                    df = pd.read_csv(f"./data/{method.dataset}")
                    df = df[(df['SMILES_LIGANDS'] != '') & (df['SMILES_LIGANDS'].notna())]
                    labels = df['Dock']
                    smiles_CD = df['SMILES_CD']
                    smiles_LIG = df['SMILES_LIGANDS']
                    
                    mols_CD = [Chem.MolFromSmiles(x) for x in smiles_CD]
                    mols_LIG = [Chem.MolFromSmiles(x) for x in smiles_LIG]
                    
                    # Generate features
                    ECFP_CD = features.getFeature(mols_CD, 'ECFP')
                    ECFP_LIG = features.getFeature(mols_LIG, 'ECFP')
                    RDKit_CD = features.getFeature(mols_CD, 'RDKit')
                    RDKit_LIG = features.getFeature(mols_LIG, 'RDKit')
                    
                    # Get feature names
                    ecfp_cd_names = [f"ECFP_REC_{i}" for i in range(ECFP_CD.shape[1])]
                    ecfp_lig_names = [f"ECFP_LIG_{i}" for i in range(ECFP_LIG.shape[1])]
                    rdkit_cd_names = features.get_rdkit_descriptor_names(prefix="RDKIT_REC")
                    rdkit_lig_names = features.get_rdkit_descriptor_names(prefix="RDKIT_LIG")
                    
                    # Combine all feature names
                    all_feature_names = ecfp_cd_names + ecfp_lig_names + rdkit_cd_names + rdkit_lig_names
                    
                    # Combine features
                    self.X = np.concatenate([ECFP_CD, ECFP_LIG, RDKit_CD, RDKit_LIG], axis=1)
                    self.Y = np.array(labels)
                    
                    # Create DataFrame with feature names and save
                    df_feat = pd.DataFrame(self.X, columns=all_feature_names)
                    df_feat['Dock'] = self.Y
                    df_feat.to_csv(featurized_path, index=False)
                    print(f"Features saved to {featurized_path}")
                
                return
        
        self.Y = np.array(labels)
                

