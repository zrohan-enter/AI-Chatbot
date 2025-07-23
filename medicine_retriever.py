import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class MedicineRetriever:
    def __init__(self, 
                 index_path='medicine_faiss.index',
                 data_path='medicine_data.pkl',
                 model_name='all-MiniLM-L6-v2'):
        print("Loading retrieval system...")
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.df = pd.read_pickle(data_path)
        print("Loaded index and data.")

    def query(self, question, top_k=3):
        q_emb = self.model.encode([question])
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for idx in indices[0]:
            row = self.df.iloc[idx]
            results.append({
                'Medicine Name': row['Medicine Name'],
                'Composition': row['Composition'],
                'Uses': row['Uses'],
                'Side Effects': row['Side_effects'],
                'Manufacturer': row['Manufacturer'],
                'Excellent Review %': row['Excellent Review %'],
                'Average Review %': row['Average Review %'],
                'Poor Review %': row['Poor Review %']
            })
        return results
