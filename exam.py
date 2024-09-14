import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from gensim.models import word2vec
from Ringfeature import *
from typing import List

# Ensure matplotlib inline mode is used for interactive environments (optional)
# %matplotlib inline

# Add the ring2vec directory to the system path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'ring2vec'))

def load_word2vec_model(model_path: str) -> word2vec.Word2Vec:
    """
    Load a pre-trained Word2Vec model from the specified file path.
    
    Parameters:
    -----------
    model_path : str
        Path to the pre-trained Word2Vec model.
    
    Returns:
    --------
    word2vec.Word2Vec
        Loaded Word2Vec model.
    """
    return word2vec.Word2Vec.load(model_path)

def process_smiles(df: pd.DataFrame, model: word2vec.Word2Vec) -> pd.DataFrame:
    """
    Process SMILES from a DataFrame, generate molecular sentences, and compute ring2vec features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing SMILES strings in a column named 'SMILES'.
    model : word2vec.Word2Vec
        Pre-trained Word2Vec model to generate molecular feature vectors.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional columns for molecular features.
    """
    # Convert SMILES to RDKit molecule objects and drop invalid rows
    df['Mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['Mol'])

    # Generate ring-based molecular sentences
    df['Sentence'] = df['Mol'].apply(ring2alt_sentence)

    # Filter valid sentences
    sentences = [sentence for sentence in df['Sentence'] if sentence]

    # Compute feature vectors using the pre-trained Word2Vec model
    df['ring2vec'] = [DfVec(x) for x in sentences2vec(sentences, model)]

    # Convert feature vectors to comma-separated string format
    df['Features'] = df['ring2vec'].apply(lambda x: ','.join(map(str, x.vec)))

    return df

def save_processed_data(df: pd.DataFrame, output_path: str, feature_columns: List[str]) -> None:
    """
    Save the processed DataFrame to a CSV file, including selected columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The processed DataFrame with feature vectors.
    output_path : str
        The path where the CSV file should be saved.
    feature_columns : list of str
        The list of columns to include in the output file.
    
    Returns:
    --------
    None
    """
    df_to_save = df[feature_columns]
    df_to_save.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

# Main script
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('data/NFAs_cleaned.csv')
    df.columns = df.columns.str.strip()  # Clean column names by stripping whitespace

    # Load the pre-trained Word2Vec model
    model = load_word2vec_model('models/pretmodelskip4_100dim.pkl')

    # Process the SMILES data and generate molecular feature vectors
    df = process_smiles(df, model)

    # Save the processed data to a CSV file
    save_processed_data(df, 'data/NFAs_prering2alt.csv', ['SMILES', 'Features', 'HOMO', 'LUMO', 'bandgap'])


