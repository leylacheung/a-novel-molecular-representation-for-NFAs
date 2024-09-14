import re
import logging
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, PandasTools
from gensim.models import Word2Vec
from Ringfeature import *
from visualization import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV, convert SMILES to RDKit molecules, and generate sentence representations.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing SMILES strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with molecule and sentence columns.
    """
    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Data loaded, number of rows: %d", len(df))
    
    df['Mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df['Sentence'] = df['Mol'].apply(ring2sentence)
    
    logger.info("Sentence generation complete.")
    return df

def train_word2vec(sentences: List[List[str]], vector_size: int = 100, window: int = 10,
                   min_count: int = 3, n_jobs: int = 4, method: str = 'cbow') -> Word2Vec:
    """
    Train a Word2Vec model.

    Parameters
    ----------
    sentences : List[List[str]]
        Sentences for training the Word2Vec model.
    vector_size : int, optional
        Dimensionality of the word vectors, by default 100.
    window : int, optional
        Maximum distance between current and predicted word, by default 10.
    min_count : int, optional
        Ignores all words with total frequency lower than this, by default 3.
    n_jobs : int, optional
        Number of threads to use, by default 4.
    method : str, optional
        Training method, either 'skip-gram' or 'cbow', by default 'cbow'.

    Returns
    -------
    Word2Vec
        Trained Word2Vec model.
    """
    sg = 0 if method == 'cbow' else 1
    start = timeit.default_timer()
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=n_jobs, sg=sg)
    stop = timeit.default_timer()
    
    logger.info("Word2Vec model trained in %f minutes", (stop - start) / 60)
    return model

def save_model(model: Word2Vec, model_path: str):
    """
    Save the trained Word2Vec model.

    Parameters
    ----------
    model : Word2Vec
        The trained Word2Vec model.
    model_path : str
        Path where the model will be saved.
    """
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

def get_ring_fingerprints(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> List[List[int]]:
    """
    Extract rings from molecules and obtain Morgan fingerprints for the rings.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.
    radius : int, optional
        Radius of Morgan fingerprints, by default 2.
    n_bits : int, optional
        Size of the fingerprint bit vector, by default 2048.

    Returns
    -------
    List[List[int]]
        List of Morgan fingerprints for the rings.
    """
    sssr = Chem.GetSymmSSSR(mol)
    ring_fps = []

    for ring in sssr:
        ring_atoms = list(ring)
        ring_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useFeatures=True, useChirality=True, atomsToUse=ring_atoms)
        ring_fps.append(list(ring_fp))

    return ring_fps

def display_vocab(model: Word2Vec):
    """
    Display vocabulary of the trained Word2Vec model.

    Parameters
    ----------
    model : Word2Vec
        The trained Word2Vec model.
    """
    vocab = list(model.wv.key_to_index.keys())
    logger.info("Vocabulary size: %d", len(vocab))
    
    for word in vocab[:10]:  # Show first 10 words as a sample
        logger.info("Word: %s \tVector: %s", word, model.wv[word])

def main():
    # Load and preprocess the CEPDB dataset
    df = load_and_preprocess_data('data/CEPDB.csv')

    # Prepare sentences for Word2Vec training
    sentences = [sentence for sentence in df['Sentence'] if sentence]
    
    logger.info("Training Word2Vec model on prepared sentences")
    model = train_word2vec(sentences, vector_size=100, window=10, min_count=3, n_jobs=4, method='cbow')
    
    # Save the trained model
    save_model(model, "models/pretmodelcbow_100dim.pkl")
    
    # Display vocabulary information
    display_vocab(model)

if __name__ == "__main__":
    main()


