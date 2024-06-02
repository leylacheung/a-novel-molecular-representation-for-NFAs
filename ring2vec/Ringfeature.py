"""
Features - Main Ring2vec Module
==============================
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from gensim.models import word2vec
import timeit
from joblib import Parallel, delayed 
from rdkit.Chem import rdMolDescriptors, Descriptors

class DfVec(object):
    """
    Helper class to store vectors in a pandas DataFrame
    
    Parameters  
    ---------- 
    vec: np.array
    """
    def __init__(self, vec):
        self.vec = vec
        if type(self.vec) != np.ndarray:
            raise TypeError('numpy.ndarray expected, got %s' % type(self.vec))

    def __str__(self):
        return "%s dimensional vector" % str(self.vec.shape)

    __repr__ = __str__

    def __len__(self):
        return len(self.vec)

    _repr_html_ = __str__

class MolSentence:
    """Class for storing mol sentences in pandas DataFrame
    """
    def __init__(self, sentence):
        # If the provided list is empty, initialize a list containing an empty string to avoid IndexError.
        if not sentence:
            self.sentence = [""]
        else:
            self.sentence = sentence
        # Check if the first element in the list is a string; if not, raise a TypeError.
        if not all(isinstance(word, str) for word in self.sentence):
            raise TypeError('List with strings expected')

    def __len__(self):
        return len(self.sentence)

    def __str__(self):  # String representation
        return 'MolSentence with %i words' % len(self.sentence)

    __repr__ = __str__  # Default representation

    def contains(self, word):
        """Contains (and __contains__) method enables usage of "'Word' in MolSentence"""
        if word in self.sentence:
            return True
        else:
            return False

    __contains__ = contains  # MolSentence.contains('word')

    def __iter__(self):  # Iterate over words (for word in MolSentence:...)
        for x in self.sentence:
            yield x

    _repr_html_ = __str__

def clean_smiles_and_save(input_csv, output_csv):
    # load data
    df = pd.read_csv(input_csv)
    
    # Attempt to convert SMILES to a molecule object.
    df['Mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    
    # Remove rows with failed conversion (invalid molecules).
    df = df[df['Mol'].notnull()]
    
    # Delete the Mol column as it cannot be correctly saved to a CSV file.
    df.drop(columns=['Mol'], inplace=True)
    
    # Save the processed data to a new CSV file.
    df.to_csv(output_csv, index=False)
    
    print(f"The cleaned data has been saved to {output_csv}")

electronegativity_table = {
    6: 2.55,  # C
    7: 3.04,  # N
    8: 3.44,  # O
    9: 3.98,  # F
    15: 2.19, # P
    16: 2.58, # S
    17: 3.16, # Cl
    35: 2.96, # Br
    53: 2.66, # I
    34: 2.55, # Se
    # ... other elements ...
}

def get_atom_fingerprints(mol):
    """Generate Morgan fingerprints for each atom in the molecule."""
    atom_fingerprints = []
    for idx in range(mol.GetNumAtoms()):
        atom_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=216, fromAtoms=[idx])
        atom_fingerprints.append(list(map(int, atom_fp.ToBitString())))
    return atom_fingerprints

# A function to convert the feature vectors of rings into a single vocabulary.
def vector_to_word(vector):
    """Convert feature vectors into a single string without spaces"""
    return '_'.join(map(str, vector))

def get_ring_features(mol, ring):
    """Retrieve feature vectors of rings"""
    # Generate fingerprint for the entire molecule.
    atom_fps = get_atom_fingerprints(mol)
    
    # The size of the ring.
    ring_size = len(ring)
    
    # Aromaticity
    aromatic = int(all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring))
    
    # Saturation, calculating the number of double and single bonds in the ring.
    double_bonds = sum(1 for idx in ring for nbr in mol.GetAtomWithIdx(idx).GetNeighbors() 
                       if mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE)
    single_bonds = sum(1 for idx in ring for nbr in mol.GetAtomWithIdx(idx).GetNeighbors() 
                       if mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.SINGLE)
    
    # The number of heteroatoms and electronegativity.
    heteroatoms = sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6)
    heteroatoms_electroneg = []
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        atomic_num = atom.GetAtomicNum()
        if atomic_num != 6:  
            # Retrieve the electronegativity of atoms from the electronegativity table.
            electronegativity = electronegativity_table.get(atomic_num, None)
            if electronegativity is not None:
                heteroatoms_electroneg.append(electronegativity)
    
    # The connectivity pattern of the ring system.
    sssr = mol.GetRingInfo().AtomRings()
    fused_rings = sum(1 for other_ring in sssr if set(ring).intersection(other_ring)) - 1
    
    # The types of atoms in the ring.
    atom_types = [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in ring]
    
    # Select the Morgan fingerprints of heteroatoms in the ring
    ring_atom_fps = [atom_fps[idx] for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6]

    # Merge Morgan fingerprints, ensuring correct merging of lists of different lengths.
    ring_fps_flattened = []
    for fp in ring_atom_fps:
        ring_fps_flattened.extend(fp)

    # Merge features into a single vector.
    ring_vector = [ring_size, aromatic, double_bonds, single_bonds, fused_rings, heteroatoms] + atom_types + heteroatoms_electroneg + ring_fps_flattened

    return vector_to_word(ring_vector)

def get_combined_ring_features(mol, ring1, ring2):
    """Retrieve the feature vectors of the combination of two rings."""
    # Generate fingerprint for the entire molecule.
    atom_fps = get_atom_fingerprints(mol)

    # Retrieve all atom indices in the combined rings
    combined_atoms = set(ring1).union(set(ring2))
    
    # The size of the combined rings.
    combined_size = len(combined_atoms)
    
    # Aromaticity
    aromatic = int(all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in combined_atoms))
    
    # Saturation, calculating the number of double and single bonds in the combined rings.
    double_bonds = sum(1 for idx in combined_atoms for nbr in mol.GetAtomWithIdx(idx).GetNeighbors() 
                       if mol.GetBondBetweenAtoms(idx, nbr.GetIdx()) and 
                       mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE)
    single_bonds = sum(1 for idx in combined_atoms for nbr in mol.GetAtomWithIdx(idx).GetNeighbors() 
                       if mol.GetBondBetweenAtoms(idx, nbr.GetIdx()) and 
                       mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.SINGLE)

    # The number of heteroatoms and electronegativity.
    heteroatoms = 0
    heteroatoms_electroneg = []
    for idx in combined_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atomic_num = atom.GetAtomicNum()
        if atomic_num != 6:  # If the atom is not a carbon atom.
            heteroatoms += 1
            electronegativity = electronegativity_table.get(atomic_num, None)
            if electronegativity is not None:
                heteroatoms_electroneg.append(electronegativity)
    # Get atom types for atoms in the combined rings
    atom_types = [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in combined_atoms]
    # The connectivity pattern of the ring system.
    sssr = mol.GetRingInfo().AtomRings()
    fused_rings = sum(1 for other_ring in sssr if set(ring1).intersection(other_ring) or set(ring2).intersection(other_ring)) - 1

    # Select the Morgan fingerprints of heteroatoms in the combined rings.
    combined_ring_atom_fps = [atom_fps[idx] for idx in combined_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6]

    # Merge Morgan fingerprints, ensuring correct merging of lists of different lengths.
    combined_fps_flattened = []
    for fp in combined_ring_atom_fps:
        combined_fps_flattened.extend(fp)

    # Merge features into a single vector.
    combined_ring_vector = [combined_size, aromatic, double_bonds, single_bonds, fused_rings, heteroatoms] + atom_types + heteroatoms_electroneg + combined_fps_flattened
    return vector_to_word(combined_ring_vector)
# def singlering2sentence(mol):
#     sssr = [list(ring) for ring in Chem.GetSymmSSSR(mol)]
#     ring_sentences = [vector_to_word(get_ring_features(mol, ring)) for ring in sssr]
#     return ' '.join(ring_sentences)

def singlering2sentence(mol):
    """Convert molecules into sentences based on individual rings (returning a list of tokens)"""
    sssr = [list(ring) for ring in Chem.GetSymmSSSR(mol)]
    ring_sentences = [get_ring_features(mol, ring) for ring in sssr]
    # Do not use join anymore, directly return a list of ring feature strings.
    return ring_sentences

def combring2sentence(mol):
    """Convert molecules into sentences based on combined rings"""
    sssr = [list(ring) for ring in Chem.GetSymmSSSR(mol)]
    ring_sentences = []

    for i, ring1 in enumerate(sssr):
        for j, ring2 in enumerate(sssr):
            if i < j and set(ring1).intersection(set(ring2)):
                combined_feature_str = ' '.join(map(str, get_combined_ring_features(mol, ring1, ring2)))
                ring_sentences.append(combined_feature_str)

    return ring_sentences

def ring2sentence(mol):
    """Convert molecules into sentences based on rings (including features of individual rings and combined rings)"""
    sssr = [list(ring) for ring in Chem.GetSymmSSSR(mol)]
    ring_sentences = []

    for ring in sssr:
        ring_feature_str = ' '.join(map(str, get_ring_features(mol, ring)))
        ring_sentences.append(ring_feature_str)

    for i, ring1 in enumerate(sssr):
        for j, ring2 in enumerate(sssr):
            if i < j and set(ring1).intersection(set(ring2)):
                combined_feature_str = ' '.join(map(str, get_combined_ring_features(mol, ring1, ring2)))
                ring_sentences.append(combined_feature_str)

    return ring_sentences

def ring2alt_sentence(mol):
    """Convert molecules into sentences based on rings, returning string representations of ring features"""
    sssr = Chem.GetSymmSSSR(mol)
    ring_features = [get_ring_features(mol, ring) for ring in sssr]

    adjacent_rings = {}
    for i, ring1 in enumerate(sssr):
        for j, ring2 in enumerate(sssr):
            if i < j and set(ring1).intersection(set(ring2)):
                adjacent_rings.setdefault(i, []).append(j)

    ring_sentences = []
    for i, ring in enumerate(sssr):
        # Convert ring feature vectors into strings
        ring_feature_str = ' '.join(map(str, ring_features[i]))
        ring_sentences.append(ring_feature_str)

        for adj_ring_idx in adjacent_rings.get(i, []):
            combined_features = get_combined_ring_features(mol, ring, sssr[adj_ring_idx])
            # Convert combined ring feature vectors into strings
            combined_feature_str = ' '.join(map(str, combined_features))
            ring_sentences.append(combined_feature_str)

    return ring_sentences

def _parallel_job(mol, r):
    """Helper function for joblib jobs
    """
    if mol is not None:
        # Convert molecules into SMILES strings
        smiles = Chem.MolToSmiles(mol)
        # Reconvert SMILES strings back into molecule objects
        mol = Chem.MolFromSmiles(smiles)
        # Retrieve alternating identifier sentences for molecules
        sentence = ring2alt_sentence(mol)
        # Convert alternating identifier sentences into strings and return
        return " ".join(sentence)

def _read_smi(file_name):
    while True:
        line = file_name.readline()
        if not line:
            break
        # Extract SMILES strings from each row and convert them into molecule objects
        yield Chem.MolFromSmiles(line.split('\t')[0])

def generate_corpus(in_file, out_file, r, sentence_type='alt', n_jobs=1):

    """Generates corpus file from sdf
    
    Parameters
    ----------
    in_file : str
        Input sdf
    out_file : str
        Outfile name prefix, suffix is either _r0, _r1, etc. or _alt_r1 (max radius in alt sentence)
    r : int
        Radius of morgan fingerprint
    sentence_type : str
        Options:    'all' - generates all corpus files for all types of sentences, 
                    'alt' - generates a corpus file with only combined alternating sentence, 
                    'individual' - generates corpus files for each radius
    n_jobs : int
        Number of cores to use (only 'alt' sentence type is parallelized)

    Returns
    -------
    """

    # File type detection
    # Split the filename and extension of the input file
    in_split = in_file.split('.')
    # Check if the file extension is supported
    if in_split[-1].lower() not in ['sdf', 'smi', 'ism', 'gz']:
        raise ValueError('File extension not supported (sdf, smi, ism, sdf.gz, smi.gz)')
    # Initialize a flag to detect if the file is compressed
    gzipped = False
    # If the file is compressed, set the flag to True, and check the extension of the compressed file
    if in_split[-1].lower() == 'gz':
        gzipped = True
        if in_split[-2].lower() not in ['sdf', 'smi', 'ism']:
            raise ValueError('File extension not supported (sdf, smi, ism, sdf.gz, smi.gz)')
    # Initialize a list of file handlers.
    file_handles = []
    # If the sentence_type is 'individual' or 'all', create files for each radius
    # write only files which contain corpus
    if (sentence_type == 'individual') or (sentence_type == 'all'):
        
        f1 = open(out_file+'_r0.corpus', "w")
        f2 = open(out_file+'_r1.corpus', "w")
        file_handles.append(f1)
        file_handles.append(f2)
    # If the sentence_type is 'alt' or 'all', create files for alternating sentences
    if (sentence_type == 'alt') or (sentence_type == 'all'):
        f3 = open(out_file, "w")
        file_handles.append(f3)
    # If the file is compressed, choose the appropriate handling method based on the file type
    if gzipped:
        import gzip
        if in_split[-2].lower() == 'sdf':
            mols_file = gzip.open(in_file, mode='r')
            suppl = Chem.ForwardSDMolSupplier(mols_file)
        else:
            mols_file = gzip.open(in_file, mode='rt')
            suppl = _read_smi(mols_file)
    else:
        # If the file is not compressed, choose the appropriate handling method based on the file type
        if in_split[-1].lower() == 'sdf':
            suppl = Chem.ForwardSDMolSupplier(in_file)
        else:
            mols_file = open(in_file, mode='rt')
            suppl = _read_smi(mols_file)
    # If the sentence_type is 'alt', parallel processing can be used
    if sentence_type == 'alt':  # This can run parallelized
        # Use the joblib library for parallel processing
        result = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_parallel_job)(mol, r) for mol in suppl)
        # Write the results to alternating sentence files
        for i, line in enumerate(result):
            f3.write(str(line) + '\n')
        print('% molecules successfully processed.')

    else:
        # If the sentence_type is not 'alt', process molecules one by one
        for mol in suppl:
            if mol is not None:
                # Invoke the new function to generate sentences
                single_ring_sentences = singlering2sentence(mol)
                combined_ring_sentences = combring2sentence(mol)
                all_ring_sentences = ring2alt_sentence(mol)

                # Write the generated sentences to a file
                for sentence in single_ring_sentences:
                    # Write single-ring sentences
                    f1.write(" ".join(map(str, sentence)) + '\n')
                for sentence in combined_ring_sentences:
                    # Write combined-ring sentences.
                    f2.write(" ".join(map(str, sentence)) + '\n')
                for sentence in all_ring_sentences:
                    # Write all ring sentences
                    f3.write(" ".join(map(str, sentence)) + '\n')

        # Close file handlers
        for fh in file_handles:
            fh.close()

def _read_corpus(file_name):
    while True:
        line = file_name.readline()
        if not line:
            break
        # Split each line into a list of words
        yield line.split()

# Define the insert_unk function to mark uncommon words in the corpus：
def insert_unk(corpus, out_corpus, threshold=3, uncommon='UNK'):
    """Handling of uncommon "words" (i.e. identifiers). It finds all least common identifiers (defined by threshold) and
    replaces them by 'uncommon' string.

    Parameters
    ----------
    corpus : str
        Input corpus file
    out_corpus : str
        Outfile corpus file
    threshold : int             # threshold：Determine the threshold for deciding which words are considered uncommon
        Number of identifier occurrences to consider it uncommon
    uncommon : str
        String to use to replace uncommon words/identifiers

    Returns
    -------
    """
    # Find least common identifiers in corpus
    # Open the corpus file and create an empty dictionary 'unique' to count the occurrences of each word
    f = open(corpus)
    unique = {}
    # Through a loop, count the occurrences of each word
    # If the word appears for the first time, add it to the dictionary with a count of 1; otherwise, increment the count by 1
    for i, x in tqdm(enumerate(_read_corpus(f)), desc='Counting identifiers in corpus'):
        for identifier in x:
            if identifier not in unique:
                unique[identifier] = 1
            else:
                unique[identifier] += 1
    # Record the total number of lines in the corpus
    n_lines = i + 1
    # Identify words that appear no more than 'threshold' times and place them in the set 'least_common
    least_common = set([x for x in unique if unique[x] <= threshold])
    # Close the corpus file
    f.close()
    # Reopen the corpus file and create a new file for writing the processed data
    f = open(corpus)
    fw = open(out_corpus, mode='w')
    # Read each line, replace uncommon words with the 'uncommon' tag, and then write the processed lines to the new file
    for line in tqdm(_read_corpus(f), total=n_lines, desc='Inserting %s' % uncommon):
        intersection = set(line) & least_common
        if len(intersection) > 0:
            new_line = []
            for item in line:
                if item in least_common:
                    new_line.append(uncommon)
                else:
                    new_line.append(item)
            fw.write(" ".join(new_line) + '\n')
        else:
            fw.write(" ".join(line) + '\n')
    # Close both files
    f.close()
    fw.close()

# Define the train_word2vec_model function for training the Word2Vec model：
def train_word2vec_model(infile_name, outfile_name=None, vector_size=100, window=10, min_count=3, n_jobs=1,
                         method='skip-gram', **kwargs):
    """Trains word2vec (Mol2vec, ProtVec) model on corpus file extracted from molecule/protein sequences.
    The corpus file is treated as LineSentence corpus (one sentence = one line, words separated by whitespaces)
    
    Parameters
    ----------
    infile_name : str The input file path used for training the model.
        Corpus file, e.g. proteins split in n-grams or compound identifier
    outfile_name : str  The path where the model is saved.
        Name of output file where word2vec model should be saved
    vector_size : int  The dimensionality of the vectors
        Number of dimensions of vector
    window : int  The window size during word vector training.
        Number of words considered as context
    min_count : int  Words with a frequency less than this value will be ignored.
        Number of occurrences a word should have to be considered in training
    n_jobs : int  The number of threads used during training.
        Number of cpu cores used for calculation
    method : str  The type of model，'skip-gram'or'cbow'。
        Method to use in model training. Options cbow and skip-gram, default: skip-gram)
    
    Returns
    -------
    word2vec.Word2Vec
    """
    # Select the model type based on the 'method' parameter
    if method.lower() == 'skip-gram':
        sg = 1
    elif method.lower() == 'cbow':
        sg = 0
    else:
        raise ValueError('skip-gram or cbow are only valid options')
    # Record the start time of model training
    start = timeit.default_timer()
    # Load the corpus and train the model using Word2Vec.
    corpus = word2vec.LineSentence(infile_name)
    # If an output file path is provided, save the model.
    model = word2vec.Word2Vec(corpus, vector_size=vector_size, window=window, min_count=min_count, workers=n_jobs, sg=sg,
                              **kwargs)
    if outfile_name:
        model.save(outfile_name)
    # Record and print the time taken for model training
    stop = timeit.default_timer()
    print('Runtime: ', round((stop - start)/60, 2), ' minutes')
    # Return the trained model.
    return model
    
# Define the remove_salts_solvents function to remove salts and solvents from chemical expressions    
def remove_salts_solvents(smiles, hac=3):
    """Remove solvents and ions have max 'hac' heavy atoms. This function removes any fragment in molecule that has
    number of heavy atoms <= "hac" and it might not be an actual solvent or salt
    
    Parameters
    ----------
    smiles : str
        SMILES
    hac : int  The threshold for the number of atoms
        Max number of heavy atoms

    Returns
    -------
    str
        smiles
    """
    # Parse the SMILES expression and remove parts with fewer atoms than 'hac'
    save = []
    for el in smiles.split("."):
        mol = Chem.MolFromSmiles(str(el))
        if mol.GetNumHeavyAtoms() <= hac:
            save.append(mol)
    return ".".join([Chem.MolToSmiles(x) for x in save])

def sentences2vec(sentences, model):
    sentence_vectors = []
    for sentence in sentences:
        sentence_vec = np.zeros(model.vector_size)  
        for word in sentence:
           
            if word in model.wv.key_to_index:
                word_vec = model.wv[word]
                sentence_vec += word_vec  
        
        if np.all(sentence_vec == 0):
            sentence_vec = np.zeros(model.vector_size)
        sentence_vectors.append(sentence_vec)
    return sentence_vectors

def featurize(in_file, out_file, model_path, uncommon=None):
    """
    Featurize mols in CSV.
    SMILES are regenerated with RDKit to get canonical SMILES.

    Parameters
    ----------
    in_file : str
        Input CSV file with SMILES and other molecular parameters.
    out_file : str
        Output csv file.
    model_path : str
        File path to pre-trained Gensim word2vec model.
    uncommon : str
        String to used to replace uncommon words/identifiers.
    """

    # Load the Word2Vec model.
    word2vec_model = word2vec.Word2Vec.load(model_path)

    # Load the CSV file.
    print('Loading molecules from CSV.')
    df = pd.read_csv(in_file)
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES')
    df = df[df['ROMol'].notnull()]
    df['SMILES'] = df['ROMol'].map(Chem.MolToSmiles)

    # Featurize molecules.
    print('Featurizing molecules.')
    df['mol-sentence'] = df.apply(lambda x: MolSentence(ring2alt_sentence(x['ROMol'])), axis=1)
    vectors = sentences2vec(df['mol-sentence'], word2vec_model, unseen=uncommon)

    # Merge features with original data and save.
    df_vec = pd.DataFrame(vectors, columns=['ring2vec-%03i' % x for x in range(vectors.shape[1])])
    df = pd.concat([df.drop(['ROMol', 'mol-sentence'], axis=1), df_vec], axis=1)
    df.to_csv(out_file, index=False)

    print('Featurization completed.')

