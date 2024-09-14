import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import List

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_smiles_and_save(input_csv: str, output_csv: str) -> None:
    """
    Cleans SMILES strings by converting them to RDKit molecule objects, removing invalid entries,
    and saving the cleaned data to a new CSV file.

    Parameters:
    -----------
    input_csv : str
        Path to the input CSV file containing SMILES strings.
    output_csv : str
        Path to the output CSV file where cleaned data will be saved.

    Returns:
    --------
    None
    """
    try:
        # Load data from the input CSV file
        df = pd.read_csv(input_csv)
        
        # Check if 'SMILES' column exists in the dataframe
        if 'SMILES' not in df.columns:
            raise ValueError("The input CSV must contain a 'SMILES' column.")
        
        # Convert SMILES strings to RDKit molecule objects
        df['Mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
        
        # Remove rows with invalid SMILES (i.e., where the molecule conversion failed)
        df_valid = df[df['Mol'].notnull()].copy()

        # Log the number of invalid SMILES that were removed
        removed_count = len(df) - len(df_valid)
        if removed_count > 0:
            print(f"Removed {removed_count} invalid SMILES strings.")

        # Drop the 'Mol' column before saving the cleaned data
        df_valid.drop(columns=['Mol'], inplace=True)
        
        # Save the cleaned data to a new CSV file
        df_valid.to_csv(output_csv, index=False)
        print(f"The cleaned data has been saved to {output_csv}")
    
    except FileNotFoundError:
        print(f"File '{input_csv}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
# Function to preprocess features (string to float list)
def preprocess_features(feature_str: str) -> List[float]:
    """Convert string representation of features to a list of floats."""
    try:
        return [float(i) for i in feature_str.replace('[', '').replace(']', '').split(',')]
    except ValueError as e:
        logger.error(f"Error in processing features: {feature_str}. Error: {e}")
        return []

# Function to check and pad feature lists
def check_and_pad_feature_list(feature_list: List[float], target_length: int = 110) -> List[float]:
    """Pad feature list with zeros to the target length or truncate if it exceeds."""
    if len(feature_list) < target_length:
        return feature_list + [0.0] * (target_length - len(feature_list))
    elif len(feature_list) > target_length:
        return feature_list[:target_length]
    return feature_list

# Expand features based on target dimension
def expand_features(feature: str, target_dim: int = 2048) -> List[float]:
    """Expand the feature list to a target dimension."""
    return [float(i) for i in feature.replace('[', '').replace(']', '').split(',')] + [0] * (target_dim - len(feature.split(',')))

# Self-Attention 
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size).to(device)

    def forward(self, values, keys, queries, feature_dim, mask=None):
        N = queries.shape[0]

        # Reshape values, keys, queries
        values = values.reshape(N, -1, self.heads, self.head_dim).to(device)
        keys = keys.reshape(N, -1, self.heads, self.head_dim).to(device)
        queries = queries.reshape(N, -1, self.heads, self.head_dim).to(device)

        # Apply custom logic to modify the query features
        queries[:, -feature_dim:, :, :] *= 2

        # Calculate energy and attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            mask = mask.to(device)
            energy = energy.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, -1, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

# Feature Fusion Model
class FeatureFusionModel(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(FeatureFusionModel, self).__init__()
        self.self_attention = SelfAttention(feature_dim * 3, num_heads).to(device)
        self.fc = nn.Linear(feature_dim * 3, feature_dim * 3).to(device)

    def forward(self, feature1, feature2, feature3):
        # Concatenate the features
        combined_features = torch.cat((feature1, feature2, feature3), dim=1).to(device)
        attention_output = self.self_attention(combined_features, combined_features, combined_features, feature_dim=2048)
        fused_features = torch.mean(attention_output, dim=1)
        return self.fc(fused_features)

# Load and preprocess data
def load_and_preprocess_data(filepath: str, feature_dim: int) -> pd.DataFrame:
    """Load CSV, preprocess features, and pad them to the target length."""
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1')

    logger.info(f"Data loaded, number of rows: {len(df)}")

    # Preprocess and expand features
    df['Features1'] = df['Features1'].apply(lambda x: expand_features(x, target_dim=feature_dim))
    df['Features2'] = df['Features2'].apply(lambda x: expand_features(x, target_dim=feature_dim))
    df['Features3'] = df['Features3'].apply(lambda x: expand_features(x, target_dim=feature_dim))

    return df

# Main function for feature fusion
def fuse_features(df: pd.DataFrame, feature_dim: int, num_heads: int) -> pd.DataFrame:
    """Fuse features using the self-attention mechanism."""
    model = FeatureFusionModel(feature_dim, num_heads)

    # Convert to PyTorch tensors
    feature1_tensor = torch.tensor(df['Features1'].tolist(), dtype=torch.float32).to(device)
    feature2_tensor = torch.tensor(df['Features2'].tolist(), dtype=torch.float32).to(device)
    feature3_tensor = torch.tensor(df['Features3'].tolist(), dtype=torch.float32).to(device)

    # Fuse features
    fused_features = model(feature1_tensor, feature2_tensor, feature3_tensor)
    fused_features_detached = fused_features.detach().cpu().numpy()

    # Convert to list and store in DataFrame
    df['Fused_Features'] = [','.join(map(str, f)) for f in fused_features_detached]

    return df

# Save the processed DataFrame to a CSV file
def save_dataframe(df: pd.DataFrame, filepath: str):
    """Save the DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")


