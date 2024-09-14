"""
Visualization - Mostly plotting functions
===================================
"""

from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Tuple, Optional, Union

# Electronegativity table for different elements
electronegativity_table = {
    6: 2.55,  # Carbon
    7: 3.04,  # Nitrogen
    8: 3.44,  # Oxygen
    9: 3.98,  # Fluorine
    15: 2.19, # Phosphorus
    16: 2.58, # Sulfur
    17: 3.16, # Chlorine
    35: 2.96, # Bromine
    53: 2.66, # Iodine
    34: 2.55, # Selenium
    # Other elements can be added here as needed
}


def _prepare_mol(mol: Chem.Mol, kekulize: bool) -> Chem.Mol:
    """
    Prepare molecule for SVG depiction by embedding 2D coordinates and optionally kekulizing.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    kekulize : bool
        Whether to kekulize the molecule.
    
    Returns
    -------
    mc : rdkit.Chem.rdchem.Mol
        Processed molecule object with 2D coordinates.
    """
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except Chem.KekulizeException:
            mc = Chem.Mol(mol.ToBinary())
    
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    
    return mc

def mol_to_svg(mol: Chem.Mol, mol_size: Tuple[int, int] = (300, 300), kekulize: bool = True, 
               drawer: Optional[rdMolDraw2D.MolDraw2D] = None, font_size: float = 0.8, **kwargs) -> str:
    """
    Generates an SVG from the given molecule structure.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    mol_size : tuple of int, optional
        Size of the molecule image, default is (300, 300).
    kekulize : bool, optional
        Whether to kekulize the molecule, default is True.
    drawer : rdMolDraw2D.MolDraw2D, optional
        Drawer to use for generating the SVG, default is rdMolDraw2D.MolDraw2DSVG.
    font_size : float, optional
        Font size for atom labels, default is 0.8.
    
    Returns
    -------
    svg : str
        SVG string representation of the molecule.
    """
    from IPython.display import SVG
    
    mc = _prepare_mol(mol, kekulize)
    mol_atoms = [atom.GetIdx() for atom in mc.GetAtoms()]
    
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(*mol_size)
    
    drawer.SetFontSize(font_size)
    drawer.DrawMolecule(mc, highlightAtomRadii={x: 0.5 for x in mol_atoms}, **kwargs)
    drawer.FinishDrawing()
    
    svg = drawer.GetDrawingText()
    return SVG(svg.replace('svg:', '')).data

def depict_rings(mol: Chem.Mol, mol_size: Tuple[int, int] = (300, 300), **kwargs) -> str:
    """
    Highlight and depict all rings in a molecule.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    mol_size : tuple of int, optional
        Size of the molecule image, default is (300, 300).
    
    Returns
    -------
    svg : str
        SVG string representation with rings highlighted.
    """
    ring_info = mol.GetRingInfo()
    ring_atoms = [atom for ring in ring_info.AtomRings() for atom in ring]
    
    return mol_to_svg(mol, mol_size=mol_size, highlightAtoms=ring_atoms, **kwargs)

def some_condition_to_identify_ring(ring: List[int], identifier: List[Union[int, float]], mol: Chem.Mol) -> bool:
    """
    Check whether a ring in the molecule matches certain criteria.
    
    Parameters
    ----------
    ring : list of int
        List of atom indices representing the ring.
    identifier : list of int or float
        List of expected features corresponding to the ring.
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    
    Returns
    -------
    bool
        True if the ring matches the expected identifier, False otherwise.
    """
    expected_ring_size = identifier[0]
    expected_aromatic = identifier[1]
    expected_double_bonds = identifier[2]
    expected_heteroatoms = identifier[3]
    expected_atom_types = identifier[4:4 + expected_ring_size]
    expected_heteroatoms_electroneg = identifier[4 + expected_ring_size:]
    
    if len(ring) != expected_ring_size:
        return False
    
    if any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring) != expected_aromatic:
        return False
    
    double_bonds_count = sum(1 for idx in ring for nbr in mol.GetAtomWithIdx(idx).GetNeighbors() 
                             if mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE)
    if double_bonds_count != expected_double_bonds:
        return False
    
    heteroatoms_count = sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6)
    if heteroatoms_count != expected_heteroatoms:
        return False
    
    atom_types_in_ring = [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in ring]
    if atom_types_in_ring != expected_atom_types:
        return False
    
    heteroatoms_electroneg_in_ring = [
        electronegativity_table.get(mol.GetAtomWithIdx(idx).GetAtomicNum()) 
        for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6
    ]
    if heteroatoms_electroneg_in_ring != expected_heteroatoms_electroneg:
        return False

    return True

def depict_identifier(mol: Chem.Mol, identifier: List[Union[int, float]], mol_size: Tuple[int, int] = (300, 300), **kwargs) -> str:
    """
    Depict a specific ring feature identifier in a molecule.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    identifier : list of int or float
        Identifier corresponding to a specific ring feature.
    mol_size : tuple of int, optional
        Size of the molecule image, default is (300, 300).
    
    Returns
    -------
    svg : str
        SVG string representation with the identifier highlighted.
    """
    ring_info = mol.GetRingInfo()
    ring_atoms = ring_info.AtomRings()
    atoms_to_highlight = []
    
    for ring in ring_atoms:
        if some_condition_to_identify_ring(ring, identifier, mol):
            atoms_to_highlight.extend(ring)

    return mol_to_svg(mol, mol_size=mol_size, highlightAtoms=atoms_to_highlight, **kwargs)


def plot_class_distribution(df: pd.DataFrame, x_col: str, y_col: str, c_col: str, ratio: float = 0.1, n: int = 1, 
                            marker: str = 'o', alpha: float = 1.0, x_label: str = 'auto', y_label: str = 'auto', 
                            cmap=plt.cm.viridis, size: Tuple[int, int] = (8, 8), share_axes: bool = False):
    """
    Scatter + histogram plots of x and y values with color based on classes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.
    x_col : str
        Column name for x values.
    y_col : str
        Column name for y values.
    c_col : str
        Column name for classes (used as color basis).
    ratio : float, optional
        Ratio to determine empty space limits for the x/y axis, default is 0.1.
    marker : str, optional
        Marker style for scatter plot, default is 'o'.
    alpha : float, optional
        Alpha value for scatter plot, default is 1.0.
    x_label : str, optional
        Label for the x-axis, default is 'auto' (same as x_col).
    y_label : str, optional
        Label for the y-axis, default is 'auto' (same as y_col).
    cmap : matplotlib.colors.ListedColormap, optional
        Color map to use for the plot, default is plt.cm.viridis.
    size : tuple of int, optional
        Size of the output figure, default is (8, 8).
    share_axes : bool, optional
        Whether to share axes, default is False.
    """
    
    if y_label == 'auto':
        y_label = y_col
    if x_label == 'auto':
        x_label = x_col    
    
    fig, ((h1, xx), (sc, h2)) = plt.subplots(2, 2, squeeze=True, sharex=share_axes, sharey=share_axes, figsize=size,
                                             gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1, 3]})
    
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    xx.axis('off')
    
    ratio_xaxis = (max(df[x_col]) - min(df[x_col])) * ratio
    ratio_yaxis = (max(df[y_col]) - min(df[y_col])) * ratio
               
    x_max = max(df[x_col]) + ratio_xaxis
    x_min = min(df[x_col]) - ratio_xaxis

    y_max = max(df[y_col]) + ratio_yaxis
    y_min = min(df[y_col]) - ratio_yaxis
    
    h1.set_xlim(x_min, x_max)
    h1.xaxis.set_visible(False)
    sc.set_xlim(x_min, x_max)
    sc.set_xlabel(x_label)
    
    h2.set_ylim(y_min, y_max)
    h2.yaxis.set_visible(False)
    sc.set_ylim(y_min, y_max)
    sc.set_ylabel(y_label)
    
    class_unique = np.sort(df[c_col].unique())
    h, bins = np.histogram(range(len(cmap.colors)), bins=len(class_unique))
    colors = [cmap.colors[int(x)] for x in bins[1:]]
    
    for cl, color in zip(class_unique, colors):
        if len(df[df[c_col] == cl]) > 1:
            sns.kdeplot(df[df[c_col] == cl][x_col], ax=h1, color=color, label=cl, legend=False)  # x-axis histogram
            sns.kdeplot(df[df[c_col] == cl][y_col], ax=h2, color=color, vertical=True, label=cl, legend=False)  # y-axis histogram
        handles, labels = h1.get_legend_handles_labels()
        h1.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=n)
        sc.scatter(df[df[c_col] == cl][x_col], df[df[c_col] == cl][y_col], c=color, marker=marker, alpha=alpha)
       
    return fig


def plot_2D_vectors(vectors: List[List[float]], sumup: bool = True, min_max_x: Optional[Tuple[float, float]] = None, 
                    min_max_y: Optional[Tuple[float, float]] = None, cmap=plt.cm.viridis_r, colors: Optional[List[str]] = None, 
                    vector_labels: Optional[List[str]] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot 2D vectors sequentially and transpose them.
    
    Parameters
    ----------
    vectors : list of list of float
        2D vectors, e.g. [[0, 1], [3, 4]].
    sumup : bool, optional
        Whether to show a vector representing the sum of vectors, default is True.
    min_max_x : tuple of float, optional
        Minimum and maximum values for the x-axis, default is None.
    min_max_y : tuple of float, optional
        Minimum and maximum values for the y-axis, default is None.
    cmap : matplotlib.colors.ListedColormap, optional
        Colormap to use for the vectors, default is plt.cm.viridis_r.
    colors : list of str, optional
        Colors for the vectors, default is None (chooses from colormap).
    vector_labels : list of str, optional
        Labels for the vectors, default is None.
    ax : plt.Axes, optional
        Axis object to plot on, default is None (creates a new axis).
    
    Returns
    -------
    ax : plt.Axes
        Axis with the plotted vectors.
    """
    soa = []  # Vectors with start and end points
    for vec in vectors:
        if len(soa) == 0:
            soa.append([0, 0] + vec)
        else:
            last = soa[-1]
            soa.append([last[0] + last[2], last[1] + last[3]] + vec)
    
    if sumup:
        soa.append([0, 0] + list(np.sum(vectors, axis=0)))
    
    X, Y, U, V = zip(*soa)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    if colors is None and sumup:
        colors = [cmap.colors[120]] * (len(soa) - 1) + [cmap.colors[-1]]
    elif colors is None and not sumup:
        colors = [cmap.colors[120]] * len(soa)
    
    if vector_labels:
        if len(vector_labels) != len(vectors) + int(sumup):
            raise ValueError("Number of vector labels does not match the number of vectors.")
        
        for x, y, u, v, c, vl in zip(X, Y, U, V, colors, vector_labels):
            q = ax.quiver(x, y, u, v, color=c, angles='xy', scale_units='xy', scale=1)
            ax.quiverkey(q, x, y, u, vl, coordinates='data', color=[0, 0, 0, 0], labelpos='N')
    else:
        ax.quiver(X, Y, U, V, color=colors, angles='xy', scale_units='xy', scale=1)
   
    # Set plot limits
    if min_max_x is None:
        min_max_x = (min(X), max([x + u for x, u in zip(X, U)]))
    if min_max_y is None:
        min_max_y = (min(Y), max([y + v for y, v in zip(Y, V)]))
    
    ax.set_xlim(min_max_x[0], min_max_x[1])
    ax.set_ylim(min_max_y[0], min_max_y[1])
    
    return ax

class IdentifierTable:
    """
    Class for displaying molecule identifiers in tabular format with corresponding depictions.
    """
    def __init__(self, identifiers: List[Union[int, str]], mols: List[Chem.Mol], sentences: List[str], 
                 cols: int, radius: int, size: Tuple[int, int] = (150, 150)):
        self.mols = mols
        self.sentences = sentences
        self.identifiers = identifiers
        self.cols = cols
        self.radius = radius
        self.size = size
        self.depictions = []
        self._get_depictions()

    def _get_depictions(self):
        """Generate depictions for the first molecule containing each identifier."""
        for idx in self.identifiers:
            for mol, sentence in zip(self.mols, self.sentences):
                if idx in sentence:
                    self.depictions.append(depict_identifier(mol, idx, molSize=self.size).data)
                    break

    def _repr_html_(self) -> str:
        """Generate HTML representation of the table."""
        table = '<table style="width:100%">'
        c = 1
        
        for depict, idx in zip(self.depictions, self.identifiers):
            if c == 1:
                table += '<tr>'
            
            table += f'<td><div align="center">{depict}</div>\n<div align="center">{idx}</div></td>'
            
            if c == self.cols:
                table += '</tr>'
                c = 0
            c += 1
        table += '</table>'
        
        return table
