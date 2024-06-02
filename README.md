# **A Ring2Vec description method enables automatic feature extraction and highly accurate predictions of material properties in organic solar cells.**

### Author list
![table1](https://github.com/leylacheung/a-novel-molecular-representation-for-NFAs/assets/161421118/5b9d5a58-65e7-494f-bdb6-cb99bf94b97b)


## Introduction

Ring2vec aims to generate a compressive representation for NFAs by leveraging deep learning and machine learning methods. 

Our molecular description framework consists of three sub-modules, namely, the Fin2Vec, MG2Vec, and Ring2Vec modules. The Fin2Vec and MG2Vec modules are based on algorithms similar to the previously reported Fingerprint and MG methods. Our innovation mainly lies in the Ring2Vec module, which, when combined with the Fin2Vec and MG2Vec, can yield significantly improved prediction performance. 

![overview](https://github.com/leylacheung/a-novel-molecular-representation-for-NFAs/assets/161421118/4cc28dd0-5ca5-476e-ba55-fec1e6674f28)
This is overview of our framework! (a) Ring2Vec model. It starts by preprocessing the SMILES strings of molecules, extracting ring-based subunits with defined features, either in the form of single rings (radius 0) or combined double rings (radius 1). These ring-based representations are subsequently trained by a similar algorithm to Word2Vec, generating thousands of embeddings, similar to a vocabulary. (b) MG2Vec module. The molecule is processed through a GNN model to generate embeddings that effectively capture atom-level information. (c) we also use the Fin2Vec model to generate embeddings that represent fingerprint data. Lastly, by integrating the embeddings from Ring2Vec, MG2Vec, and Fin2Vec modules using the commonly used self-attention mechanism, we create a comprehensive representation of NFA molecules.


In addition, we have also made a Figure to illustrate the overall process of our method. we hope that it is useful to help readers to understand our method.

![图片 1](https://github.com/leylacheung/a-novel-molecular-representation-for-NFAs/assets/161421118/6c973d68-567a-4760-8b15-f45f5840cc7c)

## Abstract
Predicting the properties of non-fullerene acceptors (NFAs), complex organic molecules used in organic solar cells (OSCs), poses a significant challenge. Some existing approaches primarily focus on atom-level information and may overlook high-level molecular features, including the subunits of NFAs. While other methods that effectively represent subunit information show improved prediction performance, they require labor-intensive data labeling and lack broad applicability. In this paper, we introduce an efficient molecular description method that extracts molecular information at both the atom and subunit levels. Importantly, our method ensures a comprehensive molecular representation while also automating feature extraction, rendering both effectiveness and efficiency. Inspired by Word2Vec algorithms in natural language processing (NLP), our Ring2Vec method treats the "rings" in organic molecules as analogous to "words" in sentences. Using our method, we achieve ultra-fast and remarkably accurate predictions of the energy levels of NFA molecules, with a minimal prediction error of merely 0.06 eV in predicting energy levels, markedly surpassing the performance of conventional computational chemistry methods. Furthermore, our method can potentially have broad applicability across various domains of molecular description and property prediction, owing to the versatility and efficiency of the Ring2Vec model.

### Dataset
1, CEPDB: This extensive repository contains data on over 2.3 million organic semiconductor molecules, each representing a potential candidate for photovoltaic applications. It offers a wealth of quantum-mechanically derived parameters crucial for assessing the performance of organic solar cells. These parameters comprise molecular geometry, power conversion efficiency (PCE), HOMO/LUMO energy levels, band gaps, open-circuit voltage (Voc), and short-circuit current density (Jsc). A noteworthy characteristic of these molecules is their composition, which is based on 26 fundamental building blocks. These building blocks, characterized by their ring-based nature, align perfectly with our model's focus on capturing ring-related information within molecules. The substantial size of the CEPDB proves highly advantageous for our pretraining purposes, as such procedures typically demand extensive datasets. 

2, small dataset from the published paper titled "Machine Learning-Assisted Development of Organic Solar Cell Materials: Issues, Analyses, and Outlooks." 

## Results
Results have been released!👏
![图片 2](https://github.com/leylacheung/a-novel-molecular-representation-for-NFAs/assets/161421118/cc976805-ee5e-4856-bb78-5df3c14be644)

![figure2](https://github.com/leylacheung/a-novel-molecular-representation-for-NFAs/assets/161421118/6cbec69c-dbb9-4808-8f7c-a7f428d5ac97)

## Usage
We will soon release all the codes, data for Ring2vec. Please stay tuned! Thanks for your patience!
