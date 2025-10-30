# ALS Motor Neuron Expression x iPSC Microglia Secreted Ligand Analysis

Evidence of motor neuron gene expression was based on an external data [GSE76220](https://pubmed.ncbi.nlm.nih.gov/29881994/):
* 20 samples from 13 ALS (9 males, 4 females) and 7 CTL (5 males, 2 females)
* fastq files downloaded for CTL-3 (GSM1977029) and CTL-8 (GSM1977034) were identical, so the latter sample CTL-8 (GSM1977034) was dropped in the analysis
* Laser capture microdissection-enriched surviving lumbar spinal cords motor neurons (MNs) directly sampled from Healthy and ALS post-mortem tissues

It was the deepest sequenced data of this type from this [meta-analysis](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2024.1385114/full).

Download Series RNA-seq normalized counts matrix (GSE76220_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz).

# ligand receptor reference - CellChatDB
interaction_input_CellChatDB.csv from [Download link](https://figshare.com/articles/dataset/human_CellChatDB/20469720).

### Set up
```
bash setup.sh
```

### Analysis
All scripts are in `notebooks/`.