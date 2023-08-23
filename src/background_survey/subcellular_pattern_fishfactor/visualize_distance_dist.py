import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# ridgeplot
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

df_distance = pd.read_csv('distance.csv')
print(df_distance.head())

cell_maxdist = pd.DataFrame(df_distance.groupby('cell')['distance'].max())
df_distance['max_distance'] = df_distance['cell'].map(cell_maxdist['distance'])
df_distance['distance_norm'] = df_distance['distance'] / df_distance['max_distance']
gene_to_col = {'MALAT1': '#F4A460', 'NEAT1': '#D2B48C', 'MZT2A': '#A0522D', 'OLFM4': '#5F9EA0', 'DUSP5': '#4682B4', 'S100A6': '#00BFFF'}

# draw norm dist for each gene
# fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
fig, ax = plt.subplots(1, 1, figsize=(3, 4), sharex=True, sharey=True)
genes = np.unique(df_distance['gene'])
for gene in gene_to_col.keys():
    df_gene = df_distance[df_distance['gene'] == gene]
    # ax.hist(df_gene['distance_norm'], bins=50, alpha=0.5, color = gene_to_col[gene], label = gene)
    sns.kdeplot(df_gene['distance_norm'], shade=True, label = gene, ax = ax, color = gene_to_col[gene])
ax.legend()
ax.set_xlabel('Normalized distance', fontsize=12)
ax.set_yticks([])

# fig.savefig('figures/subcellular_distribution/_distance_norm_hist.pdf', dpi=300, bbox_inches='tight')
fig.savefig('figures/subcellular_distribution/_distance_norm_ridge.pdf', dpi=300, bbox_inches='tight')