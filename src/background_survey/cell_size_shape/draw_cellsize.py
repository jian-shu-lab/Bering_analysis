import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

df_cells = pd.read_csv('/data/aronow/Kang/spatial/Bering/cosmx/data/cell_metadata_rna.txt',sep='\t',header=0,index_col=0)
print(df_cells.head())
print(df_cells.columns)

print(df_cells['Slide_name'].value_counts())
print(df_cells['tissue'].value_counts())

lung = 'Lung9'
df_cells = df_cells[df_cells['Slide_name'] == lung].copy()

df_mean = pd.DataFrame(df_cells.groupby('cell_type').mean()[['Area','AspectRatio']])
df_mean = df_mean.sort_values(by='Area',ascending=True)
df_std = pd.DataFrame(df_cells.groupby('cell_type').std()[['Area','AspectRatio']])
df_std = df_std.loc[df_mean.index,:]

# ### error bar plot
# fig, axes = plt.subplots(1,2,figsize=(10,5))
# axes[0].errorbar(df_mean.index,df_mean['Area'],yerr=df_std['Area'],fmt='o')
# axes[0].set_ylabel('Area')
# axes[0].set_xlabel('Cell Type')
# axes[0].set_xticklabels(df_mean.index,rotation=90)
# axes[0].set_ylim(2000,4500)
# axes[1].errorbar(df_mean.index,df_mean['AspectRatio'],yerr=df_std['AspectRatio'],fmt='o')
# axes[1].set_ylabel('Aspect Ratio')
# axes[1].set_xlabel('Cell Type')
# axes[1].set_xticklabels(df_mean.index,rotation=90)
# # axes[1].set_ylim(0.5,1.5)
# plt.tight_layout()
# plt.savefig(f'cosmx_lung_tumor_cellsize_{lung}.pdf',dpi=300)

### violin plot for area column for each cell type
fig, ax = plt.subplots(1,1,figsize=(10,5))
sns.violinplot(x='cell_type',y='Area',data=df_cells,ax=ax, order = df_mean.index)
ax.set_ylabel('Area')
ax.set_xlabel('Cell Type')
ax.set_xticklabels(df_mean.index,rotation=90)
ax.set_ylim(0,6000)
plt.savefig(f'cosmx_lung_tumor_cellsize_violin_{lung}_Area.pdf',dpi=300)

fig, ax = plt.subplots(1,1,figsize=(10,5))
sns.violinplot(x='cell_type',y='AspectRatio',data=df_cells,ax=ax, order = df_mean.index)
ax.set_ylabel('Aspect Ratio')
ax.set_xlabel('Cell Type')
ax.set_xticklabels(df_mean.index,rotation=90)
ax.set_ylim(0.5,1.75)
plt.tight_layout()
plt.savefig(f'cosmx_lung_tumor_cellsize_violin_{lung}_AspectRatio.pdf',dpi=300)

# # swarm plot ### too slow
# fig, axes = plt.subplots(1,2,figsize=(16,5))
# sns.swarmplot(x='cell_type',y='Area',data=df_cells,ax=axes[0])
# axes[0].set_ylabel('Area')
# axes[0].set_xlabel('Cell Type')
# axes[0].set_xticklabels(df_mean.index,rotation=90)
# axes[0].set_ylim(0,8000)
# sns.swarmplot(x='cell_type',y='AspectRatio',data=df_cells,ax=axes[1])
# axes[1].set_ylabel('Aspect Ratio')
# axes[1].set_xlabel('Cell Type')
# axes[1].set_xticklabels(df_mean.index,rotation=90)
# axes[1].set_ylim(0,2)
# plt.tight_layout()
# plt.savefig(f'cosmx_lung_tumor_cellsize_swarm_{lung}.png',dpi=300)
