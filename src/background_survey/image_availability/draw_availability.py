import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
# Generate some random data
data = np.random.rand(10, 10)

# Define the categories for the annotations
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# Create a color palette for the annotations
colors = sns.color_palette('husl', n_colors=len(categories))

# Create a DataFrame with the annotations
annotations = pd.DataFrame(categories, columns=['Category'])

# Create the heatmap
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data, cmap='YlGnBu', ax=ax, cbar_kws={'orientation': 'vertical', 'label': 'Value'}, 
            annot_kws={'size': 12}, annot=True, fmt='.2f', 
            cbar=True, xticklabels=False, yticklabels=False, square=True)

# Add the annotations as a sidebar
sns.heatmap(np.array([annotations.index]), cmap=colors, ax=ax, 
            cbar_kws={'orientation': 'vertical', 'ticks': np.arange(len(categories))+0.5, 
                      'label': 'Category'}, 
            yticklabels=annotations['Category'], xticklabels=False, square=True, annot=False)
plt.savefig('test.pdf', dpi=300, bbox_inches='tight')
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'Arial'

# df_table = pd.read_excel('TableSX.image_avavilability_across_datasets.xlsx', sheet_name='Sheet2', keep_default_na=False)
# df_table = df_table[df_table['Technology']!=''].copy()
# df_table = df_table.loc[:, ['Technology', 'Dataset name', 'Nuclei', 'Total mRNA', 'Membrane']]

# df_tech = df_table[['Technology', 'Dataset name']]
# df_tech = df_tech.set_index('Dataset name')

# df_table = df_table.drop(['Technology'], axis=1)
# df_table = df_table.set_index('Dataset name')

# print(df_table.head())
# print(df_tech.head())

# fig, axes = plt.subplots(figsize=(4, 8), nrows=1, ncols=2, gridspec_kw={'width_ratios': [0.1, 1]})

# # draw index column dataset as another ax
# avail_tech = df_tech['Technology'].unique()
# tech_dict = dict(zip(avail_tech, range(len(avail_tech))))
# df_tech['Technology_id'] = df_tech['Technology'].map(tech_dict)
# df_tech = df_tech.drop(['Technology'], axis=1)
# # add box border
# axes[0] = sns.heatmap(df_tech, cmap='coolwarm', annot=False,ax = axes[0], cbar=False, linewidths=1, linecolor='black')
# axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, fontsize=12)

# axes[1] = sns.heatmap(df_table, annot=False, fmt='g', cmap='Blues', yticklabels = False, ax=axes[1], cbar=False, linewidths=1, linecolor='black')
# axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, fontsize=12)
# axes[1].set_title('Image availability across datasets', fontsize=16)
# axes[1].set_xlabel('Image type', fontsize=14)
# axes[1].yticks = None
# # remove y axis titile
# axes[1].set_ylabel('')

# fig.savefig('image_availability_v3.pdf', dpi=300, bbox_inches='tight')


# draw z stacks
df_table = pd.read_excel('TableSX.image_avavilability_across_datasets.xlsx', sheet_name='Sheet2', keep_default_na=False)
df_table = df_table.iloc[:, :10]
df_table = df_table[df_table['z_depth'] != ''].copy()

datasets = df_table['Dataset name'].values
technologies = df_table['Technology'].values
z_depths = df_table['z_depth'].values
z_depths = [float(i.split(' um')[0]) for i in z_depths]

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(np.arange(len(z_depths)), z_depths, color='steelblue', edgecolor='black', linewidth=1)
ax.set_ylabel('Z depth (um)')
ax.set_xlabel('Dataset')
ax.set_xticks(np.arange(len(z_depths)))
ax.set_xticklabels(datasets, rotation=45, fontsize=12, ha='right')

fig.savefig('z_depth.pdf', dpi=300, bbox_inches='tight')