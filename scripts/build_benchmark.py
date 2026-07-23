import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

rows = [
 ("UCE zero-shot - authors' TS map (kNN)",        0.838, 0.708, 'UCE'),
 ("UCE zero-shot - nearest-centroid",              0.795, 0.673, 'UCE'),
 ("SingleR-style corr - same TS ref",              0.904, 0.817, 'conventional'),
 ("Expression logistic-reg - same TS ref",         0.697, 0.518, 'conventional'),
 ("CellTypist (off-the-shelf human immune)",       0.606, 0.392, 'conventional'),
 ("Literature markers (score, all 12 types)",      0.830, 0.726, 'marker'),
]
df = pd.DataFrame(rows, columns=['method','accuracy','macroF1','family'])
df.to_csv('eval_out/benchmark_table.csv', index=False)

colors = {'UCE':'#2b6cb0','conventional':'#e08a1e','marker':'#3a9d5d'}
o = df.sort_values('accuracy').reset_index(drop=True)
y = np.arange(len(o))
fig, ax = plt.subplots(figsize=(10.5,5.6))
ax.barh(y, o['accuracy'], color=[colors[f] for f in o['family']])
ax.set_yticks(y); ax.set_yticklabels(o['method'], fontsize=9)
ax.set_xlim(0,1.02); ax.set_xlabel('Accuracy (dog cells vs author labels)')
for i,(a,f1) in enumerate(zip(o['accuracy'],o['macroF1'])):
    ax.text(a+0.008, i, f'{a:.2f}  (F1 {f1:.2f})', va='center', fontsize=8.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=colors['UCE'],label='UCE (protein embedding, no orthologs needed)'),
                   Patch(color=colors['conventional'],label='Conventional expression (needs ortholog map)'),
                   Patch(color=colors['marker'],label='Literature markers')],
          loc='lower right', fontsize=8)
ax.set_title('Dog leukocyte annotation - UCE vs conventional baselines')
plt.tight_layout(); plt.savefig('eval_out/benchmark_comparison.png', dpi=150); plt.close()
print(df.to_string(index=False))
print('saved eval_out/benchmark_comparison.png + benchmark_table.csv')
