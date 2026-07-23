import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family'] = 'DejaVu Sans'

NAVY='#14213D'; TEAL='#1C7293'; CORAL='#E07A5F'; GOLD='#E9A03B'; LIGHT='#EEF3F6'; GREY='#3A4A5A'; INK='#20303F'

def box(ax,x,y,w,h,label,fc,tc='white',fs=11,bold=True,ec=None,sub=None):
    b=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=fc,ec=ec or fc,lw=1.5,mutation_aspect=1)
    ax.add_patch(b)
    ax.text(x+w/2,y+h/2+(0.06 if sub else 0),label,ha='center',va='center',
            fontsize=fs,color=tc,weight='bold' if bold else 'normal',wrap=True)
    if sub:
        ax.text(x+w/2,y+h/2-0.16,sub,ha='center',va='center',fontsize=fs-2.5,color=tc)

def arrow(ax,x1,y1,x2,y2,c=GREY,lw=2.2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=18,
                 lw=lw,color=c,shrinkA=2,shrinkB=2))

# ---------------- MODEL SCHEMATIC ----------------
fig,ax=plt.subplots(figsize=(13.0,5.6)); ax.set_xlim(0,13.0); ax.set_ylim(0,5.6); ax.axis('off')
fig.patch.set_facecolor('white')
# inputs
box(ax,0.2,3.55,2.5,1.15,'Raw scRNA-seq',TEAL,fs=12.5,sub='counts of one cell (any species)')
box(ax,0.2,0.7,2.5,1.15,'Genome',NAVY,fs=12.5,sub='protein sequence per gene')
# ESM2
box(ax,3.15,0.7,2.05,1.15,'ESM-2 (15B)',GOLD,tc=INK,fs=12.5,sub='frozen protein LM')
# protein embeddings
box(ax,5.6,0.7,2.15,1.15,'Protein\nembeddings',NAVY,fs=11.5,sub='5120-d per gene')
# token assembly
box(ax,3.15,2.45,4.6,1.0,'Gene tokens  +  [CLS]',TEAL,fs=12.5,sub='expression-sampled · chromosome-ordered')
# transformer
box(ax,8.35,1.55,4.35,2.05,'Transformer encoder',NAVY,fs=15,sub='33 layers · 650M parameters')
# output
box(ax,8.35,0.3,4.35,1.0,'1280-d cell embedding',CORAL,tc='white',fs=13,sub='[CLS] token — universal across species')
# arrows
arrow(ax,2.7,1.27,3.15,1.27)                          # genome->ESM2
arrow(ax,5.2,1.27,5.6,1.27)                           # ESM2->prot emb
arrow(ax,6.67,1.85,5.6,2.55,c=NAVY)                   # prot emb -> tokens (up-left)
arrow(ax,2.7,4.05,5.2,3.45,c=TEAL)                    # scRNA -> tokens
arrow(ax,7.75,2.95,8.35,2.75)                         # tokens -> transformer
arrow(ax,10.52,1.55,10.52,1.3,c=CORAL)                # transformer -> output
ax.text(6.5,5.32,'The UCE model — one model, no cell-type labels, no gene-symbol matching',
        ha='center',fontsize=13.5,weight='bold',color=INK)
ax.text(6.5,5.0,'Genes are represented by the ESM-2 embedding of their protein  →  a new species just needs its proteome',
        ha='center',fontsize=10.5,color=GREY,style='italic')
plt.tight_layout(); plt.savefig('eval_out/model_schematic.png',dpi=170,bbox_inches='tight',facecolor='white'); plt.close()

# ---------------- WORKFLOW SCHEMATIC ----------------
fig,ax=plt.subplots(figsize=(12.8,3.9)); ax.set_xlim(0,12.8); ax.set_ylim(0,3.9); ax.axis('off')
fig.patch.set_facecolor('white')
steps=[('1','Dog proteome',TEAL,'Ensembl CanFam3.1'),
       ('2','ESM-2 15B\nembeddings',GOLD,'RunPod GPU'),
       ('3','Add dog as\nnew species',NAVY,'tokens · offsets'),
       ('4','Dog scRNA\nh5ad',TEAL,'GSE225599'),
       ('5','UCE\ninference',NAVY,'RunPod GPU'),
       ('6','Cell embeddings',CORAL,'93k × 1280'),
       ('7','Evaluation',TEAL,'3 strategies')]
x=0.15; w=1.63; gap=0.17; y=1.55; h=1.3
cx=[]
for i,(n,lab,c,sub) in enumerate(steps):
    tc='white' if c!=GOLD else INK
    box(ax,x,y,w,h,lab,c,tc=tc,fs=11,sub=sub)
    ax.text(x+0.19,y+h-0.17,n,ha='center',va='center',fontsize=10,weight='bold',
            color=c if c!=GOLD else INK,bbox=dict(boxstyle='circle,pad=0.16',fc='white',ec=c,lw=1.6))
    cx.append(x+w/2)
    if i<len(steps)-1:
        arrow(ax,x+w,y+h/2,x+w+gap,y+h/2)
    x+=w+gap
# three eval outputs under step 7
evs=[('Intrinsic (no ref)',TEAL),("Authors' UCE map",CORAL),('Baselines',GOLD)]
ey=0.35
for j,(lab,c) in enumerate(evs):
    bx=7.2+j*1.9
    tc='white' if c!=GOLD else INK
    box(ax,bx,ey,1.75,0.62,lab,c,tc=tc,fs=9.5,bold=True)
arrow(ax,cx[6],y, (7.2+1.9+0.87), ey+0.62, c=GREY, lw=1.6)
ax.text(3.55,3.4,'Evaluated three ways →',ha='center',fontsize=11,style='italic',color=GREY)
plt.tight_layout(); plt.savefig('eval_out/workflow_schematic.png',dpi=170,bbox_inches='tight',facecolor='white'); plt.close()
print('saved model_schematic.png + workflow_schematic.png')
