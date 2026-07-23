# Data artifacts

## Canine ESM-2 protein embeddings (the artifact we generated)

**📦 Download:** https://huggingface.co/datasets/bujnir/dog-esm2-15b-embeddings

- **File:** `dog_esm2_embeddings.pt`
- **Contents:** a dict `{ ENSCAFG_gene_id : float16[5120] }` — one ESM-2 (15B) protein embedding per canine gene
- **Genes:** 20,257 (longest protein per gene, Ensembl CanFam3.1 / release-104)
- **Size:** ~201 MB · md5 `7c35a83b285dd4163c445fb6de2ab364`

Too large for the git tree (GitHub caps normal files at 100 MB), so it's hosted on the Hugging Face Hub.

### Load it

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download(repo_id="bujnir/dog-esm2-15b-embeddings",
                       filename="dog_esm2_embeddings.pt",
                       repo_type="dataset")
emb = torch.load(path)          # dict: ENSCAFG -> float16[5120]
print(len(emb), next(iter(emb.values())).shape)   # 20257, torch.Size([5120])
```

### Regenerate it from scratch

```bash
python ../scripts/01_parse_proteome.py --fasta Canis_lupus_familiaris.CanFam3.1.pep.all.fa.gz --outdir .
python ../scripts/embed_esm2.py --fasta dog_longest.pep.fa --out dog_esm2_embeddings.pt --token-budget 4096 --resume
```

Needs a GPU with ~40 GB VRAM for ESM-2 15B (we used a RunPod A40, ~75 min).
