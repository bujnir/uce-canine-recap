# How to publish this repo

```bash
cd uce-canine-recap
git init
git add .
git commit -m "UCE canine recap: zero-shot cross-species annotation + honest benchmark"

# create the repo on GitHub (via gh CLI)
gh repo create uce-canine-recap --public --source=. --remote=origin --push
# ...or make an empty repo on github.com and:
# git remote add origin https://github.com/<your-username>/uce-canine-recap.git
# git branch -M main && git push -u origin main
```

Then publish the 201 MB ESM-2 embeddings separately (Hugging Face recommended) — see
[`data/DATA.md`](data/DATA.md) — and update the link in `data/DATA.md` / `README.md`.

Quick fill-ins before pushing:
- Replace `<your-username>` in `data/DATA.md`.
- (Optional) put your name in `LICENSE`.
- (Optional) add a repo description + topics on GitHub: `single-cell`, `foundation-model`, `cross-species`, `veterinary`, `UCE`.
