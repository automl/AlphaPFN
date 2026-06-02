# α-PFN project page

Static single-page site for **α-PFN: Fast Entropy Search via In-Context Learning** (ICML 2026).
Forked from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) (Bulma, no build step).

## Local preview

```bash
python3 -m http.server -d . 8000
# open http://localhost:8000
```

## Deploy to GitHub Pages on the `gh-pages` branch of `automl/AlphaPFN`

From a clone of `automl/AlphaPFN`:

```bash
# 1. Create an orphan gh-pages branch and clear it.
git checkout --orphan gh-pages
git rm -rf .

# 2. Copy the site files into the branch.
rsync -a --delete \
    --exclude='.git' \
    /work/dlclarge2/rakotoah-entropy_search/misc/heri/alphapfn-page/ ./

# 3. Commit and push.
git add .
git commit -m "Initial project page"
git push -u origin gh-pages
```

In the repo's *Settings → Pages*:
- **Source:** Deploy from a branch
- **Branch:** `gh-pages` / root

Site URL: `https://automl.github.io/AlphaPFN/`.

## Updating content

All copy lives in `index.html`. Figures live in `static/images/`. The page uses MathJax for inline math and
highlight.js for Python syntax highlighting — both via CDN, no rebuild step.

Acknowledgments: built on the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) by Eliahu Horwitz (adopted from the [Nerfies](https://nerfies.github.io) page).
