# Plot results

Regenerates every figure in the paper from a single aggregated CSV.

The raw input is `all_results.csv.gz` — one row per
`(benchmark, optimizer, seed, step)` triple, produced by stage 3
(`bo_evaluation/`). To rebuild the PDFs in this directory:

```bash
pip install matplotlib scipy tueplots
python plot_results/plot_from_csv.py
```

The script writes each figure next to itself, so it works with no
arguments.
