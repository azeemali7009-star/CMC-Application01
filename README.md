# CMC Application 01: tumour transport with topology-constrained learning

This repository is a reproducible computational proof of concept for therapeutic transport on a tumour cell complex. It separates the physical mesh **M** (biological cells, compartments and interfaces) from a Forman quasi-cubical computational complex **K**, uses Combinatorial Mesh Calculus (CMC) to enforce conservative transport, and learns positive interface conductances with a Hodge-propagating neural model.

## Evidence included

- **Completed:** analytical reaction–diffusion convergence study.
- **Completed:** controlled heterogeneous breast-tumour spheroid phantom study, including homogeneous, local-MLP, CMC-HodgeNet and oracle comparators.
- **Not claimed as completed:** wet-lab, clinical or patient-level validation.
- **Provided for future validation:** downloader and preprocessing interfaces for the open Diosdi et al. (2025) multicellular light-sheet spheroid dataset.

The manuscript preserves the supplied `Background and Motivation` text in `manuscript/background_original.tex`. Suggested sentence-level improvements appear only in the manuscript appendix.

## Main files

- `manuscript/main.pdf` — compiled 32-page manuscript.
- `manuscript/main.tex` — LaTeX source.
- `manuscript/references.bib` — 119-entry bibliography; 104 works are cited in the article.
- `scripts/run_all.py` — regenerates analytical/synthetic data, models, result tables and figures.
- `src/cmc_tumour/` — CMC complex, phantom, transport, model and metric modules.
- `data/demo/` — generated CSV/NPZ benchmark data.
- `results/` — trained weights, predictions, metrics and run metadata.
- `figures/` — PNG and selected PDF figures.
- `scripts/download_figshare.py` — downloads public experimental feature files; `--all` requests all files.
- `scripts/preprocess_nifti.py` — optional future CT/MRI adapter.
- `scripts/neper_generate.sh` — optional Neper geometry-generation scaffold.

## Reproduce the computational results

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
```

The included run was generated with seed 2026 on a 57-cell, 96-interface complex. It takes about seconds to tens of seconds on a typical CPU, depending on the Python/PyTorch installation.

Run tests:

```bash
python -m pytest -q
```

Build the manuscript:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Retrieve the experimental data

Network access is required:

```bash
python scripts/download_figshare.py --out data/experimental_figshare
```

This downloads the small feature tables by default. Add `--all` only after checking disk space and the repository licence. Raw microscopy files are intentionally not redistributed in this archive.

## Scientific interpretation

On six held-out controlled phantoms, mean concentration relative L2 error was 0.366 for the homogeneous comparator, 0.238 for the local edge MLP and 0.105 for CMC-HodgeNet. This result demonstrates capability under a benchmark whose hidden law contains incidence-mediated Hodge context. It does **not** establish universal superiority or clinical efficacy.

## Licence

Code is supplied under the MIT License. External experimental data remain governed by their source repository terms and must be cited independently.


\section{Software layout and execution}\label{app:software}
The package has the following structure:
\begin{verbatim}
CMC_Application_01/
  configs/default.yaml
  data/demo/
  figures/
  manuscript/main.tex
  manuscript/references.bib
  results/
  scripts/run_all.py
  scripts/download_figshare.py
  scripts/preprocess_nifti.py
  scripts/neper_generate.sh
  src/cmc_tumour/
\end{verbatim}
From the project root:
\begin{verbatim}
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
cd manuscript && latexmk -pdf main.tex
\end{verbatim}
The Neper script is optional and is not required for the included cell-centred benchmark. The downloader requires network access and uses the public Figshare API. The NIfTI preprocessing script is provided for future CT/MRI extensions, but the selected first biological validation is light-sheet spheroid imaging.
