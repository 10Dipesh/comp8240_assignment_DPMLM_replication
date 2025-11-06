# run_new_datasets.py
# Launch 10-row smoke rewrites for your NEW datasets using the original DPMLM.

from pathlib import Path
from typing import Iterable
from run_original_dpmlm import run_dpmlm

# Prefer the location created by prepare_new_datasets.py, but be flexible.
CANDIDATE_ROOTS: Iterable[Path] = (
    Path("data/original_dataset"),
    Path("data/new_dataset"),
    Path("new_dataset"),
)

# Folder names must match the subfolder names on disk
DATASETS = {
    "IMDB_reviews":       ("IMDB_reviews",       ("validation", "test")),
    "mini_yelp":          ("mini_yelp",          ("validation", "test")),
    "name_redacted_bios": ("name_redacted_bios", ("validation", "test")),
}

def _split_paths(root: Path, folder: str, split: str):
    """Return candidate filenames for a split (supports .csv/.tsv and dev-fallback)."""
    if split == "validation":
        names = ("validation.csv", "validation.tsv", "dev.csv", "dev.tsv")
    else:
        names = (f"{split}.csv", f"{split}.tsv")
    return [root / folder / n for n in names]

def _find_root_with_splits(folder: str, splits: Iterable[str]) -> Path | None:
    """Find the first data root that contains all requested splits (with fallbacks)."""
    for root in CANDIDATE_ROOTS:
        if not root.exists():
            continue
        ok = True
        for sp in splits:
            if not any(p.exists() for p in _split_paths(root, folder, sp)):
                ok = False
                break
        if ok:
            return root
    return None

if __name__ == "__main__":
    for display, (folder, splits) in DATASETS.items():
        print(f"\n=== Running smoke test for {display} ===")

        root = _find_root_with_splits(folder, splits)
        if root is None:
            # Print helpful diagnostics for the first viable root (if any), else show all candidates
            print(f"[skip] {display}: required splits not found. Checked these locations:")
            for cand_root in CANDIDATE_ROOTS:
                print(f"  - {cand_root / folder}")
                for sp in splits:
                    for p in _split_paths(cand_root, folder, sp):
                        print(f"      • {p}")
            continue

        print(f"[info] Using data root: {root}")
        run_dpmlm(
            task=folder,                 # subfolder name under the chosen root
            data_dir=str(root),          # resolved root
            out_dir="outputs",
            splits=splits,               # ("validation","test")
            eps=(10, 50, 250),
            sample_size=10,              # 10-row smoke
            model_name="roberta-base",
            original_src_dir="dpmlm_src",
            preview=2,
            skip_missing=True,
        )
