# run_original_dpmlm.py  (refined)
import os, sys, random
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
import pandas as pd
import numpy as np

os.environ.setdefault("PYTHONUTF8", "1")
random.seed(42); np.random.seed(42)

TASK_MODE: Dict[str, str] = {
    "cola": "single", "sst2": "single", "mrpc": "pair", "rte": "pair",
    # new datasets (single-text)
    "IMDB_reviews": "single", "mini_yelp": "single", "name_redacted_bios": "single",
}

SINGLE_TEXT_CANDIDATES = ["sentence","text","review","content","body","chatsentence"]
PAIR_A_CANDS = ["sentence1","text_a","premise","sentence"]
PAIR_B_CANDS = ["sentence2","text_b","hypothesis"]

def _read_any_csv_or_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower()==".tsv" else ",", encoding="utf-8")

def _read_split_df(task_dir: Path, split: str) -> pd.DataFrame:
    cand = [f"{split}.csv", f"{split}.tsv"]
    if split == "validation":
        cand = ["validation.csv","validation.tsv","dev.csv","dev.tsv"]
    for name in cand:
        f = task_dir / name
        if f.exists(): return _read_any_csv_or_tsv(f)
    raise FileNotFoundError(f"Missing split in {task_dir}: tried {cand}")

def _sample_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df): return df.reset_index(drop=True)
    return df.sample(n=n, random_state=1337).reset_index(drop=True)

def _pick_first(df: pd.DataFrame, cands: List[str]) -> str:
    cols_lc = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in cols_lc: return cols_lc[c]
    for c in df.columns:
        if df[c].dtype == object: return c
    raise ValueError(f"Could not find a text column in {list(df.columns)}")

def _normalize_columns(task: str, df: pd.DataFrame) -> Tuple[str, Tuple[str,str], pd.DataFrame]:
    mode = TASK_MODE.get(task, "single")
    out = df.copy()
    if mode == "pair":
        s1 = _pick_first(out, PAIR_A_CANDS)
        s2 = _pick_first(out, PAIR_B_CANDS)
        ren = {}
        if s1 != "sentence1": ren[s1] = "sentence1"
        if s2 != "sentence2": ren[s2] = "sentence2"
        out = out.rename(columns=ren)
        return "pair", ("sentence1","sentence2"), out
    s = _pick_first(out, SINGLE_TEXT_CANDIDATES)
    if s != "sentence": out = out.rename(columns={s:"sentence"})
    return "single", ("sentence",""), out

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _load_original_dpmlm(src_dir: Path, model_name: str):
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    DP = __import__("DPMLM")
    cls = getattr(DP, "DPMLM", None)
    if cls is None:
        raise ImportError("DPMLM class not found in DPMLM.py (expected in original repo)")
    try:
        return cls(MODEL=model_name)
    except TypeError:
        return cls()

def _truncate_words(text: str, max_words: int | None) -> str:
    if not max_words or max_words <= 0:
        return text
    parts = str(text).split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])

def _rewrite_one(dp, text: str, epsilon: float, force_replace: bool) -> str:
    try:
        new_text, *_ = dp.dpmlm_rewrite(
            str(text), float(epsilon),
            REPLACE=bool(force_replace),  # <- force change when DP returns same token
            FILTER=True,
            STOP=False,
            TEMP=True,
            POS=True,
            CONCAT=True
        )
        return str(new_text)
    except Exception:
        return str(text)

def run_dpmlm(
    task: str,
    data_dir: str = "data/original_dataset",
    out_dir: str = "outputs",
    splits: Iterable[str] = ("validation","test"),
    eps: Iterable[float] = (10,50,250),
    sample_size: int = 10,
    model_name: str = "roberta-base",
    original_src_dir: str = "dpmlm_src",
    preview: int = 0,
    skip_missing: bool = True,
    max_words: int | None = 200,     # <- NEW: keep under the 512 token window
    force_replace: bool = False,     # <- NEW: more visible changes if needed
) -> None:
    src_dir = Path(original_src_dir)
    base = Path(data_dir) / task
    if not base.exists():
        if skip_missing:
            print(f"[skip] {task}: {base} not found")
            return
        raise FileNotFoundError(f"{base} does not exist")

    dp = _load_original_dpmlm(src_dir, model_name)

    for split in splits:
        try:
            df = _read_split_df(base, split)
        except FileNotFoundError:
            if skip_missing:
                print(f"[skip] {task} {split}: split file not found")
                continue
            raise

        mode, names, df = _normalize_columns(task, df)
        needed = [n for n in names if n]
        df = df.dropna(subset=needed).reset_index(drop=True)
        df = _sample_df(df, sample_size)

        changed = 0
        total = 0

        for e in eps:
            out_split_dir = Path(out_dir) / "privatized" / task / f"eps_{int(e)}"
            _ensure_dir(out_split_dir)
            dst_csv = out_split_dir / f"{split}.csv"

            rows = []; previews = []
            for _, r in df.iterrows():
                row = r.to_dict()
                if mode == "single":
                    t0 = str(r[names[0]])
                    t0t = _truncate_words(t0, max_words)
                    t1 = _rewrite_one(dp, t0t, e, force_replace)
                    # keep original length context in file (optional)
                    row[names[0]] = t1
                    if t1 != t0: changed += 1
                    total += 1
                    if len(previews) < preview:
                        previews.append((t0[:300], t1[:300]))
                else:
                    a0 = str(r[names[0]]); b0 = str(r[names[1]])
                    a0t = _truncate_words(a0, max_words)
                    b0t = _truncate_words(b0, max_words)
                    a1 = _rewrite_one(dp, a0t, e, force_replace)
                    b1 = _rewrite_one(dp, b0t, e, force_replace)
                    row[names[0]] = a1; row[names[1]] = b1
                    changed += int(a1 != a0) + int(b1 != b0)
                    total += 2
                    if len(previews) < preview:
                        previews.append(((a0+" [SEP] "+b0)[:300], (a1+" [SEP] "+b1)[:300]))
                rows.append(row)

            pd.DataFrame(rows).to_csv(dst_csv, index=False, encoding="utf-8")
            print(f"[OK][orig] {task} | ε={int(e)} | {split}: {dst_csv}  "
                  f"(changed {changed}/{total} tokens across sampled rows)")

            if previews:
                print(f"\n[Preview] {task} | ε={int(e)} | {split}")
                for a,b in previews:
                    print("-"*60)
                    print("ORIG:", a.replace("\n"," "))
                    print("NEW :", b.replace("\n"," "))
                print("-"*60+"\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--data_dir", type=str, default="data/original_dataset")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--splits", nargs="+", default=["validation","test"])
    ap.add_argument("--eps", nargs="+", type=float, default=[10,50,250])
    ap.add_argument("--sample_size", type=int, default=10)
    ap.add_argument("--model_name", type=str, default="roberta-base")
    ap.add_argument("--original_src_dir", type=str, default="dpmlm_src")
    ap.add_argument("--preview", type=int, default=2)
    ap.add_argument("--max_words", type=int, default=200)
    ap.add_argument("--force_replace", action="store_true")
    args = ap.parse_args()
    for t in args.tasks:
        run_dpmlm(
            task=t, data_dir=args.data_dir, out_dir=args.out_dir,
            splits=tuple(args.splits), eps=tuple(args.eps), sample_size=args.sample_size,
            model_name=args.model_name, original_src_dir=args.original_src_dir,
            preview=args.preview, max_words=args.max_words, force_replace=args.force_replace
        )
