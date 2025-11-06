# prepare_new_datasets.py
# Normalize datasets from data/new_dataset/* into data/original_dataset/* (GLUE-like)

from pathlib import Path
import argparse
import pandas as pd

TEXT_CANDS  = ["sentence","text","review","content","body"]
LABEL_CANDS = ["label","sentiment","author","author_id","user_id","user",
               "gender","sex","stars","rating","score"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pick_col(df: pd.DataFrame, cands, fallback_first_object=False):
    lc = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in lc:
            return lc[c]
    if fallback_first_object:
        for c in df.columns:
            if df[c].dtype == object:
                return c
        return df.columns[0]
    return None

def fix_frame(df: pd.DataFrame):
    df2 = df.copy()
    t = pick_col(df2, TEXT_CANDS, fallback_first_object=True)
    l = pick_col(df2, LABEL_CANDS, fallback_first_object=False)
    if t != "sentence":
        df2 = df2.rename(columns={t: "sentence"})
    if l and l != "label":
        df2 = df2.rename(columns={l: "label"})
    if "label" not in df2.columns:
        df2["label"] = -1
    return df2[["sentence","label"]]

def write_csv(df: pd.DataFrame, dst: Path, limit: int):
    if limit and limit > 0:
        df = df.head(limit).copy()
    ensure_dir(dst.parent)
    df.to_csv(dst, index=False, encoding="utf-8")
    print(f"[OK] {dst}  ({len(df)} rows)")

def load_txt_pair(folder: Path, stem: str):
    txt = folder / f"{stem}.txt"
    lab = folder / f"{stem}_labels.txt"
    if not txt.exists() or not lab.exists():
        return None
    s = [line.rstrip("\r\n") for line in txt.open(encoding="utf-8")]
    y = [line.rstrip("\r\n") for line in lab.open(encoding="utf-8")]
    n = min(len(s), len(y))
    return pd.DataFrame({"sentence": s[:n], "label": y[:n]})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Keep first N rows per split (0 = all)")
    args = ap.parse_args()

    BASE = Path(__file__).resolve().parent
    # Prefer data/new_dataset/, fallback to new_dataset/
    candidates = [BASE / "data" / "new_dataset", BASE / "new_dataset"]
    NEW_ROOT = next((p for p in candidates if p.exists()), None)
    if NEW_ROOT is None:
        print("Could not find 'data/new_dataset' or 'new_dataset'.")
        return
    OUT_ROOT = BASE / "data" / "original_dataset"
    print(f"Using source: {NEW_ROOT}\nOutput root: {OUT_ROOT}\n")

    # ---------------- IMDB ----------------
    imdb = NEW_ROOT / "IMDB_reviews"
    print(f"[IMDB] {imdb}")
    if imdb.exists():
        # Accept either your generic train/test.csv OR imdb_train/imdb_test.csv
        train_csv = imdb / "train.csv"
        test_csv  = imdb / "test.csv"
        if not train_csv.exists(): train_csv = imdb / "imdb_train.csv"
        if not test_csv.exists():  test_csv  = imdb / "imdb_test.csv"
        if train_csv.exists() and test_csv.exists():
            write_csv(fix_frame(pd.read_csv(train_csv, encoding="utf-8")), OUT_ROOT / "IMDB_reviews" / "train.csv", args.limit)
            write_csv(fix_frame(pd.read_csv(test_csv,  encoding="utf-8")), OUT_ROOT / "IMDB_reviews" / "test.csv",  args.limit)
        else:
            print("  -> expected train.csv/test.csv (or imdb_train.csv/imdb_test.csv); skipping")
    else:
        print("  -> folder missing; skipping")

    print()

    # --------------- mini_yelp ---------------
    my = NEW_ROOT / "mini_yelp"
    print(f"[mini_yelp] {my}")
    if my.exists():
        # Prefer your already-made CSVs
        tr = my / "train.csv"
        va = my / "validation.csv"
        te = my / "test.csv"
        if not (tr.exists() and va.exists() and te.exists()):
            # Fallback to authors CSVs if present
            alt_tr = my / "mini_yelp_authors_train.csv"
            alt_va = my / "mini_yelp_authors_dev.csv"
            alt_te = my / "mini_yelp_authors_test.csv"
            if alt_tr.exists(): tr = alt_tr
            if alt_va.exists(): va = alt_va
            if alt_te.exists(): te = alt_te
        if tr.exists():
            write_csv(fix_frame(pd.read_csv(tr, encoding="utf-8")), OUT_ROOT / "mini_yelp" / "train.csv", args.limit)
        if va.exists():
            write_csv(fix_frame(pd.read_csv(va, encoding="utf-8")), OUT_ROOT / "mini_yelp" / "validation.csv", args.limit)
        if te.exists():
            write_csv(fix_frame(pd.read_csv(te, encoding="utf-8")), OUT_ROOT / "mini_yelp" / "test.csv", args.limit)
        if not (tr.exists() or va.exists() or te.exists()):
            print("  -> no usable CSVs found; skipping")
    else:
        print("  -> folder missing; skipping")

    print()

    # ----------- name_redacted_bios -----------
    nb = NEW_ROOT / "name_redacted_bios"
    print(f"[name_redacted_bios] {nb}")
    if nb.exists():
        out = OUT_ROOT / "name_redacted_bios"
        # Try txt+labels pairs
        for split in ("train", "validation", "test"):
            df = load_txt_pair(nb, split)
            if df is not None:
                write_csv(df, out / f"{split}.csv", args.limit)

        # If still missing, try a single CSV containing all splits
        if not (out / "train.csv").exists() and (nb / "bios_redacted_splits.csv").exists():
            whole = pd.read_csv(nb / "bios_redacted_splits.csv", encoding="utf-8")
            lc = {c.lower(): c for c in whole.columns}
            txt = lc.get("text", lc.get("sentence", None))
            lab = lc.get("label", None)
            sp  = lc.get("split", None)
            if txt and lab and sp:
                for split in ("train", "validation", "test"):
                    part = whole[whole[sp].astype(str).str.lower() == split].rename(columns={txt: "sentence"})
                    part = part[["sentence", lab]].rename(columns={lab: "label"})
                    write_csv(part, out / f"{split}.csv", args.limit)
            else:
                print("  -> bios_redacted_splits.csv lacks {text/label/split}; skipping fallback")
    else:
        print("  -> folder missing; skipping")

    print("\nDone.")

if __name__ == "__main__":
    main()
