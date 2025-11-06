# run_yelp_trustpilot.py
from run_original_dpmlm import run_dpmlm

if __name__ == "__main__":
    # Yelp10
    run_dpmlm(
        task="yelp10",
        data_dir="data/original_dataset",
        out_dir="outputs",
        splits=("validation","test"),
        eps=(10,50,250),
        sample_size=10,
        model_name="roberta-base",
        original_src_dir="dpmlm_src",
        preview=2
    )
    # Trustpilot
    run_dpmlm(
        task="trustpilot",
        data_dir="data/original_dataset",
        out_dir="outputs",
        splits=("validation","test"),
        eps=(10,50,250),
        sample_size=10,
        model_name="roberta-base",
        original_src_dir="dpmlm_src",
        preview=2
    )
