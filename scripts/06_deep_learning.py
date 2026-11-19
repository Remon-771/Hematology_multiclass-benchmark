"""Notebook §10: FT-Transformer, residual MLP and (if installed) TabNet on the training sub-split (Table 3)."""
from _common import base_parser, load_split, setup
from cbcbench import deep


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out = setup(args)
    _, split = load_split(args)
    if not deep.TORCH_OK:
        print("PyTorch is not available; deep models are skipped and the leaderboard is built from classical models only.")
        out.cache_save("deep", {"dl_val_rows": [], "dl_histories": {}})
        return
    print("device:", deep.device)
    dl_results, dl_histories, prep = deep.train_all_dl(split, cfg)
    table = deep.dl_validation_table(dl_results, prep, split)
    if len(table):
        table.to_csv(out.table("model_selection", "Table3_DL_validation_scores.csv"), index=False)
    print(table)
    out.cache_save("deep", {"dl_val_rows": table.to_dict("records"), "dl_histories": dl_histories})
    deep_names = list(dl_results)
    out.cache_save("deep_models", {"names": deep_names})


if __name__ == "__main__":
    main()
