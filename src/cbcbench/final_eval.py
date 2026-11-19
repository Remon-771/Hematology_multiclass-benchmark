"""Final model construction, locked test-set evaluation, sensitivity and interpretation (notebook §12-§20)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, classification_report, f1_score,
                             make_scorer, precision_recall_curve, roc_auc_score, roc_curve, auc, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import label_binarize

from .config import ABS_DIFF_CANDIDATES, HYBRID_NAME, SEED, WBC_CORR_THRESHOLD
from .models import TREE_MODELS
from .preprocessing import make_full_pipeline, make_preprocessor
from . import deep


# --------------------------------------------------------------------------------------------- final fit
def fit_final_model(best_name, split, cfg, tuned_models, base_models):
    """Refit the selected model on the full training split and predict the locked test set.

    Returns a dict with kind in {"sklearn_pipeline", "torch", "tabnet"}.
    """
    is_dl_best = deep.TORCH_OK and best_name in ("Tabular ResNet-MLP", "FT-Transformer", "TabNet")
    clean = best_name.replace(" (tuned)", "")
    if not is_dl_best:
        model_obj = tuned_models.get(clean, base_models.get(clean))
        pipe = make_full_pipeline(model_obj, split.feature_names, use_smote=True)
        pipe.fit(split.X_train, split.y_train)
        return {"kind": "sklearn_pipeline", "obj": pipe, "model_obj": model_obj, "preprocessor": None,
                "preds": pipe.predict(split.X_test), "probs": pipe.predict_proba(split.X_test), "is_dl": False,
                "clean_name": clean, "n_features": None, "best_name": best_name}
    prep = make_preprocessor(split.feature_names)
    Xtr = prep.fit_transform(split.X_train)
    Xte = prep.transform(split.X_test)
    Xb, yb = SMOTE(random_state=SEED, k_neighbors=5).fit_resample(Xtr, split.y_train)
    n_features = Xb.shape[1]
    if best_name == "TabNet":
        obj = deep.TabNetClassifier(n_d=32, n_a=32, n_steps=5, gamma=1.5, seed=SEED, verbose=0)
        obj.fit(Xb, yb, max_epochs=cfg.tabnet_epochs, patience=15, batch_size=256, virtual_batch_size=64)
        probs, kind = obj.predict_proba(Xte), "tabnet"
    else:
        arch = deep.TabularResNet if "ResNet" in best_name else deep.FTTransformer
        obj = arch(n_features, split.n_classes)
        # As in the study notebook, the early-stopping monitor of the final deep refit is the held-out split.
        obj, _ = deep.train_torch_model(obj, deep.to_loader(Xb, yb), deep.to_loader(Xte, split.y_test, shuffle=False),
                                        cfg.dl_epochs, name=best_name + "-final")
        probs, kind = deep.torch_probs(obj, Xte), "torch"
    return {"kind": kind, "obj": obj, "model_obj": obj, "preprocessor": prep, "preds": probs.argmax(1), "probs": probs,
            "is_dl": True, "clean_name": clean, "n_features": n_features, "best_name": best_name}


def headline_metrics(y_test, preds, probs) -> dict:
    return {"accuracy": accuracy_score(y_test, preds), "macro_f1": f1_score(y_test, preds, average="macro"),
            "weighted_f1": f1_score(y_test, preds, average="weighted"),
            "roc_auc": roc_auc_score(y_test, probs, average="macro", multi_class="ovr")}


def classification_report_table(y_test, preds, class_names) -> pd.DataFrame:
    rep = classification_report(y_test, preds, target_names=class_names, output_dict=True)
    return pd.DataFrame(rep).transpose().round(4)


# --------------------------------------------------------------------------------------------- figures
def plot_confusion(y_test, preds, class_names, path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_test, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion matrix (counts)"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True"); axes[0].tick_params(axis="x", rotation=45)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion matrix (row-normalised)"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True"); axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close(fig)


def roc_pr_and_ap(y_test, probs, class_names, best_name, roc_path, pr_path) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    plt.figure(figsize=(7.5, 6.5))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"ROC curves (one-vs-rest) -- {best_name}, test set"); plt.legend(loc="lower right", fontsize=8)
    plt.savefig(roc_path, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(7.5, 6.5))
    rows = []
    for i, cls in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i])
        rows.append({"Class": cls, "Average_Precision": round(ap, 4)})
        plt.plot(recall, precision, lw=2, label=f"{cls} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-recall curves (one-vs-rest) -- {best_name}, test set"); plt.legend(loc="lower left", fontsize=8)
    plt.savefig(pr_path, bbox_inches="tight", dpi=300); plt.close()
    pr_df = pd.DataFrame(rows).sort_values("Average_Precision", ascending=False).reset_index(drop=True)
    pr_df.loc[len(pr_df)] = ["Macro_average", pr_df["Average_Precision"].mean()]
    return pr_df


def calibration_and_brier(y_test, probs, class_names, best_name, path) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    plt.figure(figsize=(7.5, 6.5))
    rows = []
    for i, cls in enumerate(class_names):
        frac_pos, mean_pred = calibration_curve(y_bin[:, i], probs[:, i], n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_bin[:, i], probs[:, i])
        rows.append({"Class": cls, "Brier_score": round(brier, 4)})
        plt.plot(mean_pred, frac_pos, "o-", lw=1.5, label=f"{cls} (Brier={brier:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Calibration (one-vs-rest) -- {best_name}, test set"); plt.legend(loc="upper left", fontsize=8)
    plt.savefig(path, bbox_inches="tight", dpi=300); plt.close()
    df = pd.DataFrame(rows).sort_values("Brier_score").reset_index(drop=True)
    df.loc[len(df)] = ["Macro_average", df["Brier_score"].mean()]
    return df


def leaderboard_figure(leaderboard, dl_names, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    plot_df = leaderboard.sort_values("Selection_Score_MacroF1")
    colors = ["#4C72B0" if "tuned" not in m and m not in dl_names and m != HYBRID_NAME else
              ("#DD8452" if "tuned" in m else ("#8172B2" if m == HYBRID_NAME else "#55A868")) for m in plot_df["Model"]]
    ax.barh(plot_df["Model"], plot_df["Selection_Score_MacroF1"], color=colors)
    ax.set_xlabel("Training-data selection score (macro-F1)")
    ax.set_title("Model comparison (blue = baseline classical, orange = tuned classical, purple = stacking, green = deep learning)")
    ax.set_xlim(0, 1.02)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------------- WBC sensitivity (§14)
def wbc_family(feature_names, X_train, abs_diff):
    if "WBC" not in feature_names:
        return []
    name_tags = ["ANC", "ALC", "AMC", "AEC", "ABC", "_WBC_", "WBC_ratio", "ESR_WBC"] + list(abs_diff)
    name_hits = [c for c in feature_names if c != "WBC" and any(tag in c for tag in name_tags)]
    numeric_train = X_train[feature_names].select_dtypes(include=[np.number])
    corr_hits = []
    if "WBC" in numeric_train.columns:
        tc = numeric_train.drop(columns=["WBC"]).corrwith(numeric_train["WBC"]).abs()
        corr_hits = tc[tc > WBC_CORR_THRESHOLD].index.tolist()
    return sorted(set(["WBC"] + name_hits + corr_hits))


def refit_and_evaluate(drop_features, tag, split, final, cfg):
    remaining = [c for c in split.feature_names if c not in drop_features]
    assert len(remaining) >= 3, f"Too few features remain after removing {drop_features}."
    Xtr_d, Xte_d = split.X_train[remaining], split.X_test[remaining]
    if not final["is_dl"]:
        pipe = make_full_pipeline(clone(final["model_obj"]), remaining, use_smote=True)
        pipe.fit(Xtr_d, split.y_train)
        preds, probs = pipe.predict(Xte_d), pipe.predict_proba(Xte_d)
    else:
        prep = make_preprocessor(remaining)
        Xtr_p, Xte_p = prep.fit_transform(Xtr_d), prep.transform(Xte_d)
        Xtr_b, ytr_b = SMOTE(random_state=SEED, k_neighbors=5).fit_resample(Xtr_p, split.y_train)
        best_name = final["best_name"]
        if best_name == "TabNet":
            m = deep.TabNetClassifier(n_d=32, n_a=32, n_steps=5, gamma=1.5, seed=SEED, verbose=0)
            m.fit(Xtr_b, ytr_b, max_epochs=cfg.tabnet_epochs, patience=15, batch_size=256, virtual_batch_size=64)
            probs = m.predict_proba(Xte_p)
        else:
            arch = deep.TabularResNet if "ResNet" in best_name else deep.FTTransformer
            m = arch(Xtr_b.shape[1], split.n_classes)
            m, _ = deep.train_torch_model(m, deep.to_loader(Xtr_b, ytr_b), deep.to_loader(Xte_p, split.y_test, shuffle=False),
                                          cfg.dl_epochs, name=f"{best_name}-{tag}")
            probs = deep.torch_probs(m, Xte_p)
        preds = probs.argmax(1)
    report = classification_report(split.y_test, preds, target_names=split.class_names, output_dict=True)
    metrics = {"Accuracy": accuracy_score(split.y_test, preds), "MacroF1": f1_score(split.y_test, preds, average="macro"),
               "ROC_AUC": roc_auc_score(split.y_test, probs, average="macro", multi_class="ovr")}
    return report, metrics


# ----------------------------------------------------------------------------------- interpretation (§20)
class RawFeatureWrapper(BaseEstimator, ClassifierMixin):
    """Uniform predict/predict_proba over the original feature columns for sklearn pipelines, PyTorch or TabNet."""

    def __init__(self, kind, obj, preprocessor=None, n_classes=5):
        self.kind, self.obj, self.preprocessor, self.n_classes = kind, obj, preprocessor, n_classes
        self.classes_ = np.arange(n_classes)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        if self.kind == "sklearn_pipeline":
            return self.obj.predict_proba(X)
        X_proc = self.preprocessor.transform(X)
        if self.kind == "tabnet":
            return self.obj.predict_proba(X_proc)
        return deep.torch_probs(self.obj, X_proc)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def shap_summary(final, split, path, seed=SEED):
    """SHAP beeswarm for tree-based sklearn pipelines (returns False when not applicable)."""
    if final["is_dl"] or final["clean_name"] not in TREE_MODELS:
        print("Selected model is not a tree model; see permutation importance instead.")
        return False
    import matplotlib.pyplot as plt
    import shap
    pre = final["obj"].named_steps["preprocess"]
    X_sample = pre.transform(split.X_train.sample(min(2000, len(split.X_train)), random_state=seed))
    names = pre.get_feature_names_out()
    shap_values = shap.TreeExplainer(final["obj"].named_steps["model"]).shap_values(X_sample)
    plt.figure()
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_sample, feature_names=names, class_names=split.class_names, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, feature_names=names, show=False)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    return True


def permutation_table(final, split, best_name, path, seed=SEED):
    import matplotlib.pyplot as plt
    wrapped = RawFeatureWrapper(final["kind"], final["obj"], final["preprocessor"], split.n_classes)
    reps = 5 if best_name == HYBRID_NAME else 10
    res = permutation_importance(wrapped, split.X_test, split.y_test, scoring="f1_macro", n_repeats=reps, random_state=seed, n_jobs=1)
    df = pd.DataFrame({"Feature": split.X_test.columns, "Perm_Importance_mean": res.importances_mean,
                       "Perm_Importance_std": res.importances_std}).sort_values("Perm_Importance_mean", ascending=False).reset_index(drop=True)
    top_n = min(20, len(df))
    plt.figure(figsize=(8, max(4, 0.35 * top_n)))
    plt.barh(df["Feature"].head(top_n)[::-1], df["Perm_Importance_mean"].head(top_n)[::-1],
             xerr=df["Perm_Importance_std"].head(top_n)[::-1], color="#55A868")
    plt.xlabel(f"Decrease in macro-F1 when the feature is permuted (test set, {reps} repeats)")
    plt.title(f"Permutation feature importance -- {best_name}, test set")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight", dpi=300); plt.close()
    return df


# ------------------------------------------------------------------------------------ learning curve (§18)
def learning_curve_figure(model, split, clean_name, is_ensemble, path):
    import matplotlib.pyplot as plt
    cv = StratifiedKFold(n_splits=3 if is_ensemble else 5, shuffle=True, random_state=SEED)
    n_sizes = 4 if is_ensemble else 6
    pipe = make_full_pipeline(model, split.feature_names, use_smote=True)
    sizes, tr, va = learning_curve(pipe, split.X_train, split.y_train, cv=cv, scoring=make_scorer(f1_score, average="macro"),
                                   train_sizes=np.linspace(0.2, 1.0, n_sizes), n_jobs=-1, shuffle=True, random_state=SEED)
    plt.figure(figsize=(7, 5.5))
    for arr, label in ((tr, "Training score"), (va, "Cross-validation score")):
        m, s = arr.mean(axis=1), arr.std(axis=1)
        plt.plot(sizes, m, "o-", label=label); plt.fill_between(sizes, m - s, m + s, alpha=0.15)
    plt.xlabel("Training examples"); plt.ylabel("Macro-F1")
    plt.title(f"Learning curve -- {clean_name} (training data only)"); plt.legend(loc="lower right"); plt.ylim(0.0, 1.02)
    plt.savefig(path, bbox_inches="tight"); plt.close()


def dl_history_figure(dl_histories, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(dl_histories), figsize=(6 * len(dl_histories), 5), squeeze=False)
    for i, (name, hist) in enumerate(dl_histories.items()):
        ax = axes[0][i]
        ax.plot(hist["train_loss"], label="Train loss"); ax.plot(hist["val_loss"], label="Val loss")
        ax.set_title(f"{name} training history"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close(fig)
