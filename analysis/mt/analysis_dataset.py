from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from caas_jupyter_tools import display_dataframe_to_user
except Exception:
    def display_dataframe_to_user(title: str, df: pd.DataFrame):
        """Fallback display: prints head of dataframe."""
        print(f"\n== {title} ==")
        with pd.option_context("display.max_columns", None):
            print(df.head())

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tokenization dataset and run logistic regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add = parser.add_argument
    add("--input-csv", required=True, help="Input dataset CSV (needs label, subject, *_baseline/_alternative columns)")
    add("--output-dir", default="analysis_outputs", help="Where to write CSV artifacts and optional plots")
    add("--metrics", nargs="+", default=["renyi", "entropy", "norm"], help="Metric base names present as *_baseline/_alternative")
    add("--n-splits", type=int, default=5, help="StratifiedKFold splits for CV")
    add("--random-state", type=int, default=42, help="Random seed")
    add("--max-iter", type=int, default=1000, help="LogisticRegression max_iter")
    add("--top-k", type=int, default=12, help="Top/bottom K subjects to plot")
    add("--bins", type=int, default=40, help="Histogram bins (entropy_diff)")
    add("--save-plots", action="store_true", help="Save plots (PNG) instead of showing")
    add("--no-plots", action="store_true", help="Skip plot generation entirely")
    add("--subject-col", default="subject", help="Subject/group column name")
    add("--label-col", default="label", help="Label column (1=baseline,0=alt,-1=tie)")
    add(
        "--features",
        nargs="+",
        help="Explicit feature columns for logistic regression (else all *_diff incl. len_diff + token cat diffs)",
    )
    add("--list-features", action="store_true", help="List default feature columns then exit")
    add("--quiet", action="store_true", help="Minimize console output")
    return parser.parse_args()


def load_dataset(path: str | Path, label_col: str) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df_local = pd.read_csv(path)
    if label_col not in df_local.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset columns: {list(df_local.columns)}")
    if df_local[label_col].dtype == bool:
        df_local["label_int"] = df_local[label_col].astype(int)
    else:
        df_local["label_int"] = df_local[label_col].astype(int)
    return df_local

def parse_tokens(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []

def add_metric_diffs(df: pd.DataFrame, metrics: list[str]) -> None:
    for metric in metrics:
        base_col = f"{metric}_baseline"
        alt_col = f"{metric}_alternative"
        if base_col in df.columns and alt_col in df.columns:
            df[f"{metric}_diff"] = df[base_col] - df[alt_col]
        else:
            print(f"[warn] Skipping metric '{metric}' (missing '{base_col}' or '{alt_col}')")


OPS = set("+-*/^=()[]{}<>|,:;")

def token_categories(tokens):
    n = len(tokens)
    if n == 0:
        return {
            "letters": 0.0,
            "digits": 0.0,
            "ops": 0.0,
            "other": 0.0,
            "newline": 0.0,
            "spacepref": 0.0,
            "avg_token_len": np.nan,
        }
    letters = digits = ops = other = newline = spacepref = 0
    lens = []
    for t in tokens:
        if t == "Ċ":
            newline += 1
        if isinstance(t, str) and t.startswith("Ġ"):
            spacepref += 1
        s = t.replace("Ġ", "") if isinstance(t, str) else str(t)
        lens.append(len(s))
        if re.fullmatch(r"\d+", s):
            digits += 1
        elif re.fullmatch(r"[A-Za-z]+", s):
            letters += 1
        elif any(ch in OPS for ch in s):
            ops += 1
        else:
            other += 1
    return {
        "letters": letters / n,
        "digits": digits / n,
        "ops": ops / n,
        "other": other / n,
        "newline": newline / n,
        "spacepref": spacepref / n,
        "avg_token_len": float(np.mean(lens)) if lens else np.nan,
    }

def add_token_category_features(df: pd.DataFrame) -> list[str]:
    feature_bases = ["letters", "digits", "ops", "other", "newline", "spacepref", "avg_token_len"]
    for side in ["baseline", "alternative"]:
        feats = df[f"tokens_{side}_list"].apply(token_categories).apply(pd.Series)
        for c in feats.columns:
            df[f"{c}_{side}"] = feats[c]
    for c in feature_bases:
        df[f"{c}_diff"] = df[f"{c}_baseline"] - df[f"{c}_alternative"]
    return [f"{c}_diff" for c in feature_bases]


def compute_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    class_balance = (
        df["label_int"]
        .value_counts()
        .rename(index={0: "alt_better", 1: "baseline_better"})
        .rename("count")
        .to_frame()
    )
    class_balance["share"] = (class_balance["count"] / class_balance["count"].sum()).round(4)
    return class_balance


def summarize_core_metrics(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    core_cols = []
    for m in metrics:
        for suf in ["baseline", "alternative", "diff"]:
            col = f"{m}_{suf}"
            if col in df.columns:
                core_cols.append(col)
    core_cols += [c for c in ["len_baseline", "len_alternative", "len_diff"] if c in df.columns]
    core_cols = list(dict.fromkeys(core_cols))  # dedupe preserving order
    summary_by_label = df.groupby("label_int")[core_cols].mean().rename(index={0: "alt_better", 1: "baseline_better"})
    return summary_by_label


def welch_t(df: pd.DataFrame, col: str):
    a = df.loc[df["label_int"] == 1, col].dropna().values
    b = df.loc[df["label_int"] == 0, col].dropna().values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # Cohen's d
    va = np.var(a, ddof=1); vb = np.var(b, ddof=1)
    na = len(a); nb = len(b)
    pooled = ((na-1)*va + (nb-1)*vb) / (na+nb-2)
    d = (np.mean(a)-np.mean(b)) / np.sqrt(pooled) if pooled>0 else np.nan
    return pd.Series({
        "mean(baseline_better)": float(np.mean(a)),
        "mean(alt_better)": float(np.mean(b)),
        "t_stat": float(t),
        "p_value": float(p),
        "cohens_d": float(d),
    })

def compute_welch_tests(df: pd.DataFrame, diff_cols: list[str]) -> pd.DataFrame:
    ttest_table = pd.DataFrame({c: welch_t(df, c) for c in diff_cols if c in df.columns}).T.sort_values("p_value")
    return ttest_table


def run_logreg(df: pd.DataFrame, feature_cols: list[str], n_splits: int, random_state: int, max_iter: int):
    avail = [c for c in feature_cols if c in df.columns]
    if not avail:
        raise ValueError("No available feature columns for logistic regression.")
    X = df[avail].fillna(0.0)
    y = df["label_int"].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=max_iter))
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    pipe.fit(X, y)
    coefs = pd.DataFrame({
        "feature": avail,
        "coef": pipe.named_steps["logreg"].coef_[0]
    }).sort_values("coef", ascending=False)
    metrics_df = pd.DataFrame({
        "AUC_mean": [auc_scores.mean()],
        "AUC_std": [auc_scores.std()],
        "ACC_mean": [acc_scores.mean()],
        "ACC_std": [acc_scores.std()],
    })
    return coefs, metrics_df


def generate_plots(df: pd.DataFrame, bins: int, out_dir: Path, save: bool):
    mask1 = df["label_int"] == 1
    mask0 = df["label_int"] == 0
    figs_dir = out_dir / "figures"
    if save:
        figs_dir.mkdir(parents=True, exist_ok=True)

    if set(["renyi_diff", "entropy_diff"]).issubset(df.columns):
        plt.figure()
        plt.scatter(df.loc[mask1, "renyi_diff"], df.loc[mask1, "entropy_diff"], label="baseline_better", alpha=0.5, s=8)
        plt.scatter(df.loc[mask0, "renyi_diff"], df.loc[mask0, "entropy_diff"], label="alt_better", alpha=0.5, s=8)
        plt.xlabel("renyi_diff (baseline - alternative)")
        plt.ylabel("entropy_diff (baseline - alternative)")
        plt.title("renyi vs entropy differences by label")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(figs_dir / "scatter_renyi_vs_entropy.png", dpi=150)
            plt.close()
        else:
            plt.show()

    if "entropy_diff" in df.columns:
        plt.figure()
        plt.hist(df.loc[mask1, "entropy_diff"].dropna(), bins=bins, alpha=0.6, label="baseline_better")
        plt.hist(df.loc[mask0, "entropy_diff"].dropna(), bins=bins, alpha=0.6, label="alt_better")
        plt.xlabel("entropy_diff")
        plt.ylabel("Count")
        plt.title("Distribution of entropy_diff by label")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(figs_dir / "hist_entropy_diff.png", dpi=150)
            plt.close()
        else:
            plt.show()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("[info] Loading dataset...")
    df = load_dataset(args.input_csv, args.label_col)

    if not args.quiet:
        print("[info] Parsing token lists...")
    df["tokens_baseline_list"] = df["tokens_baseline"].apply(parse_tokens)
    df["tokens_alternative_list"] = df["tokens_alternative"].apply(parse_tokens)
    df["len_baseline"] = df["tokens_baseline_list"].apply(len)
    df["len_alternative"] = df["tokens_alternative_list"].apply(len)
    df["len_diff"] = df["len_baseline"] - df["len_alternative"]

    add_metric_diffs(df, args.metrics)
    cat_diff_cols = add_token_category_features(df)

    # Create filtered dataframe excluding ties for analyses (except subject stats)
    df_analysis = df[df["label_int"] != -1].copy()

    class_balance = compute_class_balance(df_analysis)
    display_dataframe_to_user("Class Balance (baseline vs alternative)", class_balance.reset_index().rename(columns={"index": "group"}))
    class_balance.to_csv(out_dir / "class_balance.csv", index=True)

    summary_by_label = summarize_core_metrics(df_analysis, args.metrics)
    display_dataframe_to_user("Summary by label (means of core metrics)", summary_by_label.round(4).reset_index().rename(columns={"label_int": "group"}))
    summary_by_label.to_csv(out_dir / "summary_by_label.csv", index=True)

    diff_cols = [f"{m}_diff" for m in args.metrics if f"{m}_diff" in df_analysis.columns] + ["len_diff"] + [c for c in cat_diff_cols if c in df_analysis.columns]
    diff_cols = [c for c in diff_cols if c in df_analysis.columns]
    diff_cols = list(dict.fromkeys(diff_cols))
    ttest_table = compute_welch_tests(df_analysis, diff_cols)
    display_dataframe_to_user("Welch t-tests: baseline minus alternative (grouped by label)", ttest_table.round(6).reset_index().rename(columns={"index": "feature"}))
    ttest_table.to_csv(out_dir / "welch_ttests.csv", index=False)

    default_feature_cols = diff_cols
    if args.list_features:
        print("Default feature columns (in order):")
        for f in default_feature_cols:
            print(f" - {f}")
        print("(Use --features to override this selection.)")
        sys.exit(0)

    if args.features:
        requested = list(dict.fromkeys(args.features))
        feature_cols = [c for c in requested if c in df.columns]
        missing = [c for c in requested if c not in df.columns]
        if missing and not args.quiet:
            print(f"[warn] Requested features not found and will be skipped: {missing}")
        if not feature_cols:
            raise ValueError("None of the requested --features exist in the dataframe.")
    else:
        feature_cols = default_feature_cols

    coefs, metrics_df = run_logreg(df_analysis, feature_cols, args.n_splits, args.random_state, args.max_iter)
    display_dataframe_to_user("Logistic regression CV metrics", metrics_df.round(4))
    display_dataframe_to_user("Logistic regression coefficients (higher => baseline more likely to be right)", coefs.reset_index(drop=True).round(5))
    coefs.to_csv(out_dir / "feature_importance_logreg.csv", index=False)
    metrics_df.to_csv(out_dir / "model_cv_metrics.csv", index=False)

    if not args.no_plots:
        if not args.quiet:
            print("[info] Generating plots...")
        generate_plots(df_analysis, args.bins, out_dir, save=args.save_plots)

    if not args.quiet:
        print("Artifacts saved in:", out_dir)


if __name__ == "__main__":
    main()
