import os, sys, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import product
from collections import Counter

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, auc, accuracy_score
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)


FASTA_FILES = {
    "Alpha":   "alpha.fasta",
    "Beta":    "beta.fasta",
    "Gamma":   "gamma.fasta",
    "Delta":   "delta.fasta",
    "Omicron": "omicron.fasta",
}

K = 4           # k-mer size
TEST_SIZE = 0.2 
CV_FOLDS  = 5   

def kmer_freq(seq: str, k: int) -> dict:
    seq = seq.upper().replace("-", "").replace("N", "")
    all_kmers = ["".join(p) for p in product("ACGT", repeat=k)]
    counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
    total = sum(counts.values()) or 1
    return {km: counts.get(km, 0) / total for km in all_kmers}


def load_fasta(path: str, label: str, k: int) -> list[dict]:
    try:
        from Bio import SeqIO
    except ImportError:
        sys.exit("Missing biopython. Run: pip install biopython")

    rows = []
    skipped = 0
    for rec in SeqIO.parse(path, "fasta"):
        seq = str(rec.seq)
        if len(seq) < k * 2:
            skipped += 1
            continue
        row = kmer_freq(seq, k)
        row["strain"] = label
        row["_seq_id"] = rec.id          # keep id so we can pick mystery seqs
        rows.append(row)

    print(f"  {label:10s}: {len(rows)} sequences loaded"
          + (f"  ({skipped} skipped — too short)" if skipped else ""))
    return rows


def build_dataset(fasta_files: dict, k: int) -> pd.DataFrame:
    
    print(f"\nExtracting k-mer features (k={k}) from {len(fasta_files)} files...")
    all_rows = []
    for label, path in fasta_files.items():
        if not os.path.exists(path):
            sys.exit(f"\nFile not found: '{path}'\nUpdate FASTA_FILES at the top of this script.")
        all_rows.extend(load_fasta(path, label, k))

    df = pd.DataFrame(all_rows).fillna(0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nDataset ready: {len(df)} sequences × {df.shape[1]-2} features")
    print("Strain counts:")
    for strain, count in df["strain"].value_counts().items():
        print(f"  {strain}: {count}")
    return df

def get_models():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_features="sqrt",
            min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "Naive Bayes": MultinomialNB(),   
    }


def train_and_evaluate(df: pd.DataFrame):

    le = LabelEncoder()
    y = le.fit_transform(df["strain"])
    X = df.drop(columns=["strain", "_seq_id"]).values
    class_names = le.classes_

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )

    models = get_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    results = {}

    print(f"\n{'='*55}")
    print(f"  TRAINING & CROSS-VALIDATION  ({CV_FOLDS}-fold)")
    print(f"{'='*55}")

    for name, model in models.items():
        print(f"\n  {name}...")

        # Cross-validation on training data
        cv_acc       = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_f1        = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro",  n_jobs=-1)
        cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring="precision_macro", n_jobs=-1)
        cv_recall    = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall_macro",    n_jobs=-1)

        # Final fit on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        holdout_acc       = accuracy_score(y_test, y_pred)
        holdout_f1        = f1_score(y_test, y_pred, average="macro")
        holdout_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        holdout_recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
        holdout_auc       = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

        results[name] = {
            "model":           model,
            "cv_acc_mean":     cv_acc.mean(),
            "cv_acc_std":      cv_acc.std(),
            "cv_f1_mean":      cv_f1.mean(),
            "cv_f1_std":       cv_f1.std(),
            "cv_precision_mean": cv_precision.mean(),
            "cv_precision_std":  cv_precision.std(),
            "cv_recall_mean":  cv_recall.mean(),
            "cv_recall_std":   cv_recall.std(),
            "holdout_acc":     holdout_acc,
            "holdout_f1":      holdout_f1,
            "holdout_precision": holdout_precision,
            "holdout_recall":  holdout_recall,
            "holdout_auc":     holdout_auc,
            "y_pred":          y_pred,
            "y_prob":          y_prob,
        }

        print(f"    CV Accuracy    : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
        print(f"    CV F1-macro    : {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
        print(f"    CV Precision   : {cv_precision.mean():.3f} ± {cv_precision.std():.3f}")
        print(f"    CV Recall      : {cv_recall.mean():.3f} ± {cv_recall.std():.3f}")
        print(f"    Holdout Acc    : {holdout_acc:.3f}")
        print(f"    Holdout F1     : {holdout_f1:.3f}")
        print(f"    Holdout Precision: {holdout_precision:.3f}")
        print(f"    Holdout Recall : {holdout_recall:.3f}")
        print(f"    Holdout AUC    : {holdout_auc:.3f}")

    return results, scaler, le, X_train, X_test, y_train, y_test, class_names

COLORS = {
    "XGBoost":     "#378ADD",
    "Extra Trees": "#1D9E75",
    "Naive Bayes": "#D4537E",
}
STRAIN_COLORS = ["#378ADD", "#1D9E75", "#D4537E", "#BA7517", "#8B5CF6"]


def plot_results(results, y_test, class_names):
    n_models = len(results)
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0F1117")

    gs = gridspec.GridSpec(
        3, n_models,
        figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.93, bottom=0.06, left=0.06, right=0.97
    )

    fig.suptitle("COVID Strain Classification — Model Comparison",
                 fontsize=15, color="white", fontweight="bold", y=0.98)

    ax_bar = fig.add_subplot(gs[0, :])
    ax_bar.set_facecolor("#1A1D27")
    labels  = ["Accuracy", "Precision", "Recall"]
    x = np.arange(len(labels))
    width = 0.25
    for idx, (name, res) in enumerate(results.items()):
        vals = [res["holdout_acc"], res["holdout_precision"], res["holdout_recall"]]
        bars = ax_bar.bar(x + idx * width, vals, width, label=name,
                          color=COLORS[name], alpha=0.88, edgecolor="none")
        for bar, val in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8, color="white", fontweight="bold")
    ax_bar.set_xticks(x + width)
    ax_bar.set_xticklabels(labels, color="white", fontsize=11)
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("Score", color="white")
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_color("#333")
    ax_bar.legend(fontsize=9, facecolor="#1A1D27", labelcolor="white", framealpha=0.8)
    ax_bar.set_title("Holdout Accuracy / Precision / Recall Across All Models", color="white", fontsize=11)
    for spine in ax_bar.spines.values():
        spine.set_color("#333")


    for col, (name, res) in enumerate(results.items()):
        color = COLORS[name]

        # Confusion matrix
        ax_cm = fig.add_subplot(gs[1, col])
        ax_cm.set_facecolor("#1A1D27")
        cm = confusion_matrix(y_test, res["y_pred"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=cm, fmt="d", ax=ax_cm,
                    xticklabels=class_names, yticklabels=class_names,
                    cmap="Blues", cbar=False, linewidths=0.5, linecolor="#0F1117",
                    annot_kws={"size": 9, "color": "white"})
        ax_cm.set_title(f"{name}\nConfusion Matrix", color="white", fontsize=10)
        ax_cm.set_xlabel("Predicted", color="#aaa", fontsize=8)
        ax_cm.set_ylabel("Actual", color="#aaa", fontsize=8)
        ax_cm.tick_params(colors="white", labelsize=8)


    plt.savefig("covid_results.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\nSaved: covid_results.png")
    plt.show()



def pick_mystery_sequence(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Pick one random sequence from the dataset to act as the mystery strain.
    Returns (seq_id, true_strain, sequence_features_as_row).
    """
    row = df.sample(1, random_state=random.randint(0, 9999)).iloc[0]
    return row["_seq_id"], row["strain"], row


def normalize_strain_label(label: str) -> str:
    label = label.strip().lower()
    if label in {"alpha", "beta", "delta", "gamma", "omicron"}:
        return label.capitalize()
    return label


def load_mystery_from_file(mystery_file: str, k: int, csv_mapping_file: str = None) -> tuple[str, str, pd.Series]:

    from Bio import SeqIO
    
    if not os.path.exists(mystery_file):
        sys.exit(f"Mystery file not found: '{mystery_file}'")
    
    csv_strain_map = {}
    if csv_mapping_file and os.path.exists(csv_mapping_file):
        with open(csv_mapping_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "," not in line:
                    continue
                parts = line.split(",", 1)
                seq_id = parts[0].strip()
                strain = normalize_strain_label(parts[1].strip())
                csv_strain_map[seq_id] = strain
        print(f"Loaded {len(csv_strain_map)} sequences from {csv_mapping_file}")
    
    # Load all valid sequences from the mystery file
    valid_sequences = []
    for rec in SeqIO.parse(mystery_file, "fasta"):
        seq = str(rec.seq)
        if len(seq) < k * 2:
            print(f"Sequence too short, skipping: {rec.id}")
            continue

        if csv_strain_map and rec.id not in csv_strain_map:
            continue
        
        valid_sequences.append((rec.id, seq))
    
    if not valid_sequences:
        if csv_strain_map:
            sys.exit(f"No sequences from {csv_mapping_file} found in '{mystery_file}'")
        else:
            sys.exit(f"No valid sequences found in '{mystery_file}'")
    
    # Randomly pick one sequence
    seq_id, seq = random.choice(valid_sequences)
    row = kmer_freq(seq, k)
    row["_seq_id"] = seq_id
    
    # Use strain from CSV if available, otherwise "Unknown"
    if csv_strain_map and seq_id in csv_strain_map:
        row["strain"] = csv_strain_map[seq_id]
    else:
        row["strain"] = "Unknown"
    
    return seq_id, row["strain"], pd.Series(row)


def load_all_mystery_sequences(mystery_file: str, k: int, csv_mapping_file: str) -> pd.DataFrame:

    from Bio import SeqIO
    
    if not os.path.exists(mystery_file):
        sys.exit(f"Mystery file not found: '{mystery_file}'")
    
    if not os.path.exists(csv_mapping_file):
        sys.exit(f"CSV mapping file not found: '{csv_mapping_file}'")
    
    # Load CSV mapping
    csv_strain_map = {}
    with open(csv_mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            parts = line.split(",", 1)
            seq_id = parts[0].strip()
            strain = normalize_strain_label(parts[1].strip())
            csv_strain_map[seq_id] = strain
    
    print(f"\nLoading mystery sequences for batch testing...")
    print(f"  {csv_mapping_file}: {len(csv_strain_map)} sequences")
    
    # Load all sequences from FASTA that are in CSV
    all_rows = []
    skipped = 0
    for rec in SeqIO.parse(mystery_file, "fasta"):
        seq = str(rec.seq)
        
        # Skip if not in CSV mapping
        if rec.id not in csv_strain_map:
            continue
        
        # Skip if too short
        if len(seq) < k * 2:
            skipped += 1
            continue
        
        row = kmer_freq(seq, k)
        row["strain"] = csv_strain_map[rec.id]
        row["_seq_id"] = rec.id
        all_rows.append(row)
    
    print(f"  Loaded: {len(all_rows)} sequences" + (f" ({skipped} skipped — too short)" if skipped else ""))
    
    if not all_rows:
        sys.exit("No sequences loaded for mystery batch testing!")
    
    return pd.DataFrame(all_rows).fillna(0)


def evaluate_mystery_batch(mystery_df: pd.DataFrame, results: dict,
                           scaler, le, class_names) -> dict:

    X_mystery = mystery_df.drop(columns=["strain", "_seq_id"]).values
    X_scaled = scaler.transform(X_mystery)
    y_true = le.transform(mystery_df["strain"])
    
    print(f"\n{'='*55}")
    print(f"  BATCH MYSTERY SEQUENCE TESTING")
    print(f"{'='*55}")
    print(f"  Testing on {len(mystery_df)} sequences from {mystery_df['strain'].nunique()} strains\n")
    
    batch_results = {}
    for name, res in results.items():
        model = res["model"]
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except:
            auc = 0.0
        
        batch_results[name] = {
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "auc": auc,
            "y_pred": y_pred,
            "y_true": y_true,
        }
        
        print(f"  {name:15s}  Acc={acc:.3f}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  AUC={auc:.3f}")
    
    return batch_results


def identify_mystery(mystery_row: pd.Series, results: dict,
                     scaler, le, class_names) -> pd.DataFrame:

    X_mystery = mystery_row.drop(["strain", "_seq_id"]).values.reshape(1, -1)
    X_scaled  = scaler.transform(X_mystery)

    print(f"\n{'='*55}")
    print(f"  MYSTERY STRAIN IDENTIFICATION")
    print(f"{'='*55}")
    print(f"  Sequence ID  : {mystery_row['_seq_id']}")
    print(f"  True strain  : {mystery_row['strain']}  (hidden from models)\n")

    rows = []
    for name, res in results.items():
        model = res["model"]
        pred_idx = model.predict(X_scaled)[0]
        probs    = model.predict_proba(X_scaled)[0]
        pred_strain = le.inverse_transform([pred_idx])[0]
        confidence  = probs[pred_idx]
        correct     = "✓" if pred_strain == mystery_row["strain"] else "✗"

        print(f"  {name:15s} → {pred_strain:10s}  "
              f"confidence={confidence:.1%}  {correct}")

        prob_dict = {cls: f"{p:.1%}" for cls, p in zip(class_names, probs)}
        rows.append({
            "Model":      name,
            "Prediction": pred_strain,
            "Confidence": f"{confidence:.1%}",
            "Correct":    correct,
            **prob_dict,
        })

    print(f"\n  True answer: {mystery_row['strain']}")
    df_out = pd.DataFrame(rows).set_index("Model")
    print("\n  Full probability breakdown:")
    print(df_out.to_string())
    return df_out


def plot_mystery(mystery_results: pd.DataFrame, true_strain: str):
    """Bar chart showing each model's probability distribution over strains."""
    strain_cols = [c for c in mystery_results.columns
                   if c not in ["Prediction", "Confidence", "Correct"]]
    
    fig, axes = plt.subplots(1, len(mystery_results), figsize=(15, 4), sharey=True)
    fig.patch.set_facecolor("#0F1117")
    fig.suptitle(f"Mystery Strain Probabilities  (true strain: {true_strain})",
                 color="white", fontsize=13, fontweight="bold")

    for ax, (model_name, row) in zip(axes, mystery_results.iterrows()):
        ax.set_facecolor("#1A1D27")
        probs  = [float(row[s].strip("%")) / 100 for s in strain_cols]
        colors = [("#4CAF50" if s == true_strain else
                   ("#EF5350" if s == row["Prediction"] and s != true_strain
                    else STRAIN_COLORS[i % len(STRAIN_COLORS)]))
                  for i, s in enumerate(strain_cols)]
        bars = ax.barh(strain_cols, probs, color=colors, alpha=0.88, edgecolor="none")
        for bar, p in zip(bars, probs):
            ax.text(max(p + 0.01, 0.03), bar.get_y() + bar.get_height()/2,
                    f"{p:.1%}", va="center", fontsize=9, color="white", fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.set_title(f"{model_name}\n→ {row['Prediction']} {row['Correct']}",
                     color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values(): spine.set_color("#333")

    plt.tight_layout()
    plt.savefig("mystery_result.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\nSaved: mystery_result.png")
    plt.show()


if __name__ == "__main__":

    df = build_dataset(FASTA_FILES, k=K)
    #df = df.groupby("strain").sample(n=100, random_state=42).reset_index(drop=True)

    results, scaler, le, X_train, X_test, y_train, y_test, class_names = \
        train_and_evaluate(df)

    plot_results(results, y_test, class_names)

    #seq_id, true_strain, mystery_row = pick_mystery_sequence(df)

    seq_id, true_strain, mystery_row = load_mystery_from_file("RandomStrain.fasta", k=K, csv_mapping_file="strainnames_cleaned.csv")
    
    mystery_results = identify_mystery(mystery_row, results, scaler, le, class_names)
    plot_mystery(mystery_results, true_strain)

    mystery_batch_df = load_all_mystery_sequences("RandomStrain.fasta", k=K, csv_mapping_file="strainnames_cleaned.csv")
    batch_results = evaluate_mystery_batch(mystery_batch_df, results, scaler, le, class_names)

    print(f"\n{'='*55}")
    print("  FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"\n  Holdout Test Set:")
    for name, res in results.items():
        print(f"    {name:15s}  "
              f"Acc={res['holdout_acc']:.3f}  "
              f"F1={res['holdout_f1']:.3f}  "
              f"Prec={res['holdout_precision']:.3f}  "
              f"Rec={res['holdout_recall']:.3f}  "
              f"AUC={res['holdout_auc']:.3f}")
    
    print(f"\n  Batch Mystery Test Set:")
    for name, batch_res in batch_results.items():
        print(f"    {name:15s}  "
              f"Acc={batch_res['accuracy']:.3f}  "
              f"F1={batch_res['f1']:.3f}  "
              f"Prec={batch_res['precision']:.3f}  "
              f"Rec={batch_res['recall']:.3f}  "
              f"AUC={batch_res['auc']:.3f}")
    
    print(f"\n  Output files:")
    print(f"    covid_results.png   — accuracy, precision, recall, confusion matrices, ROC curves")
    print(f"    mystery_result.png  — mystery strain probability breakdown")
