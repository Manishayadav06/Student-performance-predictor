"""
Student Performance Predictor
Author: S Harshitha Reddy
Description: ML-based regression system to predict student exam scores
             using Random Forest. Trained on 6,607 student records.
Dataset: StudentPerformanceFactors.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error
)

warnings.filterwarnings("ignore")

# ── 1. LOAD & INSPECT ──────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")
    return df


# ── 2. PREPROCESSING ───────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    df = df.dropna().copy()

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop("Exam_Score", axis=1)
    y = df["Exam_Score"]
    feature_names = list(X.columns)

    print(f"Features: {feature_names}")
    print(f"Target range: {y.min()} – {y.max()}, mean: {y.mean():.2f}\n")

    return X, y, feature_names, encoders


# ── 3. TRAIN & EVALUATE ────────────────────────────────────────────────────────

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression":    LinearRegression(),
        "Ridge Regression":     Ridge(alpha=1.0),
        "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    print("=" * 55)
    print(f"{'Model':<25} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("=" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "cv_r2": cv_r2,
        }
        print(f"{name:<25} {r2:>8.4f} {mae:>8.4f} {rmse:>8.4f}")

    print("=" * 55)
    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\nBest model: {best_name} (R² = {results[best_name]['r2']:.4f})\n")

    return results, best_name, X_test, y_test


# ── 4. VISUALISATIONS ──────────────────────────────────────────────────────────

def plot_all(df_raw, results, best_name, feature_names, X_test, y_test, out_path):
    best   = results[best_name]
    model  = best["model"]
    y_pred = best["y_pred"]

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor("#F7F7F5")

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.55,
        wspace=0.38,
        top=0.93, bottom=0.04,
        left=0.07, right=0.97,
    )

    ACCENT   = "#5040B4"
    ACCENT2  = "#1D9E75"
    ACCENT3  = "#E24B4A"
    NEUTRAL  = "#888780"
    BG_CARD  = "#FFFFFF"
    GRID_CLR = "#ECECEA"

    def card_ax(ax, title=""):
        ax.set_facecolor(BG_CARD)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
            spine.set_linewidth(0.8)
        ax.grid(color=GRID_CLR, linewidth=0.6, zorder=0)
        ax.tick_params(colors="#555", labelsize=8)
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold",
                         color="#1A1A1A", pad=8)

    # ── (0,0..2) Header banner ─────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(ACCENT)
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis("off")
    ax_title.text(0.5, 0.70, "Student Performance Predictor",
                  ha="center", va="center", fontsize=22, fontweight="bold",
                  color="white", transform=ax_title.transAxes)
    ax_title.text(0.5, 0.26,
                  f"Random Forest Regression  ·  6,607 students  ·  19 features  ·  "
                  f"R² = {best['r2']:.4f}  ·  MAE = {best['mae']:.4f}  ·  RMSE = {best['rmse']:.4f}",
                  ha="center", va="center", fontsize=11, color="#D8D4F8",
                  transform=ax_title.transAxes)
    ax_title.set_position([0.07, 0.88, 0.90, 0.062])

    # ── (1,0) Exam score distribution ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    card_ax(ax1, "Exam score distribution")
    ax1.hist(df_raw["Exam_Score"].dropna(), bins=25, color=ACCENT,
             edgecolor="white", linewidth=0.4, zorder=3)
    ax1.axvline(df_raw["Exam_Score"].mean(), color=ACCENT3, lw=1.4,
                linestyle="--", label=f"Mean: {df_raw['Exam_Score'].mean():.1f}")
    ax1.set_xlabel("Exam score", fontsize=8)
    ax1.set_ylabel("Count", fontsize=8)
    ax1.legend(fontsize=8)

    # ── (1,1) Actual vs predicted ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    card_ax(ax2, f"Actual vs predicted  ({best_name})")
    ax2.scatter(y_test, y_pred, alpha=0.35, s=12, color=ACCENT, zorder=3)
    lims = [min(y_test.min(), y_pred.min()) - 1,
            max(y_test.max(), y_pred.max()) + 1]
    ax2.plot(lims, lims, color=ACCENT3, lw=1.2, linestyle="--", label="Perfect fit")
    ax2.set_xlabel("Actual score", fontsize=8)
    ax2.set_ylabel("Predicted score", fontsize=8)
    ax2.legend(fontsize=8)

    # ── (1,2) Residuals ────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    card_ax(ax3, "Residuals distribution")
    residuals = y_test.values - y_pred
    ax3.hist(residuals, bins=30, color=ACCENT2, edgecolor="white",
             linewidth=0.4, zorder=3)
    ax3.axvline(0, color=ACCENT3, lw=1.2, linestyle="--")
    ax3.set_xlabel("Residual (actual − predicted)", fontsize=8)
    ax3.set_ylabel("Count", fontsize=8)

    # ── (2,0) Feature importance ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    card_ax(ax4, "Feature importance (Random Forest)")
    rf_model = results["Random Forest"]["model"]
    fi = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values()
    colors_fi = [ACCENT if v == fi.max() else
                 (ACCENT2 if v >= fi.quantile(0.75) else NEUTRAL) for v in fi]
    fi.plot(kind="barh", ax=ax4, color=colors_fi, edgecolor="none", zorder=3)
    ax4.set_xlabel("Importance", fontsize=8)
    ax4.tick_params(axis="y", labelsize=7)

    # ── (2,1) Model comparison bar chart ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    card_ax(ax5, "Model comparison (R² score)")
    model_names = list(results.keys())
    r2_vals = [results[m]["r2"] for m in model_names]
    bar_colors = [ACCENT if m == best_name else "#C8C4F0" for m in model_names]
    bars = ax5.barh(model_names, r2_vals, color=bar_colors, edgecolor="none", zorder=3)
    for bar, val in zip(bars, r2_vals):
        ax5.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=8, color="#333")
    ax5.set_xlabel("R² score", fontsize=8)
    ax5.set_xlim(0, max(r2_vals) * 1.15)

    # ── (2,2) Attendance vs Exam score (scatter) ───────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    card_ax(ax6, "Attendance vs exam score")
    sample = df_raw.dropna().sample(min(800, len(df_raw)), random_state=1)
    ax6.scatter(sample["Attendance"], sample["Exam_Score"],
                alpha=0.3, s=10, color=ACCENT, zorder=3)
    m, b = np.polyfit(sample["Attendance"], sample["Exam_Score"], 1)
    xs = np.linspace(sample["Attendance"].min(), sample["Attendance"].max(), 100)
    ax6.plot(xs, m * xs + b, color=ACCENT3, lw=1.4, label="Trend")
    ax6.set_xlabel("Attendance (%)", fontsize=8)
    ax6.set_ylabel("Exam score", fontsize=8)
    ax6.legend(fontsize=8)

    # ── (3,0) Hours studied vs score ──────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    card_ax(ax7, "Hours studied vs exam score")
    ax7.scatter(sample["Hours_Studied"], sample["Exam_Score"],
                alpha=0.3, s=10, color=ACCENT2, zorder=3)
    m2, b2 = np.polyfit(sample["Hours_Studied"], sample["Exam_Score"], 1)
    xs2 = np.linspace(sample["Hours_Studied"].min(), sample["Hours_Studied"].max(), 100)
    ax7.plot(xs2, m2 * xs2 + b2, color=ACCENT3, lw=1.4)
    ax7.set_xlabel("Hours studied / week", fontsize=8)
    ax7.set_ylabel("Exam score", fontsize=8)

    # ── (3,1) Correlation heatmap (top features) ──────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    card_ax(ax8, "Correlation — top 8 features")

    df_encoded = df_raw.dropna().copy()
    for col in df_encoded.select_dtypes(include=["object", "string"]).columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

    top8 = (["Exam_Score"] +
            pd.Series(rf_model.feature_importances_, index=feature_names)
            .nlargest(7).index.tolist())
    corr = df_encoded[top8].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax8, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.4, linecolor=GRID_CLR, annot_kws={"size": 6},
                mask=mask, cbar_kws={"shrink": 0.7})
    ax8.tick_params(labelsize=7)

    # ── (3,2) Predicted score bucket distribution ─────────────────────────────
    ax9 = fig.add_subplot(gs[3, 2])
    card_ax(ax9, "Predicted score distribution")
    bins  = [54, 60, 65, 70, 75, 102]
    lbls  = ["55–60", "61–65", "66–70", "71–75", "76+"]
    y_all = results["Random Forest"]["model"].predict(X_test)
    cnts  = pd.cut(y_all, bins=bins, labels=lbls).value_counts().reindex(lbls)
    bar_c = [ACCENT3, "#EF9F27", ACCENT2, ACCENT, "#8B5CF6"]
    cnts.plot(kind="bar", ax=ax9, color=bar_c, edgecolor="none",
              zorder=3, rot=0)
    ax9.set_xlabel("Score bucket", fontsize=8)
    ax9.set_ylabel("Count", fontsize=8)

    plt.suptitle("")
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Chart saved → {out_path}")


# ── 5. PREDICTION FUNCTION ─────────────────────────────────────────────────────

def predict_score(model, encoders: dict, feature_names: list, student: dict) -> float:
    """
    Predict a single student's exam score.

    Parameters
    ----------
    model      : trained sklearn model
    encoders   : dict of LabelEncoders keyed by column name
    feature_names: list of feature column names (in order)
    student    : dict with raw (un-encoded) feature values

    Returns
    -------
    predicted exam score (float)
    """
    row = {}
    for feat in feature_names:
        val = student[feat]
        if feat in encoders:
            val = encoders[feat].transform([str(val)])[0]
        row[feat] = val

    X_new = pd.DataFrame([row])[feature_names]
    return round(model.predict(X_new)[0], 2)


# ── 6. MAIN ────────────────────────────────────────────────────────────────────

def main():
    DATA_PATH  = "StudentPerformanceFactors.csv"
    CHART_PATH = "student_performance_report.png"

    # Load
    df = load_data(DATA_PATH)

    # Preprocess
    X, y, feature_names, encoders = preprocess(df)

    # Train all models & evaluate
    results, best_name, X_test, y_test = train_and_evaluate(X, y)

    # Visualise → save PNG
    plot_all(df, results, best_name, feature_names, X_test, y_test, CHART_PATH)

    # ── Demo prediction ────────────────────────────────────────────────────────
    sample_student = {
        "Hours_Studied":           25,
        "Attendance":              92,
        "Parental_Involvement":    "High",
        "Access_to_Resources":     "High",
        "Extracurricular_Activities": "Yes",
        "Sleep_Hours":             8,
        "Previous_Scores":         85,
        "Motivation_Level":        "High",
        "Internet_Access":         "Yes",
        "Tutoring_Sessions":       2,
        "Family_Income":           "Medium",
        "Teacher_Quality":         "High",
        "School_Type":             "Public",
        "Peer_Influence":          "Positive",
        "Physical_Activity":       3,
        "Learning_Disabilities":   "No",
        "Parental_Education_Level":"Postgraduate",
        "Distance_from_Home":      "Near",
        "Gender":                  "Female",
    }

    best_model = results[best_name]["model"]
    score = predict_score(best_model, encoders, feature_names, sample_student)
    print(f"\nSample prediction for demo student: {score} / 100")
    print(f"  Attendance: {sample_student['Attendance']}%   "
          f"Hours/week: {sample_student['Hours_Studied']}   "
          f"Prev score: {sample_student['Previous_Scores']}")


if __name__ == "__main__":
    main()
