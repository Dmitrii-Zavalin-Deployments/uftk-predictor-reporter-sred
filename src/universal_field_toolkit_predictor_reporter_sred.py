import argparse
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# Unified working directory
# ---------------------------------------------------------
WORKING_DIR = os.environ.get("WORKING_DIR", "/data/testing-input-output")
CSV_PATH = os.path.join(WORKING_DIR, "field_data.csv")


# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------
# Apply rule-based prediction
# ---------------------------------------------------------
def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule-based prediction to each row.

    Current texture classes from analyzer: "smooth", "grainy".
    Future extension: "slushy", "icy" once classifier is expanded.
    """
    df["Predicted_Change_Flag"] = 0  # Default: no strong change signal
    df["Predicted_Melt_Rate"] = "moderate"
    df["Explanation"] = "Predicted Moderate Melt BECAUSE conditions are mixed and no strong rule was triggered"

    # Rule A (current data): bright + grainy → faster melt (loose proxy for rough/wet snow)
    mask = (df.get("Brightness", 0) > 180) & (df.get("Texture_Class", "") == "grainy")
    df.loc[mask, "Predicted_Change_Flag"] = 1
    df.loc[mask, "Predicted_Melt_Rate"] = "fast"
    df.loc[mask, "Explanation"] = (
        "Predicted Fast Melt BECAUSE Brightness > 180 AND Texture_Class = 'grainy'"
    )

    # Rule B (current data): dark + smooth → slower melt (proxy for compact/frozen snow)
    mask = (df.get("Brightness", 0) < 100) & (df.get("Texture_Class", "") == "smooth")
    df.loc[mask, "Predicted_Change_Flag"] = 0
    df.loc[mask, "Predicted_Melt_Rate"] = "slow"
    df.loc[mask, "Explanation"] = (
        "Predicted Slow Melt BECAUSE Brightness < 100 AND Texture_Class = 'smooth'"
    )

    # Reserved rules for future classifier extension
    # (kept for spec alignment, but will not trigger until classes exist):
    #   Brightness > 180 AND Texture_Class = 'slushy' → fast melt
    #   Brightness < 100 AND Texture_Class = 'icy'   → slow melt

    return df


# ---------------------------------------------------------
# Fit optional regression model
# ---------------------------------------------------------
def fit_regression(df: pd.DataFrame):
    """
    Fit tiny linear regression model if labeled data available.

    X: Brightness, Texture, Mean_B
    y: labeled_melt_rate
    """
    if "labeled_melt_rate" not in df.columns or df["labeled_melt_rate"].isnull().all():
        print("⚠️ No labeled melt rate data — skipping regression")
        return None

    X_cols = ["Brightness", "Texture", "Mean_B"]
    y_col = "labeled_melt_rate"

    # Only keep rows where all features + label are present
    cols_needed = [c for c in X_cols + [y_col] if c in df.columns]
    if len(cols_needed) < 4:  # missing one or more required columns
        print("⚠️ Required feature columns for regression are missing — skipping regression")
        return None

    df_reg = df[cols_needed].dropna()
    if len(df_reg) < 2:
        print("⚠️ Insufficient labeled data for regression")
        return None

    X = df_reg[X_cols]
    y = df_reg[y_col]

    model = LinearRegression()
    model.fit(X, y)
    print("✓ Regression model fitted")
    return model


# ---------------------------------------------------------
# Generate predictions (apply rules + optional regression)
# ---------------------------------------------------------
def generate_predictions(df: pd.DataFrame, model=None) -> pd.DataFrame:
    df = apply_rules(df)

    if model is not None:
        X_cols = ["Brightness", "Texture", "Mean_B"]
        # Only rows with full feature set
        existing = [c for c in X_cols if c in df.columns]
        if len(existing) == len(X_cols):
            mask = df[X_cols].notnull().all(axis=1)
            if mask.sum() > 0:
                predicted_melt = model.predict(df.loc[mask, X_cols])
                df.loc[mask, "Predicted_Melt_Rate"] = predicted_melt.round(2)
                df.loc[mask, "Explanation"] = (
                    df.loc[mask, "Explanation"] + " (Regression override applied)"
                )

    return df


# ---------------------------------------------------------
# Generate aggregated plots
# ---------------------------------------------------------
def plot_relationships(df: pd.DataFrame) -> list[str]:
    """
    Generate aggregated plots:
    - Brightness vs Predicted Melt Rate (encoded)
    - Texture vs Predicted Change Flag
    """
    plots: list[str] = []

    # Plot 1: Brightness vs Predicted_Melt_Rate
    if "Brightness" in df.columns and "Predicted_Melt_Rate" in df.columns:
        le = LabelEncoder()
        # to handle numeric melt rates, convert to string before encoding
        melt_series = df["Predicted_Melt_Rate"].astype(str)
        df["Melt_Rate_Num"] = le.fit_transform(melt_series)

        plt.figure(figsize=(8, 6))
        plt.scatter(df["Brightness"], df["Melt_Rate_Num"])
        plt.xlabel("Brightness")
        plt.ylabel("Predicted Melt Rate (Encoded)")
        plt.title("Brightness vs Predicted Melt Rate")
        plot1 = os.path.join(WORKING_DIR, "brightness_vs_melt_rate_plot.png")
        plt.savefig(plot1)
        plt.close()
        plots.append(plot1)

    # Plot 2: Texture vs Predicted_Change_Flag
    if "Texture" in df.columns and "Predicted_Change_Flag" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df["Texture"], df["Predicted_Change_Flag"])
        plt.xlabel("Texture")
        plt.ylabel("Predicted Change Flag")
        plt.title("Texture vs Predicted Change Flag")
        plot2 = os.path.join(WORKING_DIR, "texture_vs_change_flag_plot.png")
        plt.savefig(plot2)
        plt.close()
        plots.append(plot2)

    if plots:
        print(f"✓ Created plots: {', '.join(plots)}")
    else:
        print("⚠️ No plots created (missing required columns)")

    return plots


# ---------------------------------------------------------
# Generate aggregated Markdown report
# ---------------------------------------------------------
def generate_report(df: pd.DataFrame, plots: list[str], regression_status: str) -> str:
    report_path = os.path.join(WORKING_DIR, "prediction_report.md")

    with open(report_path, "w") as f:
        f.write("# Prediction Report\n\n")

        f.write("## Summary of Predictions\n")
        summary_cols = [
            c
            for c in ["Observation_ID", "Predicted_Change_Flag", "Predicted_Melt_Rate", "Explanation"]
            if c in df.columns
        ]
        if summary_cols:
            summary = df[summary_cols].to_markdown(index=False)
            f.write(summary + "\n\n")
        else:
            f.write("_No prediction fields available in DataFrame_\n\n")

        f.write("## Explanation of Rules\n")
        f.write(
            "- If Brightness > 180 AND Texture_Class = 'grainy' → Fast Melt (proxy for bright, rough/wet snow)\n"
        )
        f.write(
            "- If Brightness < 100 AND Texture_Class = 'smooth' → Slow Melt (proxy for dark, compact/frozen snow)\n"
        )
        f.write(
            "- Reserved future rules (classifier upgrade):\n"
            "  - If Brightness > 180 AND Texture_Class = 'slushy' → Fast Melt\n"
            "  - If Brightness < 100 AND Texture_Class = 'icy' → Slow Melt\n"
        )
        f.write("- Otherwise → Moderate Melt\n\n")

        f.write("## Regression Status\n")
        f.write(regression_status + "\n\n")

        f.write("## Visualizations\n")
        if plots:
            for plot in plots:
                f.write(f"![Plot]({os.path.basename(plot)})\n")
        else:
            f.write("_No plots were generated_\n")

    print(f"✓ Created aggregated report: {report_path}")
    return report_path


# ---------------------------------------------------------
# Copy correlated → predicted (per-photo for traceability)
# ---------------------------------------------------------
def generate_predicted_image(correlated_path: str, base_name: str) -> str:
    dst = os.path.join(WORKING_DIR, f"{base_name}_predicted.jpg")
    shutil.copy2(correlated_path, dst)
    print(f"✓ Created {dst}")
    return dst


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-regression", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(CSV_PATH):
        print(f"❌ ERROR: field_data.csv not found at {CSV_PATH}.")
        return

    df = load_csv(CSV_PATH)

    # Optional regression
    model = None
    regression_status = "Regression disabled (no fit performed)."
    if args.enable_regression:
        model = fit_regression(df)
        if model is not None:
            regression_status = "Regression enabled and fitted successfully."
        else:
            regression_status = "Regression enabled but insufficient data to fit."

    # Generate predictions
    df = generate_predictions(df, model)

    # Generate aggregated plots
    plots = plot_relationships(df)

    # Generate aggregated report
    generate_report(df, plots, regression_status)

    # Image chain for traceability: correlated → predicted
    for filename in os.listdir(WORKING_DIR):
        if filename.endswith("_correlated.jpg"):
            correlated_path = os.path.join(WORKING_DIR, filename)
            base_name = filename.replace("_correlated.jpg", "")
            generate_predicted_image(correlated_path, base_name)

    # Save updated CSV
    df.to_csv(CSV_PATH, index=False)
    print("✓ Updated field_data.csv with predictions")
    print(f"🎉 Predictor-Reporter complete. Outputs written to {WORKING_DIR}")


if __name__ == "__main__":
    main()



