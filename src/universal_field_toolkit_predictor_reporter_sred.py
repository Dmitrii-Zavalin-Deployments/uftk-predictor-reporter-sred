import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil

# ---------------------------------------------------------
# Unified working directory
# ---------------------------------------------------------

WORKING_DIR = "/data/testing-input-output"
CSV_PATH = os.path.join(WORKING_DIR, "field_data.csv")

# ---------------------------------------------------------
# Prediction rules
# ---------------------------------------------------------

def apply_rules(row):
    if row["brightness"] > 180 and row["Texture_Class"] == "slushy":
        return "fast", "Predicted Fast Melt BECAUSE Brightness > 180 AND Texture = Slushy"

    if row["brightness"] < 100 and row["Texture_Class"] == "icy":
        return "slow", "Predicted Slow Melt BECAUSE Brightness < 100 AND Texture = Icy"

    return "moderate", "Predicted Moderate Melt BECAUSE conditions are mixed"

def fit_regression(df):
    # Optional regression model placeholder
    return None

# ---------------------------------------------------------
# Per-photo plotting
# ---------------------------------------------------------

def generate_plot(df_row, base_name):
    plt.scatter([df_row["brightness"]], [df_row["shadow_intensity"]])
    plt.xlabel("Brightness")
    plt.ylabel("Shadow Intensity")

    plot_path = os.path.join(WORKING_DIR, f"{base_name}_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"✓ Created {plot_path}")
    return plot_path

# ---------------------------------------------------------
# Per-photo markdown report
# ---------------------------------------------------------

def generate_report(df_row, base_name):
    report_path = os.path.join(WORKING_DIR, f"{base_name}_report.md")

    with open(report_path, "w") as f:
        f.write(f"# Prediction Report for {base_name}\n\n")
        f.write(f"- Observation ID: {df_row['Observation_ID']}\n")
        f.write(f"- Predicted Melt Rate: {df_row['Predicted_Melt_Rate']}\n")
        f.write(f"- Explanation: {df_row['Explanation']}\n")

    print(f"✓ Created {report_path}")
    return report_path

# ---------------------------------------------------------
# Copy correlated → predicted
# ---------------------------------------------------------

def generate_predicted_image(correlated_path, base_name):
    dst = os.path.join(WORKING_DIR, f"{base_name}_predicted.jpg")
    shutil.copy2(correlated_path, dst)
    print(f"✓ Created {dst}")
    return dst

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-regression", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(CSV_PATH):
        print("❌ ERROR: field_data.csv not found.")
        return

    df = pd.read_csv(CSV_PATH)

    # Prepare prediction columns
    df["Predicted_Melt_Rate"] = ""
    df["Explanation"] = ""

    # Process each correlated image
    for filename in os.listdir(WORKING_DIR):
        if filename.endswith("_correlated.jpg"):
            correlated_path = os.path.join(WORKING_DIR, filename)
            base_name = filename.replace("_correlated.jpg", "")

            # Find matching CSV row
            row_index = df.index[df["Photo_Filename"] == filename.replace("_correlated.jpg", "_ingested.jpg")]

            if len(row_index) == 0:
                print(f"⚠️ No CSV match for {filename}")
                continue

            idx = row_index[0]
            row = df.loc[idx]

            # Apply prediction rules
            rate, explanation = apply_rules(row)
            df.at[idx, "Predicted_Melt_Rate"] = rate
            df.at[idx, "Explanation"] = explanation

            # Generate outputs
            generate_predicted_image(correlated_path, base_name)
            generate_plot(row, base_name)
            generate_report(row, base_name)

    # Optional regression
    if args.enable_regression:
        fit_regression(df)

    # Save updated CSV
    df.to_csv(CSV_PATH, index=False)
    print("✓ Updated field_data.csv with predictions")

    print("🎉 Predictor‑Reporter complete. Outputs written to /data/testing-input-output")

if __name__ == "__main__":
    main()



