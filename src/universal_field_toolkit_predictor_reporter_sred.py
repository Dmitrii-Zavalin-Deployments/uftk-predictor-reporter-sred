import argparse
import pandas as pd
import matplotlib.pyplot as plt

def apply_rules(row):
    if row["brightness"] > 180 and row["texture_class"] == "slushy":
        return "fast", "Predicted Fast Melt BECAUSE Brightness > 180 AND Texture = Slushy"
    if row["brightness"] < 100 and row["texture_class"] == "icy":
        return "slow", "Predicted Slow Melt BECAUSE Brightness < 100 AND Texture = Icy"
    return "moderate", "Predicted Moderate Melt BECAUSE conditions are mixed"

def fit_regression(df):
    # Optional tiny regression model
    return None

def plot_relationships(df):
    plt.scatter(df["brightness"], df["shadow_intensity"])
    plt.xlabel("Brightness")
    plt.ylabel("Shadow Intensity")
    plt.savefig("plots/brightness_vs_shadow.png")

def generate_report(df):
    with open("reports/predictions.md", "w") as f:
        f.write("# Prediction Report\n\n")
        f.write(df[["Observation_ID", "Predicted_Melt_Rate", "Explanation"]].to_markdown())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-regression", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv("field_data.csv")

    df["Predicted_Melt_Rate"] = ""
    df["Explanation"] = ""

    for idx, row in df.iterrows():
        rate, explanation = apply_rules(row)
        df.at[idx, "Predicted_Melt_Rate"] = rate
        df.at[idx, "Explanation"] = explanation

    if args.enable_regression:
        fit_regression(df)

    plot_relationships(df)
    generate_report(df)
    df.to_csv("field_data.csv", index=False)

if __name__ == "__main__":
    main()



