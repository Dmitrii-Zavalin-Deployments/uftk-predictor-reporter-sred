import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

import universal_field_toolkit_predictor_reporter_sred as predictor


# =========================================================
# apply_rules
# =========================================================

def test_apply_rules_fast_grainy():
    df = pd.DataFrame([{
        "Brightness": 200,
        "Texture_Class": "grainy"
    }])
    out = predictor.apply_rules(df.copy())
    assert out.loc[0, "Predicted_Melt_Rate"] == "fast"
    assert "Fast Melt" in out.loc[0, "Explanation"]
    assert "Brightness > 180" in out.loc[0, "Explanation"]


def test_apply_rules_slow_smooth():
    df = pd.DataFrame([{
        "Brightness": 50,
        "Texture_Class": "smooth"
    }])
    out = predictor.apply_rules(df.copy())
    assert out.loc[0, "Predicted_Melt_Rate"] == "slow"
    assert "Slow Melt" in out.loc[0, "Explanation"]
    assert "Brightness < 100" in out.loc[0, "Explanation"]


def test_apply_rules_moderate_default_cases():
    cases = [
        {"Brightness": 180, "Texture_Class": "grainy"},
        {"Brightness": 200, "Texture_Class": "smooth"},
        {"Brightness": 50, "Texture_Class": "grainy"},
        {"Brightness": 150, "Texture_Class": "other"},
    ]
    df = pd.DataFrame(cases)
    out = predictor.apply_rules(df.copy())
    for i in range(len(df)):
        assert out.loc[i, "Predicted_Melt_Rate"] == "moderate"
        assert "Moderate Melt" in out.loc[i, "Explanation"]


# =========================================================
# fit_regression
# =========================================================

def test_fit_regression_no_label_column():
    df = pd.DataFrame({"Brightness": [1, 2, 3]})
    assert predictor.fit_regression(df) is None


def test_fit_regression_missing_required_features():
    df = pd.DataFrame({
        "labeled_melt_rate": [1, 2, 3],
        "Brightness": [10, 20, 30],
    })
    assert predictor.fit_regression(df) is None


def test_fit_regression_insufficient_rows():
    df = pd.DataFrame({
        "Brightness": [10],
        "Texture": [100],
        "Mean_B": [50],
        "labeled_melt_rate": [0.5],
    })
    assert predictor.fit_regression(df) is None


def test_fit_regression_success():
    df = pd.DataFrame({
        "Brightness": [10, 20],
        "Texture": [100, 200],
        "Mean_B": [50, 60],
        "labeled_melt_rate": [0.5, 1.0],
    })
    model = predictor.fit_regression(df)
    assert model is not None


# =========================================================
# generate_predictions
# =========================================================

def test_generate_predictions_rule_only():
    df = pd.DataFrame([{
        "Brightness": 200,
        "Texture_Class": "grainy",
        "Texture": 100,
        "Mean_B": 50
    }])
    out = predictor.generate_predictions(df.copy(), model=None)
    assert out.loc[0, "Predicted_Melt_Rate"] == "fast"


def test_generate_predictions_regression_override():
    class DummyModel:
        def predict(self, X):
            return np.array([0.77])

    df = pd.DataFrame([{
        "Brightness": 200,
        "Texture_Class": "grainy",
        "Texture": 100,
        "Mean_B": 50
    }])
    out = predictor.generate_predictions(df.copy(), DummyModel())
    assert out.loc[0, "Predicted_Melt_Rate"] == 0.77
    assert "Regression override applied" in out.loc[0, "Explanation"]


# =========================================================
# plot_relationships
# =========================================================

def test_plot_relationships_creates_plots(tmp_path, monkeypatch):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))

    df = pd.DataFrame([{
        "Brightness": 150,
        "Texture": 100,
        "Predicted_Melt_Rate": "fast",
        "Predicted_Change_Flag": 1
    }])

    plots = predictor.plot_relationships(df)
    assert len(plots) == 2
    for p in plots:
        assert Path(p).is_file()


def test_plot_relationships_missing_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))

    df = pd.DataFrame([{"Brightness": 150}])
    plots = predictor.plot_relationships(df)
    assert plots == []


# =========================================================
# generate_report
# =========================================================

def test_generate_report_creates_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))

    df = pd.DataFrame([{
        "Observation_ID": "OBS-1",
        "Predicted_Change_Flag": 1,
        "Predicted_Melt_Rate": "fast",
        "Explanation": "Because reasons"
    }])

    plots = []
    path = predictor.generate_report(df, plots, "Regression disabled")

    assert Path(path).is_file()
    content = Path(path).read_text()
    assert "Prediction Report" in content
    assert "OBS-1" in content
    assert "fast" in content
    assert "Regression disabled" in content


# =========================================================
# generate_predicted_image
# =========================================================

def test_generate_predicted_image(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))

    src = tmp_path / "img_correlated.jpg"
    src.write_bytes(b"jpegdata")

    dst = predictor.generate_predicted_image(str(src), "img")
    assert Path(dst).is_file()
    assert Path(dst).read_bytes() == b"jpegdata"

    assert "Created" in capsys.readouterr().out


# =========================================================
# main() — missing CSV
# =========================================================

def test_main_missing_csv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))
    monkeypatch.setattr(predictor, "CSV_PATH", str(tmp_path / "field_data.csv"))
    monkeypatch.setattr(sys, "argv", ["prog"])

    predictor.main()
    assert "field_data.csv not found" in capsys.readouterr().out


# =========================================================
# main() — no correlated images
# =========================================================

def test_main_no_correlated_images(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))
    csv_path = tmp_path / "field_data.csv"
    monkeypatch.setattr(predictor, "CSV_PATH", str(csv_path))

    df = pd.DataFrame([{
        "Photo_Filename": "img_ingested.jpg",
        "Brightness": 150,
        "Texture_Class": "grainy",
        "Texture": 100,
        "Mean_B": 50,
        "Observation_ID": "OBS-1"
    }])
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr(sys, "argv", ["prog"])
    predictor.main()

    out = capsys.readouterr().out
    assert "Updated field_data.csv with predictions" in out

    updated = pd.read_csv(csv_path)
    assert "Predicted_Melt_Rate" in updated.columns
    assert "Explanation" in updated.columns


# =========================================================
# main() — correlated image but no CSV match
# =========================================================

def test_main_correlated_no_csv_match(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))
    csv_path = tmp_path / "field_data.csv"
    monkeypatch.setattr(predictor, "CSV_PATH", str(csv_path))

    df = pd.DataFrame([{
        "Photo_Filename": "other_ingested.jpg",
        "Brightness": 200,
        "Texture_Class": "grainy",
        "Texture": 100,
        "Mean_B": 50,
        "Observation_ID": "OBS-2"
    }])
    df.to_csv(csv_path, index=False)

    (tmp_path / "img_correlated.jpg").write_bytes(b"jpegdata")

    monkeypatch.setattr(sys, "argv", ["prog"])
    predictor.main()

    out = capsys.readouterr().out
    # Predictor always creates predicted images; no mismatch warning expected
    assert "img_predicted.jpg" in out or "Created" in out


# =========================================================
# main() — happy path (no regression)
# =========================================================

def test_main_happy_path_no_regression(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))
    csv_path = tmp_path / "field_data.csv"
    monkeypatch.setattr(predictor, "CSV_PATH", str(csv_path))

    df = pd.DataFrame([{
        "Photo_Filename": "img_ingested.jpg",
        "Brightness": 200,
        "Texture_Class": "grainy",
        "Texture": 100,
        "Mean_B": 50,
        "Observation_ID": "OBS-123"
    }])
    df.to_csv(csv_path, index=False)

    correlated = tmp_path / "img_correlated.jpg"
    correlated.write_bytes(b"jpegdata")

    monkeypatch.setattr(sys, "argv", ["prog"])
    predictor.main()

    out = capsys.readouterr().out
    assert "Predictor-Reporter complete" in out

    assert (tmp_path / "img_predicted.jpg").is_file()
    assert (tmp_path / "brightness_vs_melt_rate_plot.png").is_file()
    assert (tmp_path / "texture_vs_change_flag_plot.png").is_file()
    assert (tmp_path / "prediction_report.md").is_file()

    updated = pd.read_csv(csv_path)
    assert updated.loc[0, "Predicted_Melt_Rate"] == "fast"


# =========================================================
# main() — happy path with regression
# =========================================================

def test_main_happy_path_with_regression(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))
    csv_path = tmp_path / "field_data.csv"
    monkeypatch.setattr(predictor, "CSV_PATH", str(csv_path))

    df = pd.DataFrame([{
        "Photo_Filename": "img_ingested.jpg",
        "Brightness": 80,
        "Texture_Class": "smooth",
        "Texture": 100,
        "Mean_B": 50,
        "Observation_ID": "OBS-999",
        "labeled_melt_rate": 0.3
    }])
    df.to_csv(csv_path, index=False)

    (tmp_path / "img_correlated.jpg").write_bytes(b"jpegdata")

    monkeypatch.setattr(sys, "argv", ["prog", "--enable-regression"])
    predictor.main()

    out = capsys.readouterr().out
    assert "Predictor-Reporter complete" in out

    updated = pd.read_csv(csv_path)
    assert updated.loc[0, "Predicted_Melt_Rate"] in ["slow", 0.3]



