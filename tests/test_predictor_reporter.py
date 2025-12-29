import os
import csv
from pathlib import Path

import pandas as pd
import pytest

import universal_field_toolkit_predictor_reporter_sred as predictor


# ---------------------------------------------------------
# apply_rules
# ---------------------------------------------------------

def test_apply_rules_fast_slushy():
    row = {
        "brightness": 181,
        "Texture_Class": "slushy",
    }
    rate, explanation = predictor.apply_rules(row)
    assert rate == "fast"
    assert "Fast Melt" in explanation
    assert "Brightness > 180" in explanation
    assert "Slushy" in explanation


def test_apply_rules_slow_icy():
    row = {
        "brightness": 99,
        "Texture_Class": "icy",
    }
    rate, explanation = predictor.apply_rules(row)
    assert rate == "slow"
    assert "Slow Melt" in explanation
    assert "Brightness < 100" in explanation
    assert "Icy" in explanation


def test_apply_rules_moderate_other_cases():
    cases = [
        {"brightness": 180, "Texture_Class": "slushy"},   # boundary, not > 180
        {"brightness": 181, "Texture_Class": "icy"},      # texture mismatch
        {"brightness": 99, "Texture_Class": "slushy"},    # brightness < 100 but texture wrong
        {"brightness": 150, "Texture_Class": "other"},
    ]
    for row in cases:
        rate, explanation = predictor.apply_rules(row)
        assert rate == "moderate"
        assert "Moderate Melt" in explanation
        assert "mixed" in explanation


# ---------------------------------------------------------
# fit_regression (placeholder)
# ---------------------------------------------------------

def test_fit_regression_returns_none():
    df = pd.DataFrame({"brightness": [1, 2, 3]})
    result = predictor.fit_regression(df)
    assert result is None


# ---------------------------------------------------------
# generate_plot
# ---------------------------------------------------------

def test_generate_plot_creates_file(tmp_path, monkeypatch):
    # Patch WORKING_DIR
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))

    df_row = {
        "brightness": 150,
        "shadow_intensity": 0.3,
    }
    base_name = "sample"

    plot_path = predictor.generate_plot(df_row, base_name)

    assert plot_path == os.path.join(str(tmp_path), "sample_plot.png")
    assert os.path.isfile(plot_path)


# ---------------------------------------------------------
# generate_report
# ---------------------------------------------------------

def test_generate_report_creates_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))

    df_row = {
        "Observation_ID": "OBS-1",
        "Predicted_Melt_Rate": "fast",
        "Explanation": "Some explanation",
    }
    base_name = "sample"

    report_path = predictor.generate_report(df_row, base_name)

    assert report_path == os.path.join(str(tmp_path), "sample_report.md")
    assert os.path.isfile(report_path)

    content = Path(report_path).read_text()
    assert "# Prediction Report for sample" in content
    assert "- Observation ID: OBS-1" in content
    assert "- Predicted Melt Rate: fast" in content
    assert "- Explanation: Some explanation" in content


# ---------------------------------------------------------
# generate_predicted_image
# ---------------------------------------------------------

def test_generate_predicted_image_copies_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))

    src = tmp_path / "img_correlated.jpg"
    dst = tmp_path / "img_predicted.jpg"
    src.write_bytes(b"fakejpegdata")

    dst_path = predictor.generate_predicted_image(str(src), "img")

    assert dst_path == str(dst)
    assert dst.is_file()
    assert dst.read_bytes() == b"fakejpegdata"

    captured = capsys.readouterr()
    assert "Created" in captured.out


# ---------------------------------------------------------
# main() – CSV missing
# ---------------------------------------------------------

def test_main_missing_csv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))
    monkeypatch.setattr(predictor, "CSV_PATH", os.path.join(str(tmp_path), "field_data.csv"))

    # No CSV created
    monkeypatch.setattr("sys.argv", ["prog"])

    predictor.main()

    captured = capsys.readouterr()
    assert "field_data.csv not found" in captured.out
    # No CSV created by predictor
    assert not os.path.isfile(os.path.join(str(tmp_path), "field_data.csv"))


# ---------------------------------------------------------
# main() – no correlated images present
# ---------------------------------------------------------

def test_main_no_correlated_images(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))
    csv_path = os.path.join(str(tmp_path), "field_data.csv")
    monkeypatch.setattr(predictor, "CSV_PATH", csv_path)

    # Create a minimal CSV that won't be used (no correlated files)
    df = pd.DataFrame(
        [
            {
                "Photo_Filename": "img_ingested.jpg",
                "brightness": 150,
                "Texture_Class": "slushy",
                "shadow_intensity": 0.3,
                "Observation_ID": "OBS-1",
            }
        ]
    )
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr("sys.argv", ["prog"])

    predictor.main()

    captured = capsys.readouterr()
    # Should still update CSV (add prediction columns) and finish
    assert "Updated field_data.csv with predictions" in captured.out
    assert "Predictor‑Reporter complete" in captured.out

    updated = pd.read_csv(csv_path)
    assert "Predicted_Melt_Rate" in updated.columns
    assert "Explanation" in updated.columns
    # No correlated images → no per-row predictions (empty strings)
    assert updated.loc[0, "Predicted_Melt_Rate"] == ""
    assert updated.loc[0, "Explanation"] == ""


# ---------------------------------------------------------
# main() – correlated image, but no CSV match
# ---------------------------------------------------------

def test_main_correlated_no_csv_match(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))
    csv_path = os.path.join(str(tmp_path), "field_data.csv")
    monkeypatch.setattr(predictor, "CSV_PATH", csv_path)

    # CSV with different filename than correlated
    df = pd.DataFrame(
        [
            {
                "Photo_Filename": "other_ingested.jpg",
                "brightness": 200,
                "Texture_Class": "slushy",
                "shadow_intensity": 0.5,
                "Observation_ID": "OBS-2",
            }
        ]
    )
    df.to_csv(csv_path, index=False)

    # Correlated image that doesn't match CSV filename
    (tmp_path / "img_correlated.jpg").write_bytes(b"fake")

    monkeypatch.setattr("sys.argv", ["prog"])

    predictor.main()

    captured = capsys.readouterr()
    assert "No CSV match for img_correlated.jpg" in captured.out

    updated = pd.read_csv(csv_path)
    # Prediction columns exist but still empty
    assert "Predicted_Melt_Rate" in updated.columns
    assert "Explanation" in updated.columns
    assert updated.loc[0, "Predicted_Melt_Rate"] == ""
    assert updated.loc[0, "Explanation"] == ""


# ---------------------------------------------------------
# main() – full happy path without regression
# ---------------------------------------------------------

def test_main_happy_path_no_regression(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))
    csv_path = os.path.join(str(tmp_path), "field_data.csv")
    monkeypatch.setattr(predictor, "CSV_PATH", csv_path)

    # Create CSV with one row that will match img_correlated.jpg
    df = pd.DataFrame(
        [
            {
                "Photo_Filename": "img_ingested.jpg",
                "brightness": 190,
                "Texture_Class": "slushy",
                "shadow_intensity": 0.4,
                "Observation_ID": "OBS-123",
            }
        ]
    )
    df.to_csv(csv_path, index=False)

    # Create correlated image
    correlated = tmp_path / "img_correlated.jpg"
    correlated.write_bytes(b"jpegdata")

    # No regression flag
    monkeypatch.setattr("sys.argv", ["prog"])

    predictor.main()

    captured = capsys.readouterr()
    assert "Updated field_data.csv with predictions" in captured.out
    assert "Predictor‑Reporter complete" in captured.out

    # Check predicted image
    predicted_img = tmp_path / "img_predicted.jpg"
    assert predicted_img.is_file()
    assert predicted_img.read_bytes() == b"jpegdata"

    # Check plot
    plot_path = tmp_path / "img_plot.png"
    assert plot_path.is_file()

    # Check report
    report_path = tmp_path / "img_report.md"
    assert report_path.is_file()

    # Check CSV updated with fast prediction
    updated = pd.read_csv(csv_path)
    assert updated.loc[0, "Predicted_Melt_Rate"] == "fast"
    assert "Fast Melt" in updated.loc[0, "Explanation"]


# ---------------------------------------------------------
# main() – happy path with regression flag
# ---------------------------------------------------------

def test_main_happy_path_with_regression(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "WORKING_DIR", str(tmp_path))
    csv_path = os.path.join(str(tmp_path), "field_data.csv")
    monkeypatch.setattr(predictor, "CSV_PATH", csv_path)

    # CSV
    df = pd.DataFrame(
        [
            {
                "Photo_Filename": "img_ingested.jpg",
                "brightness": 80,
                "Texture_Class": "icy",
                "shadow_intensity": 0.7,
                "Observation_ID": "OBS-999",
            }
        ]
    )
    df.to_csv(csv_path, index=False)

    # Correlated file
    correlated = tmp_path / "img_correlated.jpg"
    correlated.write_bytes(b"jpegdata")

    # Track calls to fit_regression by wrapping it
    called = {"value": False}

    original_fit = predictor.fit_regression

    def wrapped_fit(df_arg):
        called["value"] = True
        return original_fit(df_arg)

    monkeypatch.setattr(predictor, "fit_regression", wrapped_fit)

    # Enable regression via CLI flag
    monkeypatch.setattr("sys.argv", ["prog", "--enable-regression"])

    predictor.main()

    captured = capsys.readouterr()
    assert "Updated field_data.csv with predictions" in captured.out
    assert "Predictor‑Reporter complete" in captured.out

    # fit_regression must have been called
    assert called["value"] is True

    # Check CSV: prediction should be slow here
    updated = pd.read_csv(csv_path)
    assert updated.loc[0, "Predicted_Melt_Rate"] == "slow"
    assert "Slow Melt" in updated.loc[0, "Explanation"]



