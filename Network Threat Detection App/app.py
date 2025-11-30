from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path
from uuid import uuid4

app = Flask(__name__)
app.secret_key = "change-me"

# -----------------------------
# Paths
# -----------------------------
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load models & metadata
# -----------------------------
scaler   = joblib.load(MODELS_DIR / "scaler.pkl")
selector = joblib.load(MODELS_DIR / "selector.pkl")

models = {
    "RandomForest Classifier": joblib.load(MODELS_DIR / "rf.pkl"),
    "AdaBoost Classifier":     joblib.load(MODELS_DIR / "ada.pkl"),
    "GaussianNB Classifier":   joblib.load(MODELS_DIR / "gnb.pkl"),
    "Voting Classifier":       joblib.load(MODELS_DIR / "voting.pkl"),
}

FEATURE_78   = json.loads((MODELS_DIR / "feature_78.json").read_text())
FEATURE_15   = json.loads((MODELS_DIR / "feature_15.json").read_text())
RAW_REQUIRED = json.loads((MODELS_DIR / "raw_required.json").read_text())

# Optional label mapping
label_map_path = MODELS_DIR / "label_mapping.json"
if label_map_path.exists():
    LABEL_MAP = {int(k): v for k, v in json.loads(label_map_path.read_text()).items()}
else:
    LABEL_MAP = {0: "Benign", 1: "Malicious"}

# -----------------------------
# Helpers
# -----------------------------
def _ensure_protocol_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate training's one-hot for Protocol with drop_first=True."""
    if "Protocol" not in df.columns:
        return df
    df = df.copy()
    df["Protocol"] = df["Protocol"].astype(str)
    d = pd.get_dummies(df["Protocol"], prefix="Protocol", drop_first=True)
    for col in ["Protocol_17", "Protocol_6"]:
        if col not in d.columns:
            d[col] = False
    d = d[["Protocol_17", "Protocol_6"]]
    return pd.concat([df.drop(columns=["Protocol"]), d], axis=1)


def preprocess_raw_to_78(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Your training preprocessing → 78 columns (order = FEATURE_78)."""
    df = df_raw.copy()

    # drop extra cols if present
    for col in ["Timestamp", "Dst Port"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = _ensure_protocol_cols(df)

    # drop label if present
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # cast objects
    for c in df.columns:
        if df[c].dtype == "O":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # fill rate columns like notebook
    for rate_col in ["Flow Byts/s", "Flow Pkts/s"]:
        if rate_col in df.columns:
            m = df[rate_col].max(skipna=True)
            df[rate_col] = df[rate_col].fillna(m)

    df = df.fillna(0).clip(lower=0)

    # align to FEATURE_78
    for col in FEATURE_78:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_78]
    return df


def scale_then_select(X78: pd.DataFrame) -> np.ndarray:
    X_scaled = scaler.transform(X78)      # expects 78 cols
    X15 = selector.transform(X_scaled)    # -> 15 cols
    return X15


def predict_block(X15: np.ndarray, chosen_models: list[str]):
    """Returns dict: model_name -> dict(pred=array, proba=array|None, counts=dict)."""
    out = {}
    for name in chosen_models:
        mdl = models[name]
        y = mdl.predict(X15)
        p1 = None
        if hasattr(mdl, "predict_proba"):
            try:
                p1 = mdl.predict_proba(X15)[:, 1]
            except Exception:
                p1 = None
        benign = int((y == 0).sum())
        mal    = int((y == 1).sum())
        out[name] = {"pred": y, "proba": p1, "counts": {"Benign": benign, "Malicious": mal}}
    return out


def is_not_traffic_row_heuristic(partial_dict: dict) -> bool:
    """Only used for manual; if obviously empty line, reject."""
    keys = [
        "Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts",
        "Pkt Len Max","Fwd Pkt Len Max","Bwd Pkt Len Max"
    ]
    s = sum(float(partial_dict.get(k, 0.0)) for k in keys if k in partial_dict)
    return s == 0.0


def partial15_to_X78(partial_15: dict) -> pd.DataFrame:
    """Build a single-row 78-feature frame from only the top-15 raw features."""
    row = {c: 0.0 for c in FEATURE_78}
    for k, v in partial_15.items():
        if k in row:
            row[k] = float(v)
    return pd.DataFrame([row], columns=FEATURE_78)


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return redirect(url_for("upload"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html", model_names=list(models.keys()))

    # POST
    try:
        file = request.files.get("csv")
        chosen = request.form.getlist("models")

        if not file or file.filename == "":
            flash("Please choose a CSV file.", "warning")
            return redirect(url_for("upload"))
        if not chosen:
            flash("Select at least one model.", "warning")
            return redirect(url_for("upload"))

        df_raw = pd.read_csv(file, low_memory=False)

        # sanity check
        overlap = set(RAW_REQUIRED).intersection(df_raw.columns)
        if len(overlap) < 20:
            flash("Unknown file (does not look like CIC-IDS-2018).", "danger")
            return redirect(url_for("upload"))

        X78 = preprocess_raw_to_78(df_raw)
        if X78.shape[1] != len(FEATURE_78) or len(X78) == 0:
            flash("Could not derive required 78 features.", "danger")
            return redirect(url_for("upload"))

        X15 = scale_then_select(X78)
        results = predict_block(X15, chosen)

        flash(f"Processed {len(X15)} rows.", "success")

        # per-model summary
        for name, res in results.items():
            counts = res["counts"]
            flash(f"{name}: Benign={counts['Benign']}, Malicious={counts['Malicious']}", "info")

        # build per-row (for display, limit to 20)
        per_row = []
        max_rows = min(len(X15), 20)
        for i in range(max_rows):
            row_entry = {"idx": i + 1, "preds": {}}
            for mname, res in results.items():
                lbl = LABEL_MAP[int(res["pred"][i])]
                prob = None
                if res["proba"] is not None:
                    prob = float(res["proba"][i])
                row_entry["preds"][mname] = {"label": lbl, "proba": prob}
            per_row.append(row_entry)

        # build full CSV to download
        full_df = pd.DataFrame({"Row": range(1, len(X15) + 1)})
        for mname, res in results.items():
            labels = [LABEL_MAP[int(v)] for v in res["pred"]]
            full_df[f"{mname} Label"] = labels
            if res["proba"] is not None:
                full_df[f"{mname} Prob"] = [float(x) for x in res["proba"]]

        file_id = uuid4().hex
        out_path = OUTPUTS_DIR / f"predictions_{file_id}.csv"
        full_df.to_csv(out_path, index=False)

        return render_template(
            "upload.html",
            model_names=list(models.keys()),
            per_row=per_row,
            chosen_models=chosen,
            download_id=file_id
        )

    except Exception as e:
        flash(f"Error: {e}", "danger")
        return redirect(url_for("upload"))


@app.route("/manual", methods=["GET", "POST"])
def manual():
    if request.method == "GET":
        return render_template(
            "manual.html",
            top15_fields=FEATURE_15,
            model_names=list(models.keys()),
            form_values={},
            chosen_models=[]
        )

    # POST
    try:
        chosen = request.form.getlist("models")
        # collect 15 fields from the form
        form_values = {}
        for col in FEATURE_15:
            raw = request.form.get(col, "").strip()
            form_values[col] = raw  # keep raw text so we can re-fill

        if not chosen:
            flash("Select at least one model.", "warning")
            return render_template(
                "manual.html",
                top15_fields=FEATURE_15,
                model_names=list(models.keys()),
                form_values=form_values,
                chosen_models=chosen
            )

        # convert to floats for prediction
        partial_15 = {k: float(v) if v != "" else 0.0 for k, v in form_values.items()}

        if is_not_traffic_row_heuristic(partial_15):
            flash("This does not look like packet/flow data.", "danger")
            return render_template(
                "manual.html",
                top15_fields=FEATURE_15,
                model_names=list(models.keys()),
                form_values=form_values,
                chosen_models=chosen
            )

        X78 = partial15_to_X78(partial_15)
        X15 = scale_then_select(X78)
        results = predict_block(X15, chosen)

        for name, res in results.items():
            y = res["pred"][0]
            lbl = LABEL_MAP[int(y)]
            if res["proba"] is not None:
                flash(f"{name}: {lbl} (P_malicious={float(res['proba'][0]):.4f})", "success")
            else:
                flash(f"{name}: {lbl}", "success")

        # re-render with values still in the inputs
        return render_template(
            "manual.html",
            top15_fields=FEATURE_15,
            model_names=list(models.keys()),
            form_values=form_values,
            chosen_models=chosen
        )

    except Exception as e:
        flash(f"Error: {e}", "danger")
        return redirect(url_for("manual"))


@app.route("/download/<file_id>")
def download(file_id):
    path = OUTPUTS_DIR / f"predictions_{file_id}.csv"
    if not path.exists():
        flash("File not found or expired.", "danger")
        return redirect(url_for("upload"))
    return send_file(path, as_attachment=True, download_name=f"predictions_{file_id}.csv")


if __name__ == "__main__":
    app.run(debug=True)
