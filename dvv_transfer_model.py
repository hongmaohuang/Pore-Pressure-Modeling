#
# Transfer theoretical pore-pressure / thermoelastic responses to dv/v(t)
# HM Huang, 2026
#
import numpy as np
import pandas as pd
import glob
import os

from dvv_model_utils import (
    compute_information_criteria,
    fit_linear_model,
)


# ========== #
# User Input #
# ========== #
OUTPUT_DIR = "../output"
PORE_PRESSURE_CSV_PATH = os.path.join(OUTPUT_DIR, "pore_pressure_output.csv")
OBSERVED_DVV_INPUT_MODE = "DTT_FOLDER"  # "CSV" or "DTT_FOLDER"
OBSERVED_DVV_CSV_PATH = "observed_dvv.csv"
OBSERVED_DVV_DTT_FOLDER = "/Users/hmhuang/Sth_should_be_local/Iceland/dvv_results/1_filter_202603/01/005_DAYS/ZZ"
DVV_PAIR_MODE = "ALL"  # "ALL" or "PAIR"
DTT_STATION1 = None
DTT_STATION2 = None
DVV_VALUE_COLUMN = "M0"
DVV_SCALE = 1.0

MODEL_DEPTH_M = 1900.0
FIT_INTERCEPT = True
INCLUDE_THERMOELASTIC = True

OUTPUT_TIMESERIES_CSV = os.path.join(OUTPUT_DIR, "dvv_transfer_output.csv")
OUTPUT_COEFFICIENTS_CSV = os.path.join(OUTPUT_DIR, "dvv_transfer_coefficients.csv")

def read_pore_pressure_output(filepath):
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()


def read_observed_dvv_csv(filepath):
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df["dvv_obs"] = pd.to_numeric(df["dvv"], errors="coerce")
    df = df.dropna(subset=["datetime", "dvv_obs"]).set_index("datetime").sort_index()
    return df[["dvv_obs"]]


def split_dtt_pair_name(pair_name):
    parts = str(pair_name).split("_")
    if len(parts) < 4:
        return None, None
    midpoint = len(parts) // 2
    return "_".join(parts[:midpoint]), "_".join(parts[midpoint:])


def read_observed_dvv_from_dtt_folder(
    folder,
    pair_mode="ALL",
    station1=None,
    station2=None,
    value_column="M0",
    scale=1.0,
):
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not files:
        raise ValueError(f"No DTT files found in {folder}")

    rows = []
    target_pairs = None
    if pair_mode == "PAIR":
        if station1 is None or station2 is None:
            raise ValueError("station1 and station2 must be provided when pair_mode='PAIR'")
        target_pairs = {f"{station1}_{station2}", f"{station2}_{station1}"}

    for filepath in files:
        df = pd.read_csv(filepath)
        if pair_mode == "PAIR":
            df = df[df["Pairs"].isin(target_pairs)]
        else:
            keep_mask = []
            for pair_name in df["Pairs"]:
                sta1, sta2 = split_dtt_pair_name(pair_name)
                keep_mask.append(sta1 is not None and sta2 is not None and sta1 != sta2)
            df = df[np.asarray(keep_mask, dtype=bool)]
        if df.empty:
            continue

        time_value = pd.to_datetime(df["Date"].iloc[0], errors="coerce")
        dvv_values = pd.to_numeric(df[value_column], errors="coerce").dropna()
        if pd.isna(time_value) or dvv_values.empty:
            continue
        rows.append((time_value, float(np.median(dvv_values.to_numpy()) * scale)))

    if not rows:
        raise ValueError("No usable dv/v values were parsed from the DTT folder")

    out = pd.DataFrame(rows, columns=["datetime", "dvv_obs"])
    return out.set_index("datetime").sort_index()


def read_observed_dvv_series():
    if OBSERVED_DVV_INPUT_MODE == "CSV":
        return read_observed_dvv_csv(OBSERVED_DVV_CSV_PATH)
    if OBSERVED_DVV_INPUT_MODE == "DTT_FOLDER":
        return read_observed_dvv_from_dtt_folder(
            folder=OBSERVED_DVV_DTT_FOLDER,
            pair_mode=DVV_PAIR_MODE,
            station1=DTT_STATION1,
            station2=DTT_STATION2,
            value_column=DVV_VALUE_COLUMN,
            scale=DVV_SCALE,
        )
    raise ValueError(f"Unsupported OBSERVED_DVV_INPUT_MODE: {OBSERVED_DVV_INPUT_MODE}")


def build_predictor_dataframe(pore_df, model_depth_m, include_thermoelastic):
    depth_label = int(float(model_depth_m))
    required_columns = [
        f"Pp_gwl_z{depth_label}m_pa",
        f"Pp_atm_z{depth_label}m_pa",
    ]
    missing = [col for col in required_columns if col not in pore_df.columns]
    if missing:
        raise ValueError(f"Missing required pore-pressure columns: {missing}")

    predictors = pd.DataFrame(index=pore_df.index)
    predictors["pp_gwl_pa"] = pore_df[f"Pp_gwl_z{depth_label}m_pa"]
    predictors["pp_atm_pa"] = pore_df[f"Pp_atm_z{depth_label}m_pa"]

    if include_thermoelastic:
        if "dvv_temp_thermoelastic" not in pore_df.columns:
            raise ValueError("dvv_temp_thermoelastic is required in pore-pressure output")
        predictors["dvv_temp_thermoelastic"] = pore_df["dvv_temp_thermoelastic"]
    return predictors


def run_dvv_transfer_workflow(
    pore_pressure_csv_path,
    model_depth_m,
    include_thermoelastic,
    fit_intercept,
    output_timeseries_csv,
    output_coefficients_csv,
):
    os.makedirs(os.path.dirname(output_timeseries_csv), exist_ok=True)
    pore_df = read_pore_pressure_output(pore_pressure_csv_path)
    dvv_df = read_observed_dvv_series()
    predictors = build_predictor_dataframe(
        pore_df,
        model_depth_m=model_depth_m,
        include_thermoelastic=include_thermoelastic,
    )

    merged = predictors.join(dvv_df, how="inner")
    if merged.empty:
        raise ValueError("No overlapping timestamps between predictors and observed dv/v")

    feature_names = list(predictors.columns)
    X = merged[feature_names].to_numpy(dtype=float)
    y = merged["dvv_obs"].to_numpy(dtype=float)
    fit = fit_linear_model(X, y, feature_names=feature_names, fit_intercept=fit_intercept)
    merged["dvv_model"] = fit["predicted"]
    merged["dvv_residual"] = merged["dvv_obs"] - merged["dvv_model"]

    info = compute_information_criteria(
        y_true=merged["dvv_obs"].to_numpy(),
        y_pred=merged["dvv_model"].to_numpy(),
        n_parameters=fit["n_parameters"],
    )
    merged["model_depth_m"] = float(model_depth_m)
    merged.to_csv(output_timeseries_csv, index=True)

    coeff_df = pd.DataFrame(
        {
            "term": fit["feature_names"],
            "coefficient": fit["coefficients"],
        }
    )
    coeff_df["aic"] = info["aic"]
    coeff_df["bic"] = info["bic"]
    coeff_df["rss"] = info["rss"]
    coeff_df.to_csv(output_coefficients_csv, index=False)
    return merged, coeff_df


if __name__ == "__main__":
    run_dvv_transfer_workflow(
        pore_pressure_csv_path=PORE_PRESSURE_CSV_PATH,
        model_depth_m=MODEL_DEPTH_M,
        include_thermoelastic=INCLUDE_THERMOELASTIC,
        fit_intercept=FIT_INTERCEPT,
        output_timeseries_csv=OUTPUT_TIMESERIES_CSV,
        output_coefficients_csv=OUTPUT_COEFFICIENTS_CSV,
    )
    print("\nFinished!")
