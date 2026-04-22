#
# Evaluate modeled dv/v against observations
# HM Huang, 2026
#
import numpy as np
import pandas as pd
import os

from dvv_model_utils import (
    compute_basic_metrics,
    compute_information_criteria,
    fit_annual_harmonic,
    harmonic_phase_to_day,
    rolling_origin_cross_validation,
)


# ========== #
# User Input #
# ========== #
OUTPUT_DIR = "../output"
DVV_TRANSFER_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "dvv_transfer_output.csv")
DVV_TRANSFER_COEFFICIENTS_CSV = os.path.join(OUTPUT_DIR, "dvv_transfer_coefficients.csv")

OUTPUT_METRICS_CSV = os.path.join(OUTPUT_DIR, "dvv_model_metrics.csv")
OUTPUT_CV_CSV = os.path.join(OUTPUT_DIR, "dvv_model_cv.csv")

FIT_INTERCEPT = True
N_CV_SPLITS = 5


def read_transfer_output(filepath):
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()


def read_coefficients(filepath):
    return pd.read_csv(filepath)


def infer_feature_columns(df):
    candidate_columns = ["pp_gwl_pa", "pp_atm_pa", "dvv_temp_thermoelastic"]
    return [col for col in candidate_columns if col in df.columns]


def compute_amplitude_phase_metrics(index, observed, modeled):
    obs_fit = fit_annual_harmonic(index, observed)
    mod_fit = fit_annual_harmonic(index, modeled)
    return {
        "obs_annual_amplitude": obs_fit["amplitude"],
        "model_annual_amplitude": mod_fit["amplitude"],
        "obs_phase_day": harmonic_phase_to_day(obs_fit["phase_rad"]),
        "model_phase_day": harmonic_phase_to_day(mod_fit["phase_rad"]),
        "phase_day_difference": harmonic_phase_to_day(
            mod_fit["phase_rad"] - obs_fit["phase_rad"]
        ),
    }


def run_evaluation_workflow(
    dvv_transfer_output_csv,
    dvv_transfer_coefficients_csv,
    output_metrics_csv,
    output_cv_csv,
    fit_intercept,
    n_cv_splits,
):
    os.makedirs(os.path.dirname(output_metrics_csv), exist_ok=True)
    df = read_transfer_output(dvv_transfer_output_csv)
    coeff_df = read_coefficients(dvv_transfer_coefficients_csv)
    feature_columns = infer_feature_columns(df)
    metric_rows = []
    split_definitions = [("all", np.ones(len(df), dtype=bool))]
    if "data_split" in df.columns:
        split_definitions.extend(
            [
                ("calibration", df["data_split"].eq("calibration").to_numpy()),
                ("validation", df["data_split"].eq("validation").to_numpy()),
            ]
        )

    for split_name, split_mask in split_definitions:
        split_df = df.loc[split_mask]
        if split_df.empty:
            continue
        basic = compute_basic_metrics(
            split_df["dvv_obs"].to_numpy(),
            split_df["dvv_model"].to_numpy(),
        )
        info = compute_information_criteria(
            y_true=split_df["dvv_obs"].to_numpy(),
            y_pred=split_df["dvv_model"].to_numpy(),
            n_parameters=len(coeff_df),
        )
        amp_phase = compute_amplitude_phase_metrics(
            index=split_df.index,
            observed=split_df["dvv_obs"].to_numpy(),
            modeled=split_df["dvv_model"].to_numpy(),
        )
        metric_rows.append(
            {
                "split": split_name,
                **basic,
                **info,
                **amp_phase,
                "n_obs": len(split_df),
                "n_parameters": len(coeff_df),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_metrics_csv, index=False)

    if "method" in coeff_df.columns and coeff_df["method"].eq(
        "extended_rivet_like_band_transfer"
    ).any():
        cv_df = pd.DataFrame(
            [
                {
                    "note": "Rolling CV is not implemented for the extended Rivet-like band-transfer model.",
                }
            ]
        )
    else:
        X = df[feature_columns].to_numpy(dtype=float)
        y = df["dvv_obs"].to_numpy(dtype=float)
        cv_df = rolling_origin_cross_validation(
            X=X,
            y=y,
            feature_names=feature_columns,
            fit_intercept=fit_intercept,
            n_splits=n_cv_splits,
        )
    cv_df.to_csv(output_cv_csv, index=False)
    return metrics_df, cv_df


if __name__ == "__main__":
    run_evaluation_workflow(
        dvv_transfer_output_csv=DVV_TRANSFER_OUTPUT_CSV,
        dvv_transfer_coefficients_csv=DVV_TRANSFER_COEFFICIENTS_CSV,
        output_metrics_csv=OUTPUT_METRICS_CSV,
        output_cv_csv=OUTPUT_CV_CSV,
        fit_intercept=FIT_INTERCEPT,
        n_cv_splits=N_CV_SPLITS,
    )
    print("\nFinished!")
