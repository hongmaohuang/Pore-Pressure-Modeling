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
    excluded = {"dvv_obs", "dvv_model", "dvv_residual", "model_depth_m"}
    return [col for col in df.columns if col not in excluded]


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

    basic = compute_basic_metrics(df["dvv_obs"].to_numpy(), df["dvv_model"].to_numpy())
    info = compute_information_criteria(
        y_true=df["dvv_obs"].to_numpy(),
        y_pred=df["dvv_model"].to_numpy(),
        n_parameters=len(coeff_df),
    )
    amp_phase = compute_amplitude_phase_metrics(
        index=df.index,
        observed=df["dvv_obs"].to_numpy(),
        modeled=df["dvv_model"].to_numpy(),
    )

    metrics_df = pd.DataFrame(
        [
            {
                **basic,
                **info,
                **amp_phase,
                "n_obs": len(df),
                "n_parameters": len(coeff_df),
            }
        ]
    )
    metrics_df.to_csv(output_metrics_csv, index=False)

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
