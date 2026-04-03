#
# Transfer theoretical pore-pressure / thermoelastic responses to dv/v(t)
# HM Huang, 2026
#
import numpy as np
import pandas as pd
import glob
import os
from scipy.signal import butter, filtfilt

from dvv_model_utils import (
    compute_information_criteria,
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
DVV_VALUE_COLUMN = "M0"  # DTT dt/t column
DVV_SCALE = -100.0  # convert dt/t to dv/v (%)

MODEL_DEPTH_M = 0.4
FIT_INTERCEPT = True
INCLUDE_THERMOELASTIC = True
CALIBRATION_START_TIME = "2021-03-01"
CALIBRATION_END_TIME = "2021-06-10"
VALIDATION_START_TIME = "2021-06-11"
VALIDATION_END_TIME = "2021-12-31"
RIVET_PERIOD_BANDS_DAYS = (
    (300.0, 120.0),
    (120.0, 60.0),
    (60.0, 30.0),
    (30.0, 16.0),
    (16.0, 8.0),
)
RIVET_FILTER_ORDER = 2

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


def get_station_code(station_name):
    parts = str(station_name).split("_")
    if not parts:
        return None
    return parts[-1]


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
        target_pairs = {station1, station2}

    for filepath in files:
        df = pd.read_csv(filepath)
        if pair_mode == "PAIR":
            keep_mask = []
            for pair_name in df["Pairs"]:
                sta1, sta2 = split_dtt_pair_name(pair_name)
                if sta1 is None or sta2 is None:
                    keep_mask.append(False)
                    continue
                station_codes = {get_station_code(sta1), get_station_code(sta2)}
                keep_mask.append(station_codes == target_pairs)
            df = df[np.asarray(keep_mask, dtype=bool)]
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


def build_time_mask(index, start_time=None, end_time=None):
    mask = np.ones(len(index), dtype=bool)
    if start_time is not None:
        mask &= pd.to_datetime(index) >= pd.Timestamp(start_time)
    if end_time is not None:
        mask &= pd.to_datetime(index) <= pd.Timestamp(end_time)
    return mask


def bandpass_daily_series(values, period_long_days, period_short_days, order):
    values = np.asarray(values, dtype=float)
    fs = 1.0
    low_frequency = 1.0 / float(period_long_days)
    high_frequency = 1.0 / float(period_short_days)
    b, a = butter(order, [low_frequency, high_frequency], btype="bandpass", fs=fs)
    return filtfilt(b, a, values)


def build_regular_daily_dataframe(df, columns):
    daily_index = pd.date_range(
        pd.Timestamp(df.index.min()).normalize(),
        pd.Timestamp(df.index.max()).normalize(),
        freq="1D",
    )
    regular = df[columns].reindex(daily_index)
    regular = regular.interpolate(method="time").ffill().bfill()
    return regular


def fit_extended_rivet_like_transfer(
    df,
    feature_names,
    calibration_mask,
    period_bands_days,
    filter_order,
):
    regular = build_regular_daily_dataframe(df, feature_names + ["dvv_obs"])
    regular["dvv_obs_raw"] = regular["dvv_obs"]
    regular["dvv_obs_bandpassed"] = 0.0
    regular["dvv_model"] = 0.0

    for feature_name in feature_names:
        regular[f"dvv_model_from_{feature_name}"] = 0.0

    coefficient_rows = []
    calibration_mask_regular = build_time_mask(
        regular.index,
        start_time=df.index[calibration_mask].min(),
        end_time=df.index[calibration_mask].max(),
    )

    for period_long_days, period_short_days in period_bands_days:
        y_band = bandpass_daily_series(
            regular["dvv_obs_raw"].to_numpy(),
            period_long_days=period_long_days,
            period_short_days=period_short_days,
            order=filter_order,
        )
        X_band_columns = []
        for feature_name in feature_names:
            X_band_columns.append(
                bandpass_daily_series(
                    regular[feature_name].to_numpy(),
                    period_long_days=period_long_days,
                    period_short_days=period_short_days,
                    order=filter_order,
                )
            )
        X_band = np.column_stack(X_band_columns)
        coeffs, _, _, _ = np.linalg.lstsq(
            X_band[calibration_mask_regular],
            y_band[calibration_mask_regular],
            rcond=None,
        )
        regular["dvv_obs_bandpassed"] += y_band
        regular["dvv_model"] += X_band @ coeffs
        for feature_index, feature_name in enumerate(feature_names):
            contribution = X_band[:, feature_index] * coeffs[feature_index]
            regular[f"dvv_model_from_{feature_name}"] += contribution
            coefficient_rows.append(
                {
                    "period_long_days": float(period_long_days),
                    "period_short_days": float(period_short_days),
                    "term": feature_name,
                    "coefficient": float(coeffs[feature_index]),
                }
            )

    return regular, pd.DataFrame(coefficient_rows)


def run_dvv_transfer_workflow(
    pore_pressure_csv_path,
    model_depth_m,
    include_thermoelastic,
    fit_intercept,
    calibration_start_time,
    calibration_end_time,
    validation_start_time,
    validation_end_time,
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
    merged["dvv_obs_raw"] = merged["dvv_obs"]
    merged["pp_total_pa"] = merged["pp_gwl_pa"] + merged["pp_atm_pa"]
    merged["data_split"] = "unused"

    calibration_mask = build_time_mask(
        merged.index,
        start_time=calibration_start_time,
        end_time=calibration_end_time,
    )
    validation_mask = build_time_mask(
        merged.index,
        start_time=validation_start_time,
        end_time=validation_end_time,
    )
    if not np.any(calibration_mask):
        raise ValueError("No overlap between calibration period and transfer-model data")
    if not np.any(validation_mask):
        raise ValueError("No overlap between validation period and transfer-model data")

    merged.loc[calibration_mask, "data_split"] = "calibration"
    merged.loc[validation_mask, "data_split"] = "validation"

    regular_modeled, coeff_df = fit_extended_rivet_like_transfer(
        merged,
        feature_names=feature_names,
        calibration_mask=calibration_mask,
        period_bands_days=RIVET_PERIOD_BANDS_DAYS,
        filter_order=RIVET_FILTER_ORDER,
    )
    merged["dvv_obs_bandpassed"] = regular_modeled["dvv_obs_bandpassed"].reindex(merged.index)
    merged["dvv_obs"] = merged["dvv_obs_bandpassed"]
    merged["dvv_model"] = regular_modeled["dvv_model"].reindex(merged.index)
    for feature_name in feature_names:
        merged[f"dvv_model_from_{feature_name}"] = regular_modeled[
            f"dvv_model_from_{feature_name}"
        ].reindex(merged.index)
    merged["dvv_residual"] = merged["dvv_obs"] - merged["dvv_model"]

    info = compute_information_criteria(
        y_true=merged.loc[calibration_mask, "dvv_obs"].to_numpy(),
        y_pred=merged.loc[calibration_mask, "dvv_model"].to_numpy(),
        n_parameters=len(coeff_df),
    )
    merged["model_depth_m"] = float(model_depth_m)
    merged.to_csv(output_timeseries_csv, index=True, index_label="datetime")

    coeff_df["aic"] = info["aic"]
    coeff_df["bic"] = info["bic"]
    coeff_df["rss"] = info["rss"]
    coeff_df["method"] = "extended_rivet_like_band_transfer"
    coeff_df["calibration_start_time"] = calibration_start_time
    coeff_df["calibration_end_time"] = calibration_end_time
    coeff_df["validation_start_time"] = validation_start_time
    coeff_df["validation_end_time"] = validation_end_time
    coeff_df["n_calibration_obs"] = int(np.sum(calibration_mask))
    coeff_df["n_validation_obs"] = int(np.sum(validation_mask))
    coeff_df.to_csv(output_coefficients_csv, index=False)
    return merged, coeff_df


if __name__ == "__main__":
    run_dvv_transfer_workflow(
        pore_pressure_csv_path=PORE_PRESSURE_CSV_PATH,
        model_depth_m=MODEL_DEPTH_M,
        include_thermoelastic=INCLUDE_THERMOELASTIC,
        fit_intercept=FIT_INTERCEPT,
        calibration_start_time=CALIBRATION_START_TIME,
        calibration_end_time=CALIBRATION_END_TIME,
        validation_start_time=VALIDATION_START_TIME,
        validation_end_time=VALIDATION_END_TIME,
        output_timeseries_csv=OUTPUT_TIMESERIES_CSV,
        output_coefficients_csv=OUTPUT_COEFFICIENTS_CSV,
    )
    print("\nFinished!")
