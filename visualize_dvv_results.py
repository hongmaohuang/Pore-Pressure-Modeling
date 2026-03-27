#
# Visualize dv/v workflow outputs
# HM Huang, 2026
#
import os

import matplotlib.pyplot as plt
import pandas as pd


# ========== #
# User Input #
# ========== #
OUTPUT_DIR = "../output"
DVV_TRANSFER_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "dvv_transfer_output.csv")
DVV_MODEL_METRICS_CSV = os.path.join(OUTPUT_DIR, "dvv_model_metrics.csv")

FIG_OBS_MODEL = os.path.join(OUTPUT_DIR, "fig_dvv_observed_vs_modeled.png")
FIG_RESIDUAL = os.path.join(OUTPUT_DIR, "fig_dvv_residual.png")
FIG_PREDICTORS = os.path.join(OUTPUT_DIR, "fig_dvv_predictors.png")

OBSERVED_MEDIAN_FILTER_WINDOW = 7
Y_LIM_DVV = 0.003


def read_transfer_output(filepath):
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()


def add_filtered_observed_column(df, window):
    df = df.copy()
    if window is None or window <= 1:
        df["dvv_obs_filtered"] = df["dvv_obs"]
        return df
    df["dvv_obs_filtered"] = (
        df["dvv_obs"].rolling(window=window, center=True, min_periods=1).median()
    )
    return df


def make_observed_modeled_plot(df, output_path, y_lim):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df.index,
        df["dvv_obs"],
        label="Observed dv/v (raw)",
        color="0.75",
        linewidth=0.9,
    )
    ax.plot(
        df.index,
        df["dvv_obs_filtered"],
        label="Observed dv/v (median filtered)",
        color="black",
        linewidth=1.0,
    )
    ax.plot(df.index, df["dvv_model"], label="Modeled dv/v", color="tab:red", linewidth=1.0)
    ax.set_title("Observed vs Modeled dv/v")
    ax.set_ylabel("dv/v")
    ax.set_xlabel("Time")
    ax.set_ylim(-y_lim, y_lim)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_residual_plot(df, output_path, y_lim):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["dvv_residual"], color="tab:blue", linewidth=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("dv/v Residual")
    ax.set_ylabel("Observed - Modeled")
    ax.set_xlabel("Time")
    ax.set_ylim(-y_lim, y_lim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_predictor_plot(df, output_path):
    predictor_columns = [
        col
        for col in ["pp_gwl_pa", "pp_atm_pa", "dvv_temp_thermoelastic"]
        if col in df.columns
    ]
    fig, axes = plt.subplots(len(predictor_columns), 1, figsize=(12, 3.2 * len(predictor_columns)), sharex=True)
    if len(predictor_columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, predictor_columns):
        ax.plot(df.index, df[column], linewidth=0.9)
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Physical Predictors Used in dv/v Projection")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_visualization_workflow(dvv_transfer_output_csv):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = read_transfer_output(dvv_transfer_output_csv)
    df = add_filtered_observed_column(df, OBSERVED_MEDIAN_FILTER_WINDOW)
    make_observed_modeled_plot(df, FIG_OBS_MODEL, Y_LIM_DVV)
    make_residual_plot(df, FIG_RESIDUAL, Y_LIM_DVV)
    make_predictor_plot(df, FIG_PREDICTORS)
    return {
        "observed_modeled_figure": FIG_OBS_MODEL,
        "residual_figure": FIG_RESIDUAL,
        "predictor_figure": FIG_PREDICTORS,
    }


if __name__ == "__main__":
    run_visualization_workflow(DVV_TRANSFER_OUTPUT_CSV)
    print("\nFinished!")
