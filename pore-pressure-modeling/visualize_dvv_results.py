#
# Visualize dv/v workflow outputs
# HM Huang, 2026
#
import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 11

# ========== #
# User Input #
# ========== #
OUTPUT_DIR = "../output"
DVV_TRANSFER_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "dvv_transfer_output.csv")
DVV_MODEL_METRICS_CSV = os.path.join(OUTPUT_DIR, "dvv_model_metrics.csv")
SEISMICITY_CSV = "/Users/hmhuang/Sth_should_be_local/Iceland/datasets/seismic_catalog_202103-12.txt"
SEISMICITY_MAGNITUDE_THRESHOLD = 1.0

FIG_OBS_MODEL = os.path.join(OUTPUT_DIR, "fig_dvv_observed_vs_modeled.png")
FIG_RESIDUAL = os.path.join(OUTPUT_DIR, "fig_dvv_residual.png")
FIG_PREDICTORS = os.path.join(OUTPUT_DIR, "fig_dvv_predictors.png")

OBSERVED_MEDIAN_FILTER_WINDOW = 14
Y_LIM_DVV = 0.8
THERMO_PLOT_SCALE = 1.0e6


def read_transfer_output(filepath):
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()


def read_seismicity_counts(filepath, target_index, magnitude_threshold):
    df = pd.read_csv(filepath, parse_dates=["time"])
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    df = df[df["magnitude"] > magnitude_threshold]
    daily_counts = df.resample("1D").size().rename("event_count")
    target_daily = pd.DatetimeIndex(pd.to_datetime(target_index).normalize().unique()).sort_values()
    return daily_counts.reindex(target_daily, fill_value=0)


def add_filtered_observed_column(df, window):
    df = df.copy()
    target_column = "dvv_obs"
    if window is None or window <= 1:
        df["dvv_obs_filtered"] = df[target_column]
        return df
    df["dvv_obs_filtered"] = (
        df[target_column].rolling(window=window, center=True, min_periods=1).median()
    )
    return df


def add_split_shading(ax, df):
    split_styles = {
        "calibration": ("tab:blue", 0.06),
        "validation": ("tab:orange", 0.06),
    }
    for split_name, (color, alpha) in split_styles.items():
        split_df = df[df["data_split"] == split_name]
        if split_df.empty:
            continue
        ax.axvspan(
            split_df.index.min(),
            split_df.index.max(),
            color=color,
            alpha=alpha,
            linewidth=0,
        )


def make_observed_modeled_plot(df, output_path, y_lim):
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax_pp = ax.twinx()
    ax_temp = ax.twinx()
    ax_gwl = ax.twinx()
    ax_atm = ax.twinx()
    ax_temp.spines["right"].set_position(("axes", 1.10))
    ax_gwl.spines["right"].set_position(("axes", 1.20))
    ax_atm.spines["right"].set_position(("axes", 1.30))
    add_split_shading(ax, df)
    ax.plot(
        df.index,
        df["dvv_obs_raw"] if "dvv_obs_raw" in df.columns else df["dvv_obs"],
        label="Observed dv/v (raw)",
        color="0.75",
        linewidth=1.0,
    )
    ax.plot(
        df.index,
        df["dvv_obs_filtered"],
        label="Observed dv/v (band-summed target)",
        color="black",
        linewidth=1.0,
    )
    ax.plot(df.index, df["dvv_model"], label="Modeled dv/v", color="tab:red", linewidth=1.0)
    ax.set_title("Observed vs Modeled dv/v")
    ax.set_ylabel("dv/v (%)")
    ax.set_ylim(-y_lim, y_lim)
    ax.grid(True, alpha=0.3)

    if "pp_total_pa" in df.columns:
        ax_pp.plot(
            df.index,
            df["pp_total_pa"],
            color="tab:purple",
            linewidth=1.0,
            label="Total hydraulic pore pressure",
        )
    ax_pp.set_ylabel("Pore pressure (Pa)", color="tab:purple")
    ax_pp.tick_params(axis="y", colors="tab:purple")

    if "pp_gwl_pa" in df.columns:
        ax_gwl.plot(
            df.index,
            df["pp_gwl_pa"],
            color="tab:blue",
            linewidth=0.9,
            linestyle="-.",
            label="Groundwater pore pressure",
        )
        ax_gwl.set_ylabel("GWL pore pressure (Pa)", color="tab:blue")
        ax_gwl.tick_params(axis="y", colors="tab:blue")

    if "pp_atm_pa" in df.columns:
        ax_atm.plot(
            df.index,
            df["pp_atm_pa"],
            color="tab:orange",
            linewidth=0.9,
            linestyle=":",
            label="Atmospheric pore pressure",
        )
        ax_atm.set_ylabel("ATM pore pressure (Pa)", color="tab:orange")
        ax_atm.tick_params(axis="y", colors="tab:orange")

    if "dvv_temp_thermoelastic" in df.columns:
        ax_temp.plot(
            df.index,
            df["dvv_temp_thermoelastic"] * THERMO_PLOT_SCALE,
            color="tab:green",
            linewidth=0.9,
            linestyle="--",
            label="Thermoelastic predictor",
        )
        ax_temp.set_ylabel("Thermo predictor (x10^-6 %)", color="tab:green")
        ax_temp.tick_params(axis="y", colors="tab:green")

    ax.set_xlabel("Time")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_pp.get_legend_handles_labels()
    handles3, labels3 = ax_temp.get_legend_handles_labels()
    handles4, labels4 = ax_gwl.get_legend_handles_labels()
    handles5, labels5 = ax_atm.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2 + handles3 + handles4 + handles5,
        labels1 + labels2 + labels3 + labels4 + labels5,
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_residual_plot(df, seismicity_counts, output_path, y_lim, window):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax2 = ax.twinx()
    ax2.bar(
        seismicity_counts.index,
        seismicity_counts.values,
        width=0.9,
        color="0.8",
        alpha=0.6,
        label="Daily seismicity count (M > 3)",
    )
    ax2.set_ylabel("Event count")
    ax2.set_ylim(0, 600)

    residual_filtered = df["dvv_residual"].rolling(
        window=window,
        center=True,
        min_periods=1,
    ).median()
    ax.fill_between(
        df.index,
        0.0,
        residual_filtered,
        where=residual_filtered >= 0.0,
        color="tab:blue",
        alpha=0.25,
        interpolate=True,
        label="Positive residual",
    )
    ax.fill_between(
        df.index,
        0.0,
        residual_filtered,
        where=residual_filtered < 0.0,
        color="tab:red",
        alpha=0.22,
        interpolate=True,
        label="Negative residual",
    )
    ax.plot(
        df.index,
        residual_filtered,
        color="black",
        linewidth=0.9,
        label="Median-filtered residual",
    )
    ax.set_title("dv/v Residual")
    ax.set_ylabel("Observed - Modeled (%)")
    ax.set_xlabel("Time")
    ax.set_ylim(-y_lim, y_lim)
    add_split_shading(ax, df)
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
    seismicity_counts = read_seismicity_counts(
        SEISMICITY_CSV,
        df.index,
        SEISMICITY_MAGNITUDE_THRESHOLD,
    )
    make_observed_modeled_plot(df, FIG_OBS_MODEL, Y_LIM_DVV)
    make_residual_plot(
        df,
        seismicity_counts,
        FIG_RESIDUAL,
        Y_LIM_DVV,
        OBSERVED_MEDIAN_FILTER_WINDOW,
    )
    make_predictor_plot(df, FIG_PREDICTORS)
    return {
        "observed_modeled_figure": FIG_OBS_MODEL,
        "residual_figure": FIG_RESIDUAL,
        "predictor_figure": FIG_PREDICTORS,
    }


if __name__ == "__main__":
    run_visualization_workflow(DVV_TRANSFER_OUTPUT_CSV)
    print("\nFinished!")
