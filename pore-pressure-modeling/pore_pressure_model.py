#
# Pore Pressure Modeling
# HM Huang, 2026
#
#
import numpy as np
import pandas as pd
import os
import warnings
from scipy.signal import fftconvolve
from scipy.special import erfc, erf
from dvv_model_utils import SECONDS_PER_YEAR, fit_annual_harmonic

# ========== #
# User Input #
# ========== #

### Global settings ###
GWL_CSV_PATH = "/Users/hmhuang/Sth_should_be_local/Iceland/datasets/Well_Data_confidential/HSK-15_confidential_data.csv"
ATM_TXT_PATH = "/Users/hmhuang/Sth_should_be_local/Iceland/datasets/sta_990_atm.txt"
START_TIME = "2021-03-01 00:00:00"
END_TIME = "2021-12-31 23:59:59"
RESAMPLE_RULE = "1D"
# sampling rate of the final result 

DEPTHS_M = (0.4,)


### gwl and atm modeling parameters ###
RHO_W = 1000.0
GRAVITY = 9.8
# calculateing rho*g*h

SKEMPTON_B = 0.8
UNDRAINED_POISSON_RATIO = 0.25
# α = B(1+νu)/3(1-νu) is an elastic constant linked to the Skempton’s coefficient B and the
# undrained Poisson ratio νu. (Roeloffs, 1988; Rivet et al., 2015)

HYDRAULIC_DIFFUSIVITY_M2_S = 0.1
# c in pore pressure equation

### thermoelastic modeling parameters ###
POISSON_RATIO = 0.25
# the regular poisson ratio is for thermoelastic modeling
# not sure the physical meaning of using different poisson ratio
# so i just set the same number here for now

YOUNGS_MODULUS_PA = 3.0e10
# thermoelastic branch uses Young's modulus as the primary elastic input
# it is for calculating shear modulus (Lame parameters?)
# no big diff on results by trying 1.0e10, 2.0e10, and 3.0e10
# I think it can be lower since we are studying the area has many fractures?

M_OVER_MU_RATIO = -5000
# second Murnaghan constant divided by shear modulus
# by doing this, we dont have to define murnaghan constant
# it basically controls how strongly strain affects dv/v
# the suggested range of number would be -100 to -10k
# based on my testing, the variation of the thermo term did not change
# but the scale was changed. 
# you will get smaller number if the ratio is -100
# you will get greater number if the ratio is -10k
# the final result did not show significant difference

HORIZONTAL_WAVENUMBER_M_INV = 2.0 * np.pi / (1.0e4)
# a wavenumber as a proxy of spatial resolution of temperature  
# how large of the horizontal area affected by temperature
# for long-wavelength, the wavenumber is smaller, so the response of temperature will be smaller  
# the form is 2*pi/r, and the unit of r is meters

THERMOELASTIC_DEPTH_M = 0.4
# straitforward, just set a depth of diffused temperature
# my dv/v is of a general range (0.25-2Hz) so not direct to 
# any depth so far. However, if i set it as 0 (surface),
# it's gonna larger the thermo effect from dv/v
# the difference is small so i just leave 0.0 here for now.

THERMAL_EXPANSION_COEFF_C_INV = 1.0e-5

THERMAL_DIFFUSIVITY_M2_S = 1.0e-6
INCOMPETENT_LAYER_THICKNESS_M = 10.4

# Output files
OUTPUT_DIR = "../output"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "pore_pressure_output.csv")

# =============== #
# Data Management #
# =============== #
def read_groundwater_csv(filepath):
    """
    Read groundwater and well-temperature data from the well CSV.
    """
    df = pd.read_csv(
        filepath,
        usecols=["datetime", "temperature [°C]", "groundwater level [m a.s.l.]"],
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["temperature [°C]"] = pd.to_numeric(df["temperature [°C]"], errors="coerce")
    df["groundwater level [m a.s.l.]"] = pd.to_numeric(df["groundwater level [m a.s.l.]"])
    df.loc[df["groundwater level [m a.s.l.]"].isin([-777, -777.0]), "groundwater level [m a.s.l.]"] = np.nan
    return (
        df.rename(
            columns={
                "temperature [°C]": "well_temp_c",
                "groundwater level [m a.s.l.]": "gwl_m_asl",
            }
        )
        .dropna(subset=["gwl_m_asl", "well_temp_c"])
        .set_index("datetime")
        .sort_index()
    )

def read_imo_monthly_pressure_txt(filepath):
    """
    Read the air pressure data
    """
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        next(handle)
        next(handle)
        for line in handle:
            parts = line.strip().split()
            year = int(parts[1])
            month = int(parts[2])
            p_hpa = float(parts[14])
            rows.append((year, month, p_hpa))

    df = pd.DataFrame(rows, columns=["year", "month", "p_hpa"])
    df["time"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=np.ones(len(df), dtype=int))
    )
    df = df.set_index("time").sort_index()
    df["patm_pa"] = df["p_hpa"] * 100.0
    return df[["patm_pa"]]

def infer_median_dt_seconds(index):
    """
    Calculate time step of original datasets
    """
    if len(index) < 2:
        return None
    dt_s = float(np.median(np.diff(index).astype("timedelta64[s]").astype(float)))
    if dt_s <= 0:
        return None
    return dt_s

def warn_if_source_resolution_is_coarser(
    dataset_name,
    filepath,
    source_dt_s,
    target_dt_s,
    target_sampling_label,
    action_message,
):
    """
    Warning users the original temporal resolution is lower than users' setting
    It still runs it but the results should be double-checked 
    or just change the final temporal resolution (RESAMPLE_RULE)     
    """
    if source_dt_s is None or target_dt_s >= source_dt_s:
        return
    file_line = f"File: {filepath}\n" if filepath is not None else ""
    warnings.warn(
        "Input data resolution mismatch.\n"
        f"Dataset: {dataset_name}\n"
        f"Requested final sampling: {target_sampling_label}\n"
        f"{file_line}"
        f"Action: {action_message}",
        stacklevel=1,
    )

def prepare_time_series(
    source_df,
    start=None,
    end=None,
    rule=None,
    target_index=None,
    interp_method="time",
    dataset_name="input data",
    filepath=None,
):
    """
    align the original time steps to the final one
    if the original one has higher sampling rate, then it will downsample
    if the original one has lower sampling rate, then it will interpolate
    """

    series = source_df.iloc[:, 0].copy()
    series = series.loc[start:end].copy()
    if series.empty:
        raise ValueError(f"No {dataset_name} data in the requested time window")
    if rule is None:
        raise ValueError("rule must be provided")

    if target_index is None:
        series = series.resample(rule).mean().dropna()
        if series.empty:
            raise ValueError(f"{dataset_name} data became empty after resampling")
        return series

    source_dt_s = infer_median_dt_seconds(series.index)
    target_dt_s = infer_median_dt_seconds(target_index)
    if source_dt_s is not None and target_dt_s is not None and source_dt_s > target_dt_s:
        warn_if_source_resolution_is_coarser(
            dataset_name=dataset_name,
            filepath=filepath,
            source_dt_s=source_dt_s,
            target_dt_s=target_dt_s,
            target_sampling_label=rule,
            action_message="the dataset will be interpolated to the final time grid.",
        )
        expanded = series.reindex(series.index.union(target_index)).sort_index()
        expanded = expanded.interpolate(method=interp_method).ffill().bfill()
        return expanded.reindex(target_index).ffill().bfill()
    series = series.resample(rule).mean().dropna()
    if series.empty:
        raise ValueError(f"{dataset_name} data became empty after resampling")
    return series.reindex(target_index).ffill().bfill()

# ================== #
# Diffusion Equation #
# ================== #
# NTAL
# Non tidal atm loading
# correction

# NTOL
# Non tidal ocean loading
# TLA, TOL

def diffusion_kernel(depth_m, dt_s, n_samples, hydraulic_diffusivity_m2_s):
    """
    Calculate the complementary error function
    """
    lag_s = np.arange(n_samples, dtype=float) * dt_s
    kernel = np.zeros(n_samples, dtype=float)

    positive_lag = lag_s > 0
    kernel[positive_lag] = erfc(
        depth_m / np.sqrt(4.0 * hydraulic_diffusivity_m2_s * lag_s[positive_lag])
    )
    if depth_m == 0:
        kernel[0] = 1.0

    return kernel

def compute_poroelastic_alpha(skempton_b, undrained_poisson_ratio):
    """
    Calculating alpha
    """
    denominator = 3.0 * (1.0 - undrained_poisson_ratio)
    if denominator == 0:
        raise ValueError("Invalid undrained Poisson ratio: denominator became zero")
    poroelastic_alpha = (
        skempton_b * (1.0 + undrained_poisson_ratio) / denominator
    )
    if not 0.0 <= poroelastic_alpha <= 1.0:
        raise ValueError(
            "Computed poroelastic alpha is outside [0, 1]. "
            "Check SKEMPTON_B and UNDRAINED_POISSON_RATIO."
        )
    return poroelastic_alpha


def compute_shear_modulus(youngs_modulus_pa, poisson_ratio):
    """
    Calculating shear modulus from poisson ratio
    """
    denominator = 2.0 * (1.0 + poisson_ratio)
    if denominator == 0:
        raise ValueError("Invalid Poisson ratio: denominator became zero")
    return youngs_modulus_pa / denominator

def compute_second_murnaghan_constant(shear_modulus_pa, m_over_mu_ratio):
    """
    Calculating murnaghan constant
    """
    return shear_modulus_pa * m_over_mu_ratio


def build_default_loadings(gwl_m_asl, patm_pa, rho_w, g, poroelastic_alpha):
    """
    Calculate pressure from gwl (m) 
    also grab pressure from air pressure data
    """
    p_gwl_pa = rho_w * g * np.asarray(gwl_m_asl, dtype=float)
    p_atm_pa = np.asarray(patm_pa, dtype=float)
    return [
        {"name": "gwl", "values_pa": p_gwl_pa, "poroelastic_alpha": poroelastic_alpha},
        {"name": "atm", "values_pa": p_atm_pa, "poroelastic_alpha": poroelastic_alpha},
    ]

def compute_loading_response(
    loading_pa, depth_m, dt_s, hydraulic_diffusivity_m2_s, poroelastic_alpha
):
    """
    Compute pore-pressure response using the Rivet (2015) / Roeloffs (1988)
    superposition form:

    sum_i [alpha * dp_i * erf(x_i) + dp_i * erfc(x_i)]

    with x_i = z / sqrt(4 c tau_i)
    """
    loading_pa = np.asarray(loading_pa, dtype=float)
    if not 0.0 <= poroelastic_alpha <= 1.0:
        raise ValueError("poroelastic_alpha must be between 0 and 1")

    dp = np.zeros_like(loading_pa)
    dp[0] = loading_pa[0]
    dp[1:] = np.diff(loading_pa)

    lag_s = np.arange(len(loading_pa), dtype=float) * dt_s
    kernel = np.zeros(len(loading_pa), dtype=float)
    positive_lag = lag_s > 0
    x = np.zeros(len(loading_pa), dtype=float)
    x[positive_lag] = depth_m / np.sqrt(
        4.0 * hydraulic_diffusivity_m2_s * lag_s[positive_lag]
    )
    # you might notice that the equation has a (n-i)
    # that is already included in lag_s

    kernel[positive_lag] = (
        poroelastic_alpha * erf(x[positive_lag]) + erfc(x[positive_lag])
    )
    if depth_m == 0:
        kernel[0] = 1.0
        # meaning no diffusion on the surface?
    else:
        kernel[0] = poroelastic_alpha

    return fftconvolve(dp, kernel, mode="full")[: len(loading_pa)]

def infer_regular_dt_seconds(index):
    """
    Calculate time step of final results
    for example:
    
    1h >> 3600s
    1D >> 84600s
    """
    t_s = (index - index[0]).total_seconds().astype(float)
    dt_s = float(np.median(np.diff(t_s)))
    if dt_s <= 0:
        raise ValueError("Time index must be increasing")
    return t_s, dt_s


def compute_tsai_thermoelastic_response(
    index,
    temperature_c,
    poisson_ratio,
    shear_modulus_pa,
    second_murnaghan_pa,
    wavenumber_m_inv,
    depth_m,
    thermal_expansion_coeff_c_inv,
    thermal_diffusivity_m2_s,
    incompetent_layer_thickness_m,
    period_s=SECONDS_PER_YEAR,
):
    harmonic = fit_annual_harmonic(index, temperature_c, period_s=period_s)
    t_s = (pd.to_datetime(index) - pd.to_datetime(index)[0]).total_seconds().astype(float)
    omega = harmonic["omega"]
    temp_phase_rad = harmonic["phase_rad"]
    thermal_lag_s = (
        np.pi / (4.0 * omega)
        + incompetent_layer_thickness_m / np.sqrt(2.0 * omega * thermal_diffusivity_m2_s)
    )
    strain_prefactor = (
        ((1.0 + poisson_ratio) / (1.0 - poisson_ratio))
        * wavenumber_m_inv
        * thermal_expansion_coeff_c_inv
        * harmonic["amplitude"]
        * np.sqrt((thermal_diffusivity_m2_s / omega) * np.pi / 4.0)
    )
    strain = strain_prefactor * np.cos(omega * t_s + temp_phase_rad - omega * thermal_lag_s)
    dvv_thermo = (
        (second_murnaghan_pa / shear_modulus_pa)
        * strain
        * np.exp(-wavenumber_m_inv * depth_m)
        * (1.0 - 2.0 * poisson_ratio)
    )
    return pd.Series(dvv_thermo, index=index, name="dvv_temp_thermoelastic")

def run_pore_pressure_workflow(
    gwl_csv_path,
    atm_txt_path,
    output_csv_path,
    start="2021-03-01",
    end="2021-12-31 23:59:59",
    resample_rule="1D",
    depths_m=(500.0,),
    rho_w=None,
    g=None,
    skempton_b=SKEMPTON_B,
    undrained_poisson_ratio=UNDRAINED_POISSON_RATIO,
    hydraulic_diffusivity_m2_s=None,
    poisson_ratio=POISSON_RATIO,
    youngs_modulus_pa=YOUNGS_MODULUS_PA,
    m_over_mu_ratio=M_OVER_MU_RATIO,
    horizontal_wavenumber_m_inv=HORIZONTAL_WAVENUMBER_M_INV,
    thermoelastic_depth_m=THERMOELASTIC_DEPTH_M,
    thermal_expansion_coeff_c_inv=THERMAL_EXPANSION_COEFF_C_INV,
    thermal_diffusivity_m2_s=THERMAL_DIFFUSIVITY_M2_S,
    incompetent_layer_thickness_m=INCOMPETENT_LAYER_THICKNESS_M,
):
    """
    Run the functions above all together to get the result
    Output is a csv file containing modeled pore pressure at different depth
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    model_index = pd.date_range(start=start, end=end, freq=resample_rule)
    if len(model_index) < 2:
        raise ValueError("The requested model time grid must contain at least two samples")

    gwl = read_groundwater_csv(gwl_csv_path)
    gwl_rs = prepare_time_series(
        gwl[["gwl_m_asl"]],
        start=start,
        end=end,
        rule=resample_rule,
        target_index=model_index,
        dataset_name="groundwater",
        filepath=gwl_csv_path,
    )
    _, dt_s = infer_regular_dt_seconds(model_index)
    temp_rs = prepare_time_series(
        gwl[["well_temp_c"]],
        start=start,
        end=end,
        rule=resample_rule,
        target_index=model_index,
        dataset_name="well temperature",
        filepath=gwl_csv_path,
    )

    # the gwl data aligns with the final temporal resolution (defined by users from RULE)

    patm_monthly = read_imo_monthly_pressure_txt(atm_txt_path)
    patm_rs = prepare_time_series(
        patm_monthly,
        start=start,
        end=end,
        rule=resample_rule,
        target_index=model_index,
        dataset_name="atmospheric pressure",
        filepath=atm_txt_path,
    )
    
    # bc the gwl data has aligned with the final temporal resolution,
    # the atm can just align with it

    poroelastic_alpha = compute_poroelastic_alpha(
        skempton_b=skempton_b,
        undrained_poisson_ratio=undrained_poisson_ratio,
    )
    shear_modulus_pa = compute_shear_modulus(
        youngs_modulus_pa=youngs_modulus_pa,
        poisson_ratio=poisson_ratio,
    )
    second_murnaghan_pa = compute_second_murnaghan_constant(
        shear_modulus_pa=shear_modulus_pa,
        m_over_mu_ratio=m_over_mu_ratio,
    )
    loadings = build_default_loadings(
        gwl_rs.values,
        patm_rs.values,
        rho_w,
        g,
        poroelastic_alpha,
    )

    depths_m = tuple(float(depth) for depth in depths_m)

    out = pd.DataFrame(index=model_index)
    out["gwl_m_asl"] = gwl_rs.values
    out["well_temp_c"] = temp_rs.values
    out["patm_pa"] = patm_rs.values
    out["poroelastic_alpha"] = poroelastic_alpha
    out["shear_modulus_pa"] = shear_modulus_pa
    out["second_murnaghan_pa"] = second_murnaghan_pa
    out["dvv_temp_thermoelastic"] = compute_tsai_thermoelastic_response(
        index=model_index,
        temperature_c=temp_rs.values,
        poisson_ratio=poisson_ratio,
        shear_modulus_pa=shear_modulus_pa,
        second_murnaghan_pa=second_murnaghan_pa,
        wavenumber_m_inv=horizontal_wavenumber_m_inv,
        depth_m=thermoelastic_depth_m,
        thermal_expansion_coeff_c_inv=thermal_expansion_coeff_c_inv,
        thermal_diffusivity_m2_s=thermal_diffusivity_m2_s,
        incompetent_layer_thickness_m=incompetent_layer_thickness_m,
    ).values

    for loading in loadings:
        out[f"{loading['name']}_loading_pa"] = loading["values_pa"]
        for depth_m in depths_m:
            response = compute_loading_response(
                loading_pa=loading["values_pa"],
                depth_m=depth_m,
                dt_s=dt_s,
                hydraulic_diffusivity_m2_s=hydraulic_diffusivity_m2_s,
                poroelastic_alpha=loading["poroelastic_alpha"],
            )
            out[f"Pp_{loading['name']}_z{int(depth_m)}m_pa"] = response

    for depth_m in depths_m:
        total_pp = np.zeros(len(out), dtype=float)
        for loading in loadings:
            total_pp += out[f"Pp_{loading['name']}_z{int(depth_m)}m_pa"].values
        out[f"Pp_total_z{int(depth_m)}m_pa"] = total_pp
        out[f"dPp_total_z{int(depth_m)}m_pa"] = total_pp - np.nanmean(total_pp)

    out.to_csv(output_csv_path, index=True, index_label="datetime")
    return out

if __name__ == "__main__":
    result = run_pore_pressure_workflow(
        gwl_csv_path=GWL_CSV_PATH,
        atm_txt_path=ATM_TXT_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        start=START_TIME,
        end=END_TIME,
        resample_rule=RESAMPLE_RULE,
        depths_m=DEPTHS_M,
        rho_w=RHO_W,
        g=GRAVITY,
        skempton_b=SKEMPTON_B,
        undrained_poisson_ratio=UNDRAINED_POISSON_RATIO,
        hydraulic_diffusivity_m2_s=HYDRAULIC_DIFFUSIVITY_M2_S,
        youngs_modulus_pa=YOUNGS_MODULUS_PA,
        m_over_mu_ratio=M_OVER_MU_RATIO,
    )

print("\nFinished!")
print("Please read the warning carefully!\n")
