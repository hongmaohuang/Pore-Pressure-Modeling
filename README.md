# Pore Pressure, Thermoelastic, and dv/v Transfer Workflow

This branch contains a small workflow for:

1. Building physical predictors from groundwater level, atmospheric pressure, and well temperature
2. Projecting those predictors to theoretical `dv/v(t)`
3. Evaluating the modeled `dv/v` against observed `dv/v`

The implementation is organized into four scripts:

- `pore_pressure_model.py`
- `dvv_transfer_model.py`
- `evaluate_dvv_model.py`
- `visualize_dvv_results.py`

An additional helper module is shared by the workflow:

- `dvv_model_utils.py`

## Workflow Overview

The workflow is intentionally split by responsibility:

1. `pore_pressure_model.py`
   Builds the physical predictors on a common time grid.

2. `dvv_transfer_model.py`
   Reads those predictors and fits a transfer model to observed `dv/v`.

3. `evaluate_dvv_model.py`
   Computes fit statistics, annual amplitude/phase metrics, and rolling cross-validation.

4. `visualize_dvv_results.py`
   Produces figures from the modeled and observed `dv/v` outputs.

The current physical branches are:

- Groundwater loading -> hydraulic pore-pressure response
- Atmospheric pressure loading -> hydraulic pore-pressure response
- Well temperature -> thermoelastic `dv/v` predictor

The current modeling logic does not force temperature into the hydraulic pore-pressure branch. Instead, temperature is treated as a separate thermoelastic mechanism and is combined with the pressure-driven branches only at the `dv/v` projection stage.

## Script Roles

### `pore_pressure_model.py`

This script:

- Reads the well CSV
- Reads the atmospheric pressure text file
- Builds an explicit regular model time grid using `pd.date_range(start, end, freq=RESAMPLE_RULE)`
- Aligns all data sets to that final user-defined time resolution
- Computes pore-pressure responses from:
  - groundwater level
  - atmospheric pressure
- Computes a thermoelastic predictor from well temperature using a Tsai (2011)-style harmonic formulation
- Writes a single output CSV that contains the aligned observations and all physical predictors

Typical output columns include:

- `gwl_m_asl`
- `well_temp_c`
- `patm_pa`
- `gwl_loading_pa`
- `atm_loading_pa`
- `Pp_gwl_z500m_pa`, `Pp_gwl_z700m_pa`, `Pp_gwl_z1900m_pa`
- `Pp_atm_z500m_pa`, `Pp_atm_z700m_pa`, `Pp_atm_z1900m_pa`
- `Pp_total_z500m_pa`, ...
- `dPp_total_z500m_pa`, ...
- `dvv_temp_thermoelastic`

### `dvv_transfer_model.py`

This script:

- Reads the predictor CSV from `pore_pressure_model.py`
- Reads observed `dv/v`
- Builds the regression design matrix from:
  - groundwater pore-pressure predictor
  - atmospheric pore-pressure predictor
  - thermoelastic predictor
- Fits a linear transfer model
- Outputs:
  - modeled `dv/v`
  - residual `dv/v`
  - transfer coefficients

Observed `dv/v` can currently be read in two ways:

- `CSV`
- `DTT_FOLDER`

For the current branch settings, the script is configured to read from a DTT results folder and to use the median of all non-autocorrelation station pairs when `DVV_PAIR_MODE = "ALL"`.

### `evaluate_dvv_model.py`

This script:

- Reads the modeled and observed `dv/v` output from `dvv_transfer_model.py`
- Computes:
  - RMSE
  - MAE
  - correlation coefficient
  - `R^2`
  - RSS
  - AIC
  - BIC
  - annual amplitude
  - annual phase
  - annual phase difference
- Performs rolling-origin cross-validation

### `visualize_dvv_results.py`

This script:

- Reads the output from `dvv_transfer_model.py`
- Produces PNG figures for:
  - observed vs modeled `dv/v`
  - `dv/v` residual
  - physical predictors used in the transfer model

The current plotting script writes the figures to the common output folder:

- `../output/fig_dvv_observed_vs_modeled.png`
- `../output/fig_dvv_residual.png`
- `../output/fig_dvv_predictors.png`

In the current plotting defaults:

- the observed `dv/v` is shown both as a raw gray curve and as a median-filtered black curve
- the observed-vs-modeled and residual plots use a fixed y-axis range of `±0.3%` (`±0.003` in `dv/v` units)

## Shared Helper Module

### `dvv_model_utils.py`

This module provides reusable utilities for:

- elapsed-time conversion
- annual harmonic fitting
- linear model fitting
- model prediction
- information criteria
- basic performance metrics
- rolling-origin cross-validation

## Main Equations Used

### 1. Hydraulic pressure loading from groundwater

Groundwater level is converted to pressure loading by:

```text
P_gwl(t) = rho_w * g * h(t)
```

where:

- `rho_w` is water density
- `g` is gravity
- `h(t)` is groundwater level

### 2. Hydraulic diffusion kernel

The pore-pressure response uses a complementary error function kernel:

```text
K(z, tau) = erfc(z / sqrt(4 * c * tau))
```

where:

- `z` is depth
- `tau` is lag time
- `c` is hydraulic diffusivity

### 3. Mixed drained / undrained response

For a loading time series `L(t)`, the current implementation computes:

```text
dL(t) = L(t) - L(t-1)
```

and then:

```text
Response(t) = (1 - alpha) * [dL * K](t) + alpha * L(t)
```

where:

- `alpha` is the direct-fraction mixing term
- `[dL * K](t)` denotes convolution with the diffusion kernel

This formulation is applied to:

- groundwater pressure loading
- atmospheric pressure loading

### 4. Total pore pressure

At each modeled depth:

```text
Pp_total(z, t) = Pp_gwl(z, t) + Pp_atm(z, t)
```

The script also writes the mean-removed total pore-pressure anomaly:

```text
dPp_total(z, t) = Pp_total(z, t) - mean(Pp_total(z, t))
```

### 5. Thermoelastic predictor

The thermoelastic branch follows the Luo et al. (2025) supplementary formulation based on Tsai (2011).

The implementation first fits an annual harmonic to the well-temperature record:

```text
T(t) ~= T_mean + A_T * cos(omega * t) + B_T * sin(omega * t)
```

Then the thermoelastic strain amplitude is represented as:

```text
A(t) = ((1 + nu) / (1 - nu)) * k * alpha_th * T0 * sqrt((kappa_th / omega) * pi / 4)
       * cos(omega * (t - Delta_t))
```

with lag:

```text
Delta_t = pi / (4 * omega) + y_s / sqrt(2 * omega * kappa_th)
```

The thermoelastic `dv/v` predictor is then approximated by:

```text
dv/v_thermo(t) ~= (m / mu) * A(t) * exp(-k * z) * (1 - 2 * nu)
```

where:

- `nu` is Poisson's ratio
- `mu` is shear modulus
- `m` is the second Murnaghan constant
- `k` is horizontal wavenumber
- `z` is investigation depth
- `alpha_th` is thermal expansion coefficient
- `kappa_th` is thermal diffusivity
- `y_s` is incompetent-layer thickness

This branch is computed in `pore_pressure_model.py` and exported as:

- `dvv_temp_thermoelastic`

### 6. Transfer from predictors to modeled `dv/v`

The current transfer model is linear:

```text
dv/v_model(t) = b0 + b1 * X_gwl(t) + b2 * X_atm(t) + b3 * X_temp(t)
```

where:

- `X_gwl(t)` is the groundwater pore-pressure predictor at the selected depth
- `X_atm(t)` is the atmospheric pore-pressure predictor at the selected depth
- `X_temp(t)` is the thermoelastic predictor
- `b0, b1, b2, b3` are fitted coefficients

The fitted output is written to:

- `dvv_model`
- `dvv_residual = dvv_obs - dvv_model`

## Time-Grid Note

The physical predictors are not indexed by the raw groundwater sampling times. Instead, the workflow explicitly builds a complete regular model grid from:

```text
pd.date_range(start, end, freq=RESAMPLE_RULE)
```

and aligns groundwater, atmospheric pressure, and well temperature to that common grid.

This matters because observed `dv/v` is later joined to the predictor table using exact timestamps. Using a complete regular model grid avoids artificially sparse `dv/v` comparison results caused by irregular groundwater data availability.

## Inputs

### Well data

Expected columns in the groundwater CSV:

- `datetime`
- `temperature [°C]`
- `groundwater level [m a.s.l.]`

Special handling:

- `-777` and `-777.0` are treated as missing groundwater values

### Atmospheric pressure data

Expected input is the monthly IMO text format currently used in this project.

### Observed `dv/v`

The transfer script supports:

- a simple CSV with `datetime` and `dvv`
- a DTT folder containing daily text files with columns such as:
  - `Date`
  - `Pairs`
  - `M0`
  - `EM0`

## How to Run

Run the four scripts in order:

```bash
python pore_pressure_model.py
python dvv_transfer_model.py
python evaluate_dvv_model.py
python visualize_dvv_results.py
```

## Output Directory

All generated CSV files and figures are written to:

```text
../output
```

Each script creates the directory automatically if it does not already exist.

## Configuration Notes

### In `pore_pressure_model.py`

Set:

- `GWL_CSV_PATH`
- `ATM_TXT_PATH`
- `START_TIME`
- `END_TIME`
- `RESAMPLE_RULE`
- `DEPTHS_M`
- `RHO_W`
- `GRAVITY`
- `ALPHA`
- `HYDRAULIC_DIFFUSIVITY_M2_S`
- thermoelastic constants if needed
- `OUTPUT_CSV_PATH`

### In `dvv_transfer_model.py`

Set:

- `PORE_PRESSURE_CSV_PATH`
- `OBSERVED_DVV_INPUT_MODE`
- `OBSERVED_DVV_DTT_FOLDER` or `OBSERVED_DVV_CSV_PATH`
- `DVV_PAIR_MODE`
- `MODEL_DEPTH_M`
- `INCLUDE_THERMOELASTIC`
- `FIT_INTERCEPT`

### In `evaluate_dvv_model.py`

Set:

- `DVV_TRANSFER_OUTPUT_CSV`
- `DVV_TRANSFER_COEFFICIENTS_CSV`
- `OUTPUT_METRICS_CSV`
- `OUTPUT_CV_CSV`
- `N_CV_SPLITS`

## Output Files

### From `pore_pressure_model.py`

- `../output/pore_pressure_output.csv`

### From `dvv_transfer_model.py`

- `../output/dvv_transfer_output.csv`
- `../output/dvv_transfer_coefficients.csv`

### From `evaluate_dvv_model.py`

- `../output/dvv_model_metrics.csv`
- `../output/dvv_model_cv.csv`

### From `visualize_dvv_results.py`

- `../output/fig_dvv_observed_vs_modeled.png`
- `../output/fig_dvv_residual.png`
- `../output/fig_dvv_predictors.png`

## Current Assumptions and Limits

- The hydraulic branch currently uses the same diffusion-style kernel for groundwater and atmospheric pressure loading.
- The thermoelastic branch currently uses a seasonal harmonic formulation rather than a fully transient thermal diffusion inversion.
- The transfer from physical predictors to `dv/v` is currently linear.
- The observed `dv/v` aggregation in `ALL` mode currently uses the median of all non-autocorrelation station pairs per day.

## References

- Luo, B., Zhu, H., & Lumley, D. (2025). Seismic Monitoring of Baseflow and Groundwater Changes in the Yellowstone National Park. Geophysical Research Letters.
- Tsai, V. C. (2011). A model for seasonal changes in GPS positions and seismic wave speeds due to thermoelastic and hydrologic variations.
