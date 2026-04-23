# Pore Pressure, Thermoelastic, and dv/v Transfer Workflow

This branch contains a small workflow for:

1. Building physical predictors from groundwater level, atmospheric pressure, ERA5 snow depth, and well temperature
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
   Reads those predictors and builds an extended Rivet-like transfer model to observed `dv/v`.

3. `evaluate_dvv_model.py`
   Computes fit statistics, annual amplitude/phase metrics, and rolling cross-validation.

4. `visualize_dvv_results.py`
   Produces figures from the modeled and observed `dv/v` outputs.

The current physical branches are:

- Groundwater loading -> hydraulic pore-pressure response
- Atmospheric pressure loading -> hydraulic pore-pressure response
- Snow loading from ERA5 snow depth -> hydraulic pore-pressure response
- Well temperature -> thermoelastic `dv/v` predictor

The current modeling logic does not force temperature into the hydraulic pore-pressure branch. Instead, temperature is treated as a separate thermoelastic mechanism and is combined with the pressure-driven branches only at the `dv/v` projection stage.

## Script Roles

### `pore_pressure_model.py`

This script:

- Reads the well CSV
- Reads the atmospheric pressure file
- Reads the ERA5-Land snow CSV
- Builds an explicit regular model time grid using `pd.date_range(start, end, freq=RESAMPLE_RULE)`
- Aligns all data sets to that final user-defined time resolution
- Computes pore-pressure responses from:
  - groundwater level
  - atmospheric pressure
  - snow depth loading
- Computes a thermoelastic predictor from well temperature using a Tsai (2011)-style harmonic formulation
- Writes a single output CSV that contains the aligned observations and all physical predictors

Typical output columns include:

- `gwl_m_asl`
- `well_temp_c`
- `patm_pa`
- `era5_snow_depth_m`
- `era5_snow_cover`
- `gwl_loading_pa`
- `atm_loading_pa`
- `snow_loading_pa`
- `Pp_gwl_z500m_pa`, `Pp_gwl_z700m_pa`, `Pp_gwl_z1900m_pa`
- `Pp_atm_z500m_pa`, `Pp_atm_z700m_pa`, `Pp_atm_z1900m_pa`
- `Pp_snow_z500m_pa`, `Pp_snow_z700m_pa`, `Pp_snow_z1900m_pa`
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
  - snow pore-pressure predictor
  - thermoelastic predictor
- Band-pass filters each predictor and the observed `dv/v` into Rivet-style period bands
- Estimates band-dependent transfer coefficients using only the configured calibration period
- Applies those coefficients to both the calibration and validation periods
- Outputs:
  - modeled `dv/v`
  - residual `dv/v`
  - transfer coefficients
  - a `data_split` label for each sample (`calibration` or `validation`)
  - predictor-specific modeled contributions

Observed `dv/v` can currently be read in two ways:

- `CSV`
- `DTT_FOLDER`

For the current branch settings, the script is configured to read from a DTT results folder and to use the median of all non-autocorrelation station pairs when `DVV_PAIR_MODE = "ALL"`.
When `DVV_VALUE_COLUMN = "M0"`, the script converts DTT `dt/t` to observed `dv/v (%)` using:

```text
dv/v (%) = -100 * dt/t
```

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
- Writes metrics separately for `all`, `calibration`, and `validation`
- Performs rolling-origin cross-validation

### `visualize_dvv_results.py`

This script:

- Reads the output from `dvv_transfer_model.py`
- Produces PNG figures for:
  - observed vs modeled `dv/v`
  - `dv/v` residual
  - physical predictors used in the transfer model

The current plotting script writes the figures to the common output folder:

- `../output-2/fig_dvv_observed_vs_modeled.png`
- `../output-2/fig_dvv_residual.png`
- `../output-2/fig_dvv_predictors.png`

In the current plotting defaults:

- the observed `dv/v` is shown both as a raw gray curve and as a median-filtered black curve
- the observed-vs-modeled and residual plots use a fixed y-axis range of `±0.3%`
- the residual plot includes daily seismicity counts for events with `M > 3`
- the observed-vs-modeled figure also includes a lower panel showing total hydraulic pore pressure (`gwl + atm + snow`) before transfer, with the thermoelastic predictor shown on a secondary axis when available
- calibration and validation periods are lightly shaded in the `dv/v` comparison and residual plots

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

### 3. Rivet / Roeloffs poroelastic response

For a loading time series `L(t)`, the current implementation computes:

```text
dL(t) = L(t) - L(t-1)
```

and then:

```text
Response(z, t) = sum_i dL_i * [alpha_p * erf(x_i) + erfc(x_i)]
```

where:

- `x_i = z / sqrt(4 * c * tau_i)`
- `tau_i` is the lag time for each increment
- `alpha_p` is the poroelastic constant

This formulation is applied to:

- groundwater pressure loading
- atmospheric pressure loading
- snow loading derived from ERA5 snow depth

The poroelastic constant is computed from:

```text
alpha_p = B * (1 + nu_u) / [3 * (1 - nu_u)]
```

where:

- `B` is Skempton's coefficient
- `nu_u` is the undrained Poisson ratio

### 4. Total pore pressure

At each modeled depth:

```text
Pp_total(z, t) = Pp_gwl(z, t) + Pp_atm(z, t) + Pp_snow(z, t)
```

The script also writes the mean-removed total pore-pressure anomaly:

```text
dPp_total(z, t) = Pp_total(z, t) - mean(Pp_total(z, t))
```

### 5. Snow loading

The snow branch currently uses ERA5-Land time-series `snow depth` (`sde`) as a snow-thickness proxy.

The current implementation converts snow depth to an approximate surface load by:

```text
P_snow(t) = rho_snow * g * H_snow(t)
```

where:

- `rho_snow` is the assumed bulk snow density
- `g` is gravity
- `H_snow(t)` is ERA5 snow depth in meters

The default branch in `pore_pressure_model.py` currently uses:

- `rho_snow = 300 kg m^-3`

This is an approximate loading conversion. It is not equivalent to snow water equivalent unless the density assumption is adjusted accordingly.

### 6. Thermoelastic predictor

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

### 7. Transfer from predictors to modeled `dv/v`

The current implementation uses an extended Rivet-like band transfer.

For each period band `Bi`, the transfer coefficient is estimated from the calibration period using the Rivet-style covariance / variance idea:

```text
K(Bi) ~= cov[dv/v(Bi), X(Bi)] / var[X(Bi)]
```

In the current extended version, this is applied separately to:

- `X_gwl(Bi)`
- `X_atm(Bi)`
- `X_snow(Bi)`
- `X_temp(Bi)`

and the modeled time series is reconstructed by summing the bandwise contributions:

```text
dv/v_model(t) ~= sum_Bi [K_gwl(Bi) * X_gwl(Bi, t)
                       + K_atm(Bi) * X_atm(Bi, t)
                       + K_snow(Bi) * X_snow(Bi, t)
                       + K_temp(Bi) * X_temp(Bi, t)]
```

The default period bands are:

- `300–120 days`
- `120–60 days`
- `60–30 days`
- `30–16 days`
- `16–8 days`

The fitted output is written to:

- `dvv_model`
- `dvv_residual = dvv_obs - dvv_model`

The coefficients are fit only on the configured calibration period, then held fixed while predicting both the calibration and validation periods.

## Time-Grid Note

The physical predictors are not indexed by the raw groundwater sampling times. Instead, the workflow explicitly builds a complete regular model grid from:

```text
pd.date_range(start, end, freq=RESAMPLE_RULE)
```

and aligns groundwater, atmospheric pressure, ERA5 snow depth, and well temperature to that common grid.

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

The current branch supports:

- the monthly IMO text format used in the original workflow
- the local Grindavik station CSV used in `pore_pressure_model.py`

### ERA5 snow data

Expected input is an ERA5-Land time-series CSV containing:

- `valid_time`
- `sde` for snow depth in meters
- optionally `snowc` for snow cover

The current branch will raise an error if `sde` is entirely empty, which usually means the selected ERA5-Land point is over ocean rather than land.

### Observed `dv/v`

The transfer script supports:

- a simple CSV with `datetime` and `dvv`
- a DTT folder containing daily text files with columns such as:
  - `Date`
  - `Pairs`
  - `M0`
  - `EM0`

## Quick Start

1. Edit the file paths and physical parameters in `pore_pressure_model.py`.
2. Edit the observed `dv/v` source and the calibration / validation windows in `dvv_transfer_model.py`.
3. Run the workflow:

```bash
python pore_pressure_model.py
python dvv_transfer_model.py
python evaluate_dvv_model.py
python visualize_dvv_results.py
```

4. Check the main outputs in `../output-2`:

- `pore_pressure_output.csv`
- `dvv_transfer_output.csv`
- `dvv_transfer_coefficients.csv`
- `dvv_model_metrics.csv`
- `fig_dvv_observed_vs_modeled.png`
- `fig_dvv_residual.png`

## Practical Usage

### Example 1: Use all station pairs from a DTT folder

In `dvv_transfer_model.py`:

```python
OBSERVED_DVV_INPUT_MODE = "DTT_FOLDER"
DVV_PAIR_MODE = "ALL"
OBSERVED_DVV_DTT_FOLDER = "/path/to/ZZ"
DVV_VALUE_COLUMN = "M0"
DVV_SCALE = -100.0
```

This reads `M0` as `dt/t` and converts it to `dv/v (%)`.

### Example 2: Use a single station pair

In `dvv_transfer_model.py`:

```python
DVV_PAIR_MODE = "PAIR"
DTT_STATION1 = "R7E04"
DTT_STATION2 = "R94DB"
```

The script matches station codes inside DTT pair names, even if the file stores full names such as `5S_R7E04_5S_R94DB`.

### Example 3: Change the calibration and validation windows

In `dvv_transfer_model.py`:

```python
CALIBRATION_START_TIME = "2021-03-01"
CALIBRATION_END_TIME = "2021-06-10"
VALIDATION_START_TIME = "2021-06-11"
VALIDATION_END_TIME = "2021-12-31"
```

Only the calibration period is used to estimate the transfer coefficients. The validation period is predicted using the same fitted coefficients.

### Example 4: Change the final time resolution

In `pore_pressure_model.py`:

```python
RESAMPLE_RULE = "1D"
```

This defines the common model grid. All physical predictors are aligned to this regular grid before the transfer step.

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
../output-2
```

Each script creates the directory automatically if it does not already exist.

## Configuration Notes

### In `pore_pressure_model.py`

Set:

- `GWL_CSV_PATH`
- `ATM_TXT_PATH`
- `ERA5_SNOW_CSV_PATH`
- `START_TIME`
- `END_TIME`
- `RESAMPLE_RULE`
- `DEPTHS_M`
- `INCLUDE_SNOW_LOADING`
- `RHO_W`
- `GRAVITY`
- `SNOW_DENSITY_KG_M3`
- `SKEMPTON_B`
- `UNDRAINED_POISSON_RATIO`
- `HYDRAULIC_DIFFUSIVITY_M2_S`
- `POISSON_RATIO`
- `YOUNGS_MODULUS_PA`
- `M_OVER_MU_RATIO`
- other thermoelastic constants if needed
- `OUTPUT_CSV_PATH`

### In `dvv_transfer_model.py`

Set:

- `PORE_PRESSURE_CSV_PATH`
- `OBSERVED_DVV_INPUT_MODE`
- `OBSERVED_DVV_DTT_FOLDER` or `OBSERVED_DVV_CSV_PATH`
- `DVV_PAIR_MODE`
- `DTT_STATION1`
- `DTT_STATION2`
- `DVV_VALUE_COLUMN`
- `DVV_SCALE`
- `MODEL_DEPTH_M`
- `INCLUDE_THERMOELASTIC`
- `INCLUDE_SNOW_LOADING`
- `FIT_INTERCEPT`
- `CALIBRATION_START_TIME`
- `CALIBRATION_END_TIME`
- `VALIDATION_START_TIME`
- `VALIDATION_END_TIME`

### In `evaluate_dvv_model.py`

Set:

- `DVV_TRANSFER_OUTPUT_CSV`
- `DVV_TRANSFER_COEFFICIENTS_CSV`
- `OUTPUT_METRICS_CSV`
- `OUTPUT_CV_CSV`
- `N_CV_SPLITS`

## Output Files

### From `pore_pressure_model.py`

- `../output-2/pore_pressure_output.csv`

### From `dvv_transfer_model.py`

- `../output-2/dvv_transfer_output.csv`
- `../output-2/dvv_transfer_coefficients.csv`

### From `evaluate_dvv_model.py`

- `../output-2/dvv_model_metrics.csv`
- `../output-2/dvv_model_cv.csv`

### From `visualize_dvv_results.py`

- `../output-2/fig_dvv_observed_vs_modeled.png`
- `../output-2/fig_dvv_residual.png`
- `../output-2/fig_dvv_predictors.png`

## Current Assumptions and Limits

- The hydraulic branch currently uses the same diffusion-style kernel for groundwater, atmospheric, and snow loading.
- The thermoelastic branch currently uses a seasonal harmonic formulation rather than a fully transient thermal diffusion inversion.
- The transfer from physical predictors to `dv/v` is currently linear.
- The observed `dv/v` aggregation in `ALL` mode currently uses the median of all non-autocorrelation station pairs per day.
- The current snow-loading implementation uses snow depth with an assumed bulk snow density, not a true SWE product.

## References

- Luo, B., Zhu, H., & Lumley, D. (2025). Seismic Monitoring of Baseflow and Groundwater Changes in the Yellowstone National Park. Geophysical Research Letters.
- Rivet, D., Brenguier, F., & Cappa, F. (2015). Improved detection of preeruptive seismic velocity drops at the Piton de La Fournaise volcano. Geophysical Research Letters, 42, 6332-6339. https://doi.org/10.1002/2015GL064835
- Tsai, V. C. (2011). A model for seasonal changes in GPS positions and seismic wave speeds due to thermoelastic and hydrologic variations.
