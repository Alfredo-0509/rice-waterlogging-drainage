# Rice Waterlogging Drainage

Prototype for **food + water resilience**: reduce rice crop loss from extreme rainfall / typhoon waterlogging on polder farmland in eastern China (Jianghuai / middle-lower Yangtze).

**Implemented software.** Teammates own Raspberry Pi sensors and physical drainage. This repo is a **judge-runnable laptop demo** — one command, Excel-openable CSVs, charts that show automated drainage beating do-nothing.

Three ways to drive it: `run_demo.py` (CLI), `app/gui.py` (Tkinter control room — recommended for exploring), and `app/dashboard.py` (optional Streamlit view). All three call the same step functions in `src/pipeline.py`, so the CSVs are identical whichever you use.

# Scripts
## `generate _raw_csv`

expected outcome
1. receive data from sensor (soil moisture, rainfall sensor)
2. convert data into visible data sheet 

<center>achievement</center>

**simulate a storm over 60 hours**
- 2 main rain bands ---> moderate raining & storm burst

**check and calculate soil moisture & rainfall**
- VWC (data for soil moisture) 
- 1st case: rise
- 2nd case: drain

**randomization (injecting messy real world data)**
errror
- sensor readings (fronzen soil)
- a sudden spike in rainfall data (false positive due to environmental error)
- missing data gaps (transmission loss)

**Data automation** 
- convert all the make-up data into a useable .csv file

### Unseeded randomness + configurable noise

There is **no fixed RNG seed**: every run draws fresh OS entropy, so the storm, the injected stuck sensor, the rain spike and the missing-data gap all move a little. Pass `--seed 42` only when you need a reproducible log; blank/absent means random.

Rainfall variability and sensor noise are **separate, independently configurable** σ values (mm/10 min unless stated):

| Flag | Python argument | Default | What it jitters |
|------|-----------------|---------|-----------------|
| `--rain-noise` | `rain_noise` | 0.05 | hyetograph jitter in raining slots |
| `--rain-noise-dry` | `rain_noise_dry` | 0.01 | hyetograph jitter in dry slots |
| `--rain-sensor-noise` | `rain_sensor_noise` | 0.02 | rain-gauge measurement noise |
| `--vwc-sensor-noise` | `vwc_sensor_noise` | 0.002 m³/m³ | soil-VWC measurement noise |

```bash
python3 scripts/generate_raw_csv.py --rain-noise 0.2 --vwc-sensor-noise 0.01
python3 scripts/generate_raw_csv.py --seed 7                 # repeatable storm
```

`generate(out, *, seed=None, rain_noise=..., rain_noise_dry=..., rain_sensor_noise=..., vwc_sensor_noise=...)` is the importable API; `run_demo.py` exposes the same flags and the GUI has five input boxes for them.

## `ingest_qc.py`

expected outcome:
- quality control, feature engineering 
- clean data from messy noise, create useful data input 

<center>achievement</center>

**account for any dropped data (`drop.na()`)**
- give them a tag (`ok`, `suspect`, `stuck`, `saturated`, `missing`)
- explaining the reliability of data from sensor

**basic QC rule**
- flags values that fall outside plausible boundaries (-ve soil moisture etc.)
- catches sudden, unrealistic spike in data 
- flags sensors that stay completely frozen for 6 or more time slots (10mins)
	while cleverly ignoring natural flatlines when the soil is safely saturated.

**build Features (`features.csv`)**
- Creates a wide-format summary table calculating useful metrics like:
- Hourly and 6-hourly cumulative rainfall.
- Soil Volumetric Water Content (VWC).
- Total hours the soil has remained above field capacity.

# src
`drainage.py`
- `def PumpController` (anti-chatter, noise cleaner)
- Enforces a minimum on-time of 30 minutes (3 slots) once the pump turns on. This prevents rapid, harmful on-and-off switching ("chattering") of hardware relays.
- **`policy_do_nothing`:** 
- the pump is never turned on (`False`), measure worst case crop damage.
- `make_rule_policy` (Traditional Farmer Rule):
- Relies strictly on physical standing water depth (`pond_mm`) without looking at soil moisture or forecasts.
- Turns ON only when ponding reaches a high threshold (`≥ 40 mm`) and stays on until it drops below `12 mm`.
- `make_automated_policy` (Prediction-Aware Control):
 - Uses the machine learning model's **predicted severity score** to act proactively:
- Severe/Waterlogging risk anticipated: activate a more aggressive threshold, the pump will be activated at a lower VWC.  
- Calm conditions:  conservative baseline thresholds.
- **`build_workflow_log`:**
- Generates a step-by-step (`pump_workflow_automated.csv`) capturing every decision, state change, and the specific predictive reasoning behind why the pump turned ON or OFF.

`hydrology.py`
- models how water moves through the soil and surface ditches
- estimates the resulting crop stress and damage during a storm (AI generated loss analysis)

- **Soil & Water Balance Bucket Model:**
- Calculates how rainwater infiltrates the root zone (up to saturation limit $\theta_s$) 
- how excess water diffuses deeper down when soil is wetter than field capacity ($FC$).

- **Surface Ponding & Ditch Tracking:**
- Any rainfall that exceeds the soil's infiltration capacity turns into surface runoff/ponding ($h$)
- accumulates in the field's drainage ditch or is removed when the pump is active.

- **Severity Classification:**    
- assess the current field risk at every time step
- categorizing it as `none`, `watch`, `waterlogging`, or `severe` 
- based on recent rainfall, water depth, and soil moisture.

The final boolean output (`True` or `False`) is returned to the simulation engine (`WaterBalanceSim` from `hydrology.py`)

### Selectable simulation models

`src/hydrology.py` now exposes `HYDRO_MODELS`, `DEFAULT_MODEL`, `list_models()` and `get_model()`. Pass a model to `WaterBalanceSim(model=...)` or `run_simulation(..., model=...)`; the GUI dropdown, `--model` and `--all-models` are all built from that registry.

| Key | Soil profile | Behaviour |
|-----|--------------|-----------|
| `bucket` (default) | one 300 mm root-zone bucket | ponds as soon as the root zone saturates (original demo) |
| `two_layer` | root zone + 160 mm permeable subsoil | excess recharges the subsoil before ponding; slow drainage afterwards |
| `plough_pan` | compacted pan, infiltration capped at 6 mm/10 min | strongly reduced deep percolation → fastest, deepest ponding (worst case) |

```bash
python3 run_demo.py --model two_layer          # one model
python3 run_demo.py --all-models               # all three, side by side
```

Outputs are per model: `sim_<policy>.csv` / `pump_workflow_<policy>.csv` for the default `bucket`, and `sim_<policy>__<model>.csv` / `pump_workflow_<policy>__<model>.csv` for the others. `outputs/damage_summary.csv` carries `model`, `model_label`, `policy` and `policy_label` columns so every combination can be compared.

`predict.py`
https://uc-r.github.io/gbm_regression
![[Pasted image 20261031174621.png|371]]

![[Pasted image 20261031175231.png|375]]

Demo 
1. **Trains a Predictive Model:** Uses `scikit-learn` to train a waterlogging severity predictor on the processed features and saves performance metrics.
    
2. **Simulates Drainage Policies:** Runs three distinct drainage strategies to compare their effectiveness against the typhoon scenario:
    
    - **Do Nothing** (no intervention).
        
    - **Rule-Based** (traditional threshold logic).
        
    - **Automated** (advanced data-driven control).
        
3. **Evaluates & Summarizes Damage:** Compares crop/field stress across the policies and exports a damage summary spreadsheet (`damage_summary.csv`).
    
4. **Generates Visualizations & Artifacts:** Saves plots and JSON summaries to an `outputs/` folder and prints instructions on how to launch the interactive Streamlit dashboard.


Data Visualization
- why incorporating GBR prediction is better, than ruled based prediction
![[fig_damage_compare.png|700]]

- Loss analysis - how much crops are saved by using our model
- how much value it could create for the farmers (if estimating it with money)
![[fig_field_damage.png]]

- Ponding depth and the workflow of the water drainage pump during the simulation
![[fig_ponding_pump.png]]

- raw data input received from the sensors (messy real world data)
![[fig_rain_vwc.png]]

- predicted waterlogged severity level by GBR model
- to help make decision of water drainage pump action
![[fig_severity.png]]


| Model part                                          | What we used                                          | Closest supporting literature (ideas, not our exact params)                                                                                                                                                                                       |
| --------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Region / rice waterlogging (Yangtze / Hubei plains) | Jianghuai setting                                     | [Morphology… waterlogging stresses (Hubei hybrid rice)](https://zgnyqx.ieda.org.cn/EN/10.3969/j.issn.1000-6362.2016.02.008); [RESPONSE OF HYBRID RICE TO FLOODING… drainage index](https://en.cnki.com.cn/Article_en/CJFDTOTAL-CJLY201406019.htm) |
| Yield loss vs ponding depth × duration              | PROXY from hours ponded + hours above FC + peak depth | [Yield loss & flooding, hybrid rice (Sichuan)](https://zwxb.chinacrops.org/EN/10.3724/SP.J.1006.2016.01381); [Paddy inundation depth/duration (Huaihe)](https://www.iahr.org/library/info?pid=14182)                                              |
| Drainage / water-level control in paddies           | Pump policies                                         | [Controlled irrigation & drainage in paddies](https://pmc.ncbi.nlm.nih.gov/articles/PMC8036912/); [Optimal drainage operation for waterlogging loss](https://doi.org/10.3390/w13202811)                                                           |
| High-start / low-stop pump hysteresis (40→12 mm)    | Engineering practice, not a paper                     | Common “high start / low stop” pump logic (e.g. Chinese drainage-pump control writeups); our 40/12 mm are **demo thresholds**, not a cited drainage standard                                                                                      |


## Scope defaults

| Item | Default |
|------|---------|
| Place | Jianghuai polder farmland, Asia/Shanghai (UTC+8) |
| Crop | Rice, one plot (~10 ha) |
| Hazard | Typhoon **rainfall hyetograph** (not wind/surge) |
| Sensors | Pi + capacitive rain + capacitive soil VWC, 10-min |
| Drainage | 1 ditch + 1 on/off pump; selectable soil profile (bucket / two_layer / plough_pan) |
| RNG seed | **none** by default; `--seed` / GUI field for reproducibility only |
| Rainfall noise | σ = 0.05 mm/10 min (raining), 0.01 (dry) — configurable |
| Sensor noise | rain σ = 0.02 mm/10 min, VWC σ = 0.002 m³/m³ — independently configurable |
| Pump capacity | **25 mm/h** equivalent (~2500 m³/h over 10 ha) |
| Field capacity | **FC = 0.32** m³/m³ (clay_loam) |
| Rule ON / OFF | pond 40→12 mm hysteresis (ponding-primary farmer rule) |

See `PROVENANCE.md` for equations and the crop-loss **PROXY** (not RMB yield).

## Quick start

```bash
cd rice-waterlogging-drainage-main
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 run_demo.py                    # one model (bucket), all three policies
python3 run_demo.py --all-models       # all three hydrology models
python3 app/gui.py                     # Tkinter control room (recommended)
python3 app/gui.py --check             # headless sanity check, no window
# or: make demo / make all-models / make gui
```

On Windows PowerShell, run the commands from the project directory (the
directory containing `app`, `src`, and `run_demo.py`):

```powershell
Set-Location .\rice-waterlogging-drainage-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app\gui.py
```

If you are already in `rice-waterlogging-drainage-main`, only
`python app\gui.py` is needed. Running `python3 app/gui.py` from its parent
directory will fail because `app` is inside the project directory.

Useful CLI flags: `--model {bucket,two_layer,plough_pan}` (repeatable), `--all-models`, `--policy {do_nothing,rule_based,automated}` (repeatable), `--seed`, `--rain-noise`, `--rain-noise-dry`, `--rain-sensor-noise`, `--vwc-sensor-noise`, `--no-figures`. Run `python3 run_demo.py --help` for defaults.

This will:
1. Generate `data/raw/pi_sensor_log.csv` (60 h storm, fresh randomness)
2. QC → `data/processed/sensor_clean.csv` + `features.csv`
3. Train a small sklearn severity model (time split + dummy baseline)
4. Simulate **do-nothing / rule-based / automated** × each selected hydrology model
5. Write CSVs + PNG figures under `outputs/` (6 figures when `--all-models`)

## Tkinter GUI — control room

```bash
python3 app/gui.py
```

The left panel has **five step buttons** (① generate raw CSV with the noise/seed boxes, ② QC + features, ③ train & predict severity, ④ run simulations with the model dropdown + policy checkboxes, ⑤ summary & figures) plus **Run full pipeline**. Each step runs on a worker thread (the window stays responsive), logs its progress at the bottom and refreshes the result views when it finishes. The notebook keeps just four tabs: Overview, CSV inspector, Files, Figures.

Results are **categorized and sorted**:

- **Overview** — damage summary sorted by crop-loss PROXY (click any column heading to re-sort), a plain-language best-policy insight, model metrics (accuracy, MAE, dummy baseline, holdout, train/test sizes) and category counts by dimension, sorted by count. It aggregates every simulation CSV on disk; the CSV inspector is deliberately scoped to the latest run.
- **CSV inspector** — a **dropdown**, not one tab per file. Every CSV is classified along three categories that double as filters and as the grouping/sort order: **type of data** (predictions, model metrics, damage summary, simulation, pump workflow, …), **model** (`bucket` / `two_layer` / `plough_pan`) and **treatment** = drainage policy (`do_nothing` / `rule_based` / `automated`). Choose "Group / sort by" to order the list by any of the three; each dropdown entry shows all three categories plus the filename. The selected CSV is inspected on its own: search across all columns, filter on a categorical value, sort by clicking a heading, category counts (choose the column), numeric min/max/mean/median/std of the filtered rows, and copy/open its path.
- **Only simulated CSVs** — by default the inspector lists just the files produced by the **latest simulation run** (`outputs/simulation_manifest.json`), plus the three shared result CSVs (`features_with_prediction.csv`, `predict_metrics.csv`, `damage_summary.csv`). Raw/clean/features inputs and stale simulation CSVs left behind by earlier runs are hidden, so a single-model run shows 9 CSVs instead of 24. Uncheck "Only simulated runs" to list every CSV on disk; the **Files** tab always inventories every artefact (with kind, size, modification time and double-click-to-open).
- **Figures** — PNG selector and inline preview.

Tkinter ships with CPython; on Debian/Ubuntu install `python3-tk` if `import tkinter` fails. `python3 app/gui.py --check` verifies imports, prints the three CSV categories and their grouping previews, and exercises the categorized-stats / metrics / file-inventory / CSV-reading helpers without opening a window (it exits 1 on a machine without Tk, after printing the install hint).

Optional Streamlit view:

```bash
streamlit run app/dashboard.py
```

## Outcomes (what judges should see)

1. **Sensor → CSV** — open `data/raw/pi_sensor_log.csv` in Excel; drop future Pi logs into `data/templates/pi_sensor_log.empty.csv`.
2. **Severity prediction** — `outputs/features_with_prediction.csv` + `predict_metrics.csv`.
3. **Drainage workflow** — `outputs/pump_workflow_automated.csv` (`timestamp, predicted_severity, action, reason`).
4. **Damage comparison** — `outputs/damage_summary.csv` + figures:
   - `fig_rain_vwc.png` — storm rain + VWC
   - `fig_ponding_pump.png` — less ponding with automation
   - `fig_severity.png` — predicted severity
   - `fig_damage_compare.png` — ponded hours / peak depth / crop-loss PROXY
   - `fig_model_compare.png` — crop-loss PROXY by model × policy (when more than one model ran)
   - `fig_field_damage.png` — schematic field fill (not a China map)

## Layout

```
data/
  raw/pi_sensor_log.csv
  templates/pi_sensor_log.empty.csv
  schema/data_dictionary.md
  meta/stations.csv
  processed/sensor_clean.csv, features.csv
scripts/generate_raw_csv.py, ingest_qc.py
src/pipeline.py                 # shared step functions + config + GUI helpers
src/hydrology.py                # selectable soil models
src/drainage.py, predict.py
app/gui.py                      # Tkinter control room (dropdown CSV inspector)
app/dashboard.py                # figures + optional Streamlit view
run_demo.py                     # CLI
outputs/                        # CSVs, PNGs, demo_result.json,
                                # simulation_manifest.json (latest run)
PROVENANCE.md
```

## Policies compared

1. **Do nothing** — pump always off.
2. **Rule-based** — pump if ponding or VWC above threshold; stop below lower threshold (hysteresis + 30 min min-on).
3. **Automated** — same idea, but uses **predicted severity** to turn on earlier / stay on later (`src/drainage.py::make_automated_policy`).

## Assumptions & limitations

- Toy single-field water balance — **not** EPA SWMM / city sewers.
- Typhoon = rainfall only; no wind, surge, or salinity.
- Crop-loss is a **PROXY index [0,1]**, not certified yield or RMB.
- Fake ADC→SI calibration (`cal_v1`) for demo sensors.
- No live GPIO; pump actions are simulated workflow logs.
- sklearn model is small; if classes are thin we still ship physics severity.
- Storm/sensor noise is unseeded, so numbers differ slightly between runs; use `--seed` for identical figures.
- The extra soil models are parameter variants of the same bucket, not calibrated field models.

## Libraries

pandas, numpy, matplotlib, scikit-learn, `tkinter` (standard library), streamlit (optional, dashboard only).  
**No** PyTorch/TF, DRL, pyswmm, or hydrology desktop apps.
