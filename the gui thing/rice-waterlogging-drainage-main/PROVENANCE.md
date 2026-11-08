# PROVENANCE — Rice waterlogging drainage prototype

## Region & crop
- **Place:** Polder-style farmland, Jianghuai / middle-lower Yangtze (eastern China).
- **Timezone:** Asia/Shanghai (UTC+8).
- **Crop:** Rice (single plot `P-RICE-A`).
- **Hazard:** Waterlogging from heavy rain / typhoon **rainfall hyetograph only**.  
  No cyclone track, wind, storm surge, or salinity modules.

## Scale & hardware fiction
- **Area:** ~10 ha equivalent (one field).
- **Sensors:** Raspberry Pi + capacitive rainfall (`rain_cap`) + capacitive soil VWC (`soil_vwc_cap`), 10-min logging.
- **Drainage:** ONE ditch/sump + ONE on/off pump, over a **selectable** soil profile (see below). No GPIO in this repo.

## Soil / bucket parameters
| Symbol | Value | Meaning |
|--------|-------|---------|
| Zr | 300 mm | Root-zone depth |
| FC | 0.32 m³/m³ | Field capacity (clay_loam) |
| θs | 0.52 m³/m³ | Saturation |
| θr | 0.15 m³/m³ | Residual |
| K_perc | 0.08 /h | Deep percolation rate factor when θ > FC |
| Δt | 10 min | Simulation / logging step |

### Bucket equations (see `src/hydrology.py` comments)
1. Infiltration fills θ up to θs; remainder is surface excess.
2. Excess accumulates as ponding / ditch stage `h` (mm).
3. Pump removes up to `Qp × Δt` while ON; `h` never negative.
4. Slow percolation drains θ toward FC when wet.

## Selectable soil-profile models

`src/hydrology.py::HYDRO_MODELS` is the single source of truth. All three models share the pump, weir, severity classifier, crop-loss PROXY and time step; only infiltration / storage / percolation differ. Choose with `WaterBalanceSim(model=...)`, `run_simulation(..., model=...)`, `--model` / `--all-models` in `run_demo.py`, or the GUI dropdown.

| Key | Extra parameters | Physical change |
|-----|------------------|-----------------|
| `bucket` (default) | none (Zr 300 mm, K_perc 0.08/h) | one root-zone bucket; rain above saturation ponds immediately (original demo) |
| `two_layer` | `subsoil_mm = 160`, `subsoil_k = 0.02/h` | surface excess first recharges a subsoil store; only overflow ponds. Subsoil then drains to depth at `subsoil_k × store × Δt` |
| `plough_pan` | `infil_cap_mm = 6.0` per 10-min slot, `k_perc = 0.01/h` | compacted pan caps infiltration and almost stops deep percolation → fastest/deepest ponding (stress case) |

All models keep FC = 0.32, θs = 0.52, θr = 0.15, Qp = 25 mm/h and weir stage 200 mm. These are parameter variants of the toy bucket, **not** calibrated field models.

## Output naming (model-aware)

- `outputs/sim_<policy>.csv`, `outputs/pump_workflow_<policy>.csv` — default `bucket` (legacy names preserved).
- `outputs/sim_<policy>__<model>.csv`, `outputs/pump_workflow_<policy>__<model>.csv` — every other model.
- `outputs/damage_summary.csv` — one row per (model, policy), with `model`, `model_label`, `policy`, `policy_label` plus the PROXY columns.
- `outputs/fig_model_compare.png` — written only when more than one model was simulated.
- `outputs/simulation_manifest.json` — written by every simulation run; lists the models, treatments and the exact simulation / workflow CSVs produced, so the GUI can hide stale CSVs from earlier runs.

## Pump capacity
- **Qp = 25 mm/h** equivalent over the plot  
  ≈ 25×10⁻³ m/h × 100_000 m² = **2_500 m³/h** for 10 ha.  
  Chosen as a mid-size agricultural axial-flow pump class for a polder ditch — documented fiction for the prototype.

## Rule thresholds
| Policy | ON | OFF (hysteresis) |
|--------|----|------------------|
| Rule-based | pond ≥ 40 mm (ponding-primary) | pond ≤ 12 mm |
| Automated | lowered ON thresholds when predicted severity ≥ watch; holds ON longer while prediction elevated | see `src/drainage.py` |
| Min on-time | 30 min (3 slots) | all pump policies |

## Crop-loss PROXY (NOT RMB yield)
```
proxy = 0.45*f(hours_ponded) + 0.35*f(hours_VWC>FC) + 0.20*f(peak_depth)
```
Piecewise knots in `src/hydrology.py::damage_proxy`.  
This is a **demo index in [0,1]**, not a certified agronomic or insurance loss.

## Severity labels
Derived from ponding depth, recent ponded hours, 6-h rain, and VWC:  
`none | watch | waterlogging | severe` plus score 0–1.

## Data, randomness & configurable noise
- Synthetic but physically shaped storm (dry → typhoon burst → plateau → recession), 60 h × 10 min.
- **No fixed RNG seed.** `numpy.random.default_rng(None)` uses fresh OS entropy on every run; the QC edge-case positions/magnitudes (stuck VWC, rain spike, 30-min missing gap) are drawn from the same generator. `--seed` / `PipelineConfig.seed` is available strictly for reproducibility and is off by default.
- Rainfall variability and sensor noise are separate σ parameters (`scripts/generate_raw_csv.py`, also mirrored by `run_demo.py` and the GUI):

  | Parameter | Default | Applied to |
  |-----------|---------|------------|
  | `rain_noise` | 0.05 mm/10 min | hyetograph jitter while raining |
  | `rain_noise_dry` | 0.01 mm/10 min | hyetograph jitter in dry slots |
  | `rain_sensor_noise` | 0.02 mm/10 min | rain-gauge measurement noise (independent of hyetograph) |
  | `vwc_sensor_noise` | 0.002 m³/m³ | soil-VWC measurement noise |

  All four must be finite and ≥ 0; setting one to 0 removes that noise source only.
- QC edge cases injected: stuck VWC, rain spike, 30-min missing gap.
- sklearn model predicts **near-future (≈4 h)** severity from recent features; time-ordered split; dummy baseline = always `watch`.

## Non-goals / limitations
- Not EPA SWMM / pyswmm; not city sewers; not DRL; not live Pi GPIO.
- Rainfall-only typhoon; single field; proxy damage; toy calibration curves (`cal_v1`).

## Interfaces / execution
- `run_demo.py` (CLI) → `src/pipeline.py::run_pipeline`.
- `app/gui.py` (Tkinter) calls the same step functions one at a time on a worker thread: `generate_raw`, `run_qc`, `run_prediction`, `run_simulations`, `build_summary` + `build_figures`. Its categorized/sorted tables come from `build_category_stats()` (counts per categorical dimension, sorted by count) and `build_metrics_table()`, and every CSV from `expected_csv_paths()` gets its own inspectable tab.
- `app/dashboard.py::save_figures` builds the PNGs; `main_streamlit()` only displays them.
- `python3 app/gui.py --check` lists models and CSV tabs without opening a window (useful on headless machines).

## GUI CSV inspector (categories & scope)
- CSVs are classified by **type of data** (raw / clean / features / predictions / model metrics / damage summary / simulation / pump workflow), **model** (`bucket`, `two_layer`, `plough_pan`) and **treatment** = drainage policy (`do_nothing`, `rule_based`, `automated`). Shared CSVs have no single model or treatment and are labelled `Shared`. The three dimensions are both filters and the group/sort order of the dropdown.
- Scope: by default only CSVs from the latest simulated run (manifest) plus the three shared result CSVs are listed. Input CSVs and files from older runs remain on disk and are visible in the Files tab or when "Only simulated runs" is unchecked in the GUI, but they are not loaded into the inspector by default.
- A single dropdown replaces one notebook tab per CSV; only the selected CSV is read into a DataFrame at a time, which keeps the window responsive and avoids loading every result table at once.
