# Rice Waterlogging Drainage — Laptop Demo

Year-2 CS team prototype for **food + water resilience**: reduce rice crop loss from extreme rainfall / typhoon waterlogging on polder farmland in eastern China (Jianghuai / middle-lower Yangtze).

**You implement software.** Teammates own Raspberry Pi sensors and physical drainage. This repo is a **judge-runnable laptop demo** — one command, Excel-openable CSVs, charts that show automated drainage beating do-nothing.

## Scope defaults

| Item | Default |
|------|---------|
| Place | Jianghuai polder farmland, Asia/Shanghai (UTC+8) |
| Crop | Rice, one plot (~10 ha) |
| Hazard | Typhoon **rainfall hyetograph** (not wind/surge) |
| Sensors | Pi + capacitive rain + capacitive soil VWC, 10-min |
| Drainage | 1 soil bucket + 1 ditch + 1 on/off pump |
| Pump capacity | **25 mm/h** equivalent (~2500 m³/h over 10 ha) |
| Field capacity | **FC = 0.32** m³/m³ (clay_loam) |
| Rule ON / OFF | pond 40→12 mm hysteresis (ponding-primary farmer rule) |

See `PROVENANCE.md` for equations and the crop-loss **PROXY** (not RMB yield).

## Quick start

```bash
cd rice-waterlogging-drainage
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py
# or: make demo
```

This will:
1. Generate `data/raw/pi_sensor_log.csv` (60 h storm)
2. QC → `data/processed/sensor_clean.csv` + `features.csv`
3. Train a small sklearn severity model (time split + dummy baseline)
4. Simulate **do-nothing / rule-based / automated** pump policies
5. Write CSVs + PNG figures under `outputs/`

Optional UI:

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
src/hydrology.py, drainage.py, predict.py
app/dashboard.py
run_demo.py
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

## Libraries

pandas, numpy, matplotlib, scikit-learn, streamlit.  
**No** PyTorch/TF, DRL, pyswmm, or hydrology desktop apps.
