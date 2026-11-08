# Data Dictionary — Pi Sensor Log

**Prototype:** rice waterlogging drainage, Jianghuai / middle-lower Yangtze polder farmland  
**Timezone:** Asia/Shanghai (UTC+8)  
**Hardware fiction:** Raspberry Pi + capacitive rainfall + capacitive soil moisture, 10-min logging  
**Missing values:** empty cell or pandas NA — never mix numeric sentinels (−999, 9999, etc.)

---

## Long-format columns (`pi_sensor_log.csv`)

| Column | Chinese alias | Meaning | Unit | dtype | Valid range | Missing | Source / notes |
|--------|---------------|---------|------|-------|-------------|---------|----------------|
| `timestamp_local` | 本地时间 | Local wall-clock time at the farm | ISO-8601 with offset `+08:00` | string/datetime | campaign window | empty | Pi RTC or NTP |
| `timestamp_utc` | UTC时间 | Same instant in UTC | ISO-8601 `Z` | string/datetime | campaign window | empty | Derived from local |
| `farm_id` | 农场编号 | Farm identifier | code | string | `JH-*` | empty | Meta table |
| `plot_id` | 地块编号 | Plot / field identifier | code | string | `P-*` | empty | One plot in this prototype |
| `device_id` | 设备编号 | Raspberry Pi / logger ID | code | string | `RPi-*` | empty | One device |
| `sensor_id` | 传感器编号 | Individual sensor ID | code | string | `S-*` | empty | See stations.csv |
| `sensor_type` | 传感器类型 | Sensor modality | enum | string | `rain_cap` \| `soil_vwc_cap` | empty | Capacitive rain tip-count emulator or capacitive VWC |
| `raw_adc` | 原始ADC | Raw ADC reading from Pi GPIO / ADS1115 | counts | int | 0–4095 (12-bit) or empty | empty | Missing when quality_flag=`missing` |
| `value_si` | SI读数 | Calibrated physical value | see `unit` | float | type-dependent | empty | See calibration below |
| `unit` | 单位 | SI unit for `value_si` | string | string | `mm/10min` \| `m3/m3` | empty | Rain intensity over the 10-min slot; VWC volumetric |
| `cal_version` | 标定版本 | Calibration curve ID | string | string | `cal_v1` | empty | Documented below |
| `quality_flag` | 质量标志 | QC flag at ingest time (or raw logger guess) | enum | string | `ok` \| `suspect` \| `missing` \| `saturated` \| `stuck` | — | Raw may pre-label; ingest re-checks |

### Sensor types

| `sensor_type` | Chinese | Physical quantity | Typical SI range |
|---------------|---------|-------------------|------------------|
| `rain_cap` | 电容雨量 | Rain depth in the 10-minute interval | 0–25 mm / 10 min (extreme typhoon burst) |
| `soil_vwc_cap` | 电容土壤体积含水率 | Volumetric water content | 0.05–0.55 m³/m³ (clay-loam rice paddy) |

### Fake ADC → SI calibration (`cal_v1`)

**Rain (`rain_cap`):**  
Capacitive rain board (or tipping-bucket pulse counted into ADC proxy).  
\[
\text{value\_si (mm/10min)} = \max\bigl(0,\; (\text{raw\_adc} - 200) / 150\bigr)
\]  
- ADC 200 ≈ 0 mm (dry baseline)  
- ADC 1700 ≈ 10 mm / 10 min  
- ADC saturates near 4095 ≈ ~26 mm / 10 min (flag `saturated`)

**Soil VWC (`soil_vwc_cap`):**  
Capacitive probe at ~15 cm root-zone depth.  
\[
\text{value\_si (m³/m³)} = \text{clip}\bigl(0.05 + (\text{raw\_adc} - 800) / 6000,\; 0.05,\; 0.55\bigr)
\]  
- ADC 800 ≈ 0.05 (air-dry)  
- ADC 2000 ≈ 0.25 (near field capacity for this clay-loam)  
- ADC 3800 ≈ 0.55 (saturation; flag `saturated` if ADC ≥ 3900)

### Quality flags

| Flag | Meaning |
|------|---------|
| `ok` | Passes range, stuck, and spike checks |
| `suspect` | Spike or out-of-family jump; kept but down-weighted |
| `missing` | No reading; `raw_adc` and `value_si` empty |
| `saturated` | ADC at rail / VWC at physical max |
| `stuck` | Identical non-missing value for ≥ 6 consecutive slots (≥ 1 h) |

---

## Stations meta (`data/meta/stations.csv`)

| Column | Meaning |
|--------|---------|
| `farm_id` | Farm code |
| `plot_id` | Plot code |
| `device_id` | Logger / Pi |
| `sensor_id` | Sensor |
| `sensor_type` | `rain_cap` or `soil_vwc_cap` |
| `depth_cm` | Install depth (0 = above canopy for rain) |
| `crop` | Crop (rice) |
| `soil_class` | USDA-ish class (clay_loam) |

---

## Processed features (`data/processed/features.csv`)

Wide, 10-min index after QC. Built by `scripts/ingest_qc.py`.

| Column | Meaning | Unit |
|--------|---------|------|
| `timestamp_local` | Local time | ISO +08:00 |
| `rain_mm_h` | Instantaneous rain rate from last 10-min depth × 6 | mm/h |
| `rain_1h_mm` | Rolling 1-hour rain depth | mm |
| `rain_6h_mm` | Rolling 6-hour rain depth | mm |
| `vwc_m3m3` | QC'd soil VWC | m³/m³ |
| `hours_above_field_capacity` | Cumulative hours with VWC ≥ FC (0.32) | h |
| `quality_summary` | Per-row QC note (`ok`, `gap_filled`, `spike_flagged`, …) | string |

---

## Workflow log (`outputs/pump_workflow_*.csv`)

| Column | Meaning |
|--------|---------|
| `timestamp` | Local time |
| `predicted_severity` | `none` \| `watch` \| `waterlogging` \| `severe` |
| `action` | `PUMP_ON` \| `PUMP_OFF` |
| `reason` | Short human-readable rule hit |
