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
- **Drainage:** ONE soil bucket + ONE ditch/sump + ONE on/off pump. No GPIO in this repo.

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

## Data
- Synthetic but physically shaped storm (dry → typhoon burst → plateau → recession), 60 h × 10 min.
- QC edge cases injected: stuck VWC, rain spike, 30-min missing gap.
- sklearn model predicts **near-future (≈4 h)** severity from recent features; time-ordered split; dummy baseline = always `watch`.

## Non-goals / limitations
- Not EPA SWMM / pyswmm; not city sewers; not DRL; not live Pi GPIO.
- Rainfall-only typhoon; single field; proxy damage; toy calibration curves (`cal_v1`).
