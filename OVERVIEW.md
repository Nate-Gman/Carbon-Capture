# Project Overview -- Carbon Capture DAC Giga-Plant Simulation

## Summary

A real-time 3D simulation of a **multi-story Direct Air Capture (DAC) giga-plant** that removes **42.5 million tonnes** of CO2 from the atmosphere per year. The plant uses 4,000 large fan-array contactors arranged in 50 stacked stories to pull air through next-gen MOF sorbent beds, then regenerates the sorbent with heat and vacuum to release concentrated CO2 for underground storage. The entire facility is powered by renewable energy -- solar PV, solar thermal troughs, wind turbines, and geothermal wells -- with battery and molten-salt storage for 24/7 operation. The 50x scale design reduces the number of plants needed for global net-zero from ~47,000 to just **~94**.

## What Makes This Project Unique

- **Single-file Python application** (~5000 lines) with a custom software 3D renderer (no OpenGL/GPU required)
- **Dimensionally accurate** -- every component is sized in real-world metres based on published DAC designs
- **Blueprint-scale labeling** -- capture view parts include exact W x H x D dimensions for construction reference
- **Live energy simulation** -- solar irradiance, wind curves, geothermal baseload, sorbent cycling, component degradation
- **15-year campaign tracking** -- component health, maintenance scheduling, sorbent/battery replacement, cost accumulation
- **Honest cost model** -- OPEX only (~$1/t at giga-plant scale), with transparent acknowledgment that CAPEX would add $10-30/t

## Facility at a Glance

| Parameter | Value |
|-----------|-------|
| Annual capture | 42,500,000 t CO2/year (42.5 Mt) |
| Site area | 600 hectares (6.0 km2) |
| Stories | 50 (multi-story stacked design) |
| Air contactors | 4,000 (50 stories x 80 per level) |
| Fans total | 32,000 (8 per contactor, 3.5 m dia) |
| Sorbent beds | 4,000 (50 t each, 300 m3 each) |
| Regen chambers | 800 (12 m x 8 m x 10 m) |
| Solar PV | 3,520 MW peak (11,000,000 m2, 32% efficient) |
| Solar thermal | 8,580 MWth peak (11,000,000 m2, 78% efficient) |
| Wind | 2,250 MW (150 x 15 MW) |
| Geothermal | 2,250 MW thermal (200 wells) |
| Heat storage | 35,000 MWh molten salt |
| Battery | 20,000 MWh Li-ion (LFP) |
| Staff | 150 (advanced automation) |
| Simulated OPEX | ~$1/t CO2 |
| Energy per tonne | 1,250 kWh/t CO2 (thermal + electrical) |

## Four Viewing Modes

### FACILITY (Tab 1)
Bird's-eye view of the entire 600 ha giga-plant. All components are positioned and sized from the `DIMS` dictionary for dimensional accuracy. Shows the full energy infrastructure, 50-story capture field, CO2 processing chain, and support buildings.

### CAPTURE (Tab 2)
Close-up of a single air contactor unit with full engineering detail. Every part has a blueprint-style label with real dimensions. Includes:
- Contactor frame (20 m x 12 m x 4 m) with support columns, walkway, and ladder
- 8 fans with shrouds, CFRP blades, and Ti hubs
- Sorbent bed with honeycomb channel detail
- Regeneration chamber with insulation, heater rows, and connecting duct
- Vacuum pump, CO2 manifold, valve system, intake plenum
- Foundation pad for visual grounding

### OPERATION (Tab 3)
Live simulation with day/night cycle, weather, and real-time energy dispatch. Shows solar/wind/geothermal generation, energy storage levels, sorbent cycle phases, CO2 capture rate, and cost accumulation. Includes weekly operations panels with sparklines and a 15-year campaign summary.

### URBAN (Tab 4)
Cutaway skyscraper showing a DAC mini-plant installed on **one vacant office floor**. All other floors remain active office space with workers -- the building stays fully operational. Only 1 floor is needed. Includes:
- 12-floor cutaway building with columns, slabs, windows
- Office desks and chairs visible on non-DAC floors (workers present)
- Blue edge markers on DAC floor, green markers on active office floors
- 4 compact contactors (3m x 2.5m x 1.5m) in a 2x2 grid
- 3 fans per contactor (0.8m dia, quiet office-compatible operation)
- 2 regen chambers, CO2 compressor, buffer tanks
- Ducting and CO2 riser pipe to street-level collection
- Right panel: how-it-works, specs, economics, urban deployment plan
- 2,000 t CO2/year per unit, ~$80/t all-in, 100 Mt/yr from 100 cities

## Key Design Decisions

### Dimensions (based on real DAC designs)
- **Contactor:** 20 m W x 12 m H x 4 m D -- cross-flow slab geometry (Carbon Engineering / Climeworks style)
- **Sorbent bed:** 20 m x 10 m x 1.5 m -- honeycomb monolith, 300 m3, 50 t sorbent
- **Regen chamber:** 12 m x 8 m x 10 m -- vacuum vessel, SS 316L, 960 m3
- **Fans:** 3.5 m dia, 8 per contactor, CFRP blades with Ti-6Al-4V hubs

### Cost Reduction Strategy
- **50x scale giga-plant:** Multi-story design (50 levels) captures 50x more CO2 per plant
- Sorbent cost: $2,000/t (scaled MOF mass production, was $5,000/t)
- Sorbent capacity: 0.12 kg CO2/kg (next-gen MOF, 2.7 mmol/g, was 0.088)
- Sorbent replacement: 1.2%/year (improved stabilization, was 3%)
- Labor: $15.0M/year, 150 staff (advanced automation at 50x scale)
- Maintenance: $7.0M/year (predictive systems, sublinear scaling)
- Energy per tonne: 1,250 kWh/t (improved from 2,000 kWh/t, 37.5% reduction)
- Water: 0.8 m3/t (advanced closed-loop recovery, was 2.5 m3/t)
- Energy: $0 (renewable self-generation, no fuel purchases)
- Economies of scale: Fixed OPEX spread over 42.5 Mt/yr = $0.77/t

### Rendering
- Custom software 3D renderer using Pygame
- Painter's algorithm with back-face culling
- Per-mesh directional lighting + ambient
- Exploded, assembly, and cross-section views
- Ray-triangle intersection for hover-picking
- Scale bars and zoom indicators

## Technology Stack

- **Language:** Python 3.10+
- **Rendering:** Pygame (software 3D, no GPU required)
- **Math:** NumPy for vector/matrix operations
- **Dependencies:** `pygame`, `numpy`

## Running

```bash
pip install pygame numpy
python CC.py
```

Press **H** for controls, **I** for full in-app specification, **V** for verification checklist.

## 15-Year Global Deployment Plan

### How Many Giga-Plants to Make a Difference?

| Goal | Giga-Plants Needed | CO2 Removed | Investment |
|------|-------------------|-------------|------------|
| 1 Gt/year | ~24 | 1 Gt/yr | ~$1T CAPEX |
| 4 Gt/year (net-zero) | ~94 | 4 Gt/yr | ~$4T CAPEX |
| 10 Gt/year | ~235 | 10 Gt/yr | ~$10T CAPEX |
| Full net-zero (90% cuts) | 94-235 | 4-10 Gt/yr | $2-10T |

**Multi-story advantage:** 50x scale = 50x fewer plants vs standard 850 kt design (47,000 -> 94)

### Phase 1: First Wave (2026-2030)
- **6-24 giga-plants** built, removing 200 Mt - 1 Gt CO2/year
- $20B-$100B investment, 900-3,600 staff
- Focus: US (45Q credits), EU, China, Middle East, Australia
- 18-24 months construction per plant (factory-prefabricated)
- ~20% cost reduction per capacity doubling

### Phase 2: Rapid Scale-Up (2030-2035)
- **~235 giga-plants**, removing ~10 Gt/year (DAC) + ~1 Gt/year (EW)
- ~$10T cumulative CAPEX, ~35,000 staff globally
- Sorbent production: ~200 Mt/year (scaled supply chain)
- Annual OPEX: ~$10B ($1/t x 10 Gt)

### Phase 3: Multi-Gt/Year (2035+)
- **94-235 giga-plants** for net-zero (with 90% emissions cuts)
- Hybrid strategy: DAC + enhanced weathering + reforestation + biochar
- $2-10T total investment, ~600 ha per giga-plant

### Per-Giga-Plant Economics
| Metric | Value |
|--------|-------|
| CAPEX | ~$42.5B ($1000/t-year capacity) |
| OPEX | ~$1/t CO2 (renewable energy, $0 fuel) |
| All-in cost (w/ CAPEX) | ~$10-30/t at 50x scale |
| 15-year total OPEX | ~$500M fixed + ~$880M replacements |
| CO2 removed (15 yr) | 638 Mt = 9.2M cars off the road |

### Cost Trajectory
- Current first-of-a-kind: $400-1000/t
- Target 2030: ~$200/t
- Target 2035: ~$100/t (learning curve)
- Comparison: global fossil fuel subsidies = ~$7T/year (IMF)
- DAC for net-zero (4 Gt): ~$4T = redirecting fossil subsidies for <1 year

## Related Files

- `CC.py` -- Main simulation (single file)
- `ProjectGoal.md` -- Original project goals and research notes
- `Goalinformational.md` -- Extended research on DAC technologies
- `ReferenceCode/` -- Reference implementations and research materials
