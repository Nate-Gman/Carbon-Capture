#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 CC.py  --  CARBON CAPTURE :: Full-Scale DAC Digital Twin
================================================================================

A 100% standalone monolith that builds, animates and *operates* a full-scale
Direct Air Capture (DAC) facility in real time, mechanically and
thermodynamically to scale, in a single Python file.

WHAT IS DIRECT AIR CAPTURE?
  It's a technology that removes CO2 from the atmosphere to fight climate
  change. Giant fans pull air through a special filter that grabs CO2.
  When the filter is full, it's heated to release the CO2, which is then
  squeezed into liquid and pumped deep underground. Think of it as a
  giant air purifier for the whole planet -- powered entirely by clean
  energy (solar, wind, geothermal). No fossil fuels needed.

The facility captures CO2 directly from atmospheric air using a solid-sorbent
cyclic process, powered entirely by renewable energy (solar PV + solar thermal
+ wind + geothermal + battery storage). Every dimension is real (metres / SI)
and rendered to scale.

  AIR CONTACTORS     80 large fan arrays draw atmospheric air through
                     amine-functionalized solid sorbent beds. Each contactor
                     is 20 m x 12 m x 4 m (cross-flow slab geometry) with 8
                     fans of 3.5 m diameter. The sorbent selectively captures
                     CO2 at 420 ppm concentration.
  SORBENT CYCLE      Cyclic operation: CAPTURE (air through sorbent, ~65 min)
                     -> REGENERATION (sealed + heated to 100 C, CO2 released,
                     ~45 min) -> COOLING (~25 min). 80 beds in staggered
                     phases so capture is continuous.
  SOLAR THERMAL      Parabolic trough collectors (350,000 m2 aperture) heat
                     thermal oil to 150 C for regeneration heat -- more
                     efficient than electric heating. Molten-salt thermal
                     storage (1,500 MWh) carries regeneration through the night.
  SOLAR PV           0.5 km2 PV field (84 MW peak) powers fans, compressors,
                     vacuum pumps and controls. 800 MWh battery bank for
                     night operation.
  WIND               10 x 5 MW turbines (50 MW) supplement solar.
  GEOTHERMAL         12 wells providing 60 MW baseload thermal energy for
                     regeneration -- always-on heat source.
  CO2 COMPRESSION    4-stage centrifugal compressors liquefy CO2 to 50 bar
                     for storage and pipeline transport.
  CO2 STORAGE        8 pressurized tanks buffer liquid CO2 before pipeline
                     injection to geological sequestration.
  RENEWABLES-FIRST   USE ENERGY AS IT COMES IN -- solar thermal goes DIRECTLY
  SYNERGY            to regeneration. Solar PV runs fans + compressors first.
                     Surplus charges battery + thermal storage. Night:
                     thermal storage + geothermal + battery carry the facility.
                     Wind supplements whenever available. Goal: 2,300 t CO2/day,
                     ~850 kt CO2/year at $0 net energy cost.

Three modes (cycle with TAB):

  1. FACILITY  Orbit the whole plant to scale. Every subsystem spins/labels:
              air contactors (fans spinning), solar PV field, parabolic trough
              rows, wind turbines, geothermal wells, sorbent beds, regeneration
              units, CO2 compressors, storage tanks, cooling towers, control
              building, battery bank, thermal storage.
  2. CAPTURE   Orbit a single capture unit in mechanical detail -- the air
              contactor fans spin, the sorbent bed shows CO2 loading (color
              shift), the regeneration chamber heats, the CO2 collection
              manifold collects, the vacuum pump extracts, valves cycle.
              Exploded / section / assembly.
  3. OPERATION Run the capture campaign. Live day/night solar, wind, thermal
              storage, battery SOC, CO2 capture rate, storage filling,
              energy consumption, $/tonne CO2, and the running capture total
              vs a reference industrial emitter.

Dependencies:  numpy, pygame   Run:  python3 CC.py
Press  H  for controls,  I  for the full informational specification panel.
Every dimension in DIMS below is real (metres / SI) and rendered to scale.
================================================================================
"""

import math
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame


# =============================================================================
# SECTION 1 -- ENGINEERING SPECIFICATION (to scale, metres / SI)
# =============================================================================

# Whole-facility parameters. A 42.5 Mt CO2/year multi-story giga-plant --
# a 50x-scaled Direct Air Capture facility powered entirely by on-site renewables.
# Multi-story design (50 levels) dramatically reduces the number of plants
# needed globally: ~94 for net-zero (4 Gt/yr) instead of ~47,000.
# The 3D FACILITY view is built entirely from these, and the capture physics
# is cross-checked against the same numbers.
FACILITY = {
    "name":             "Carbon Capture DAC Giga-Plant",
    "land_area_m2":  6_000_000.0,   # 600 hectares (6.0 km2, multi-story saves land)
    "capture_t_year": 42_500_000.0,  # annual CO2 capture 42.5 Mt (50x scale, multi-story)
    "capture_t_day":   116_000.0,   # average daily capture
    "capture_t_hour":     4850.0,   # average hourly capture (peak 8750 t/h)
    "air_contactors":       4000,   # 50 stories x 80 contactors per level
    "sorbent_beds":         4000,   # one sorbent bed per contactor
    "regen_units":           800,   # shared regeneration units (5 beds each)
    "compressors":           100,   # CO2 liquefaction compressors
    "co2_storage_tanks":     200,   # buffer tanks before pipeline
    "staff":                 150,   # operating staff (advanced automation, 50x scale)
    "co2_atm_ppm":           420.0, # atmospheric CO2 concentration (2024)
    "co2_molar_mass":       44.009, # g/mol
    "air_molar_mass":       28.97,  # g/mol (dry air)
}

# DIMS: individual subsystems, to scale (metres). Everything the FACILITY view
# draws is positioned/sized from here so the model is dimensionally honest.
DIMS = {
    # --- Air contactors (the fan arrays that pull air through sorbent) ------
    # Based on real DAC designs: wide, low-profile cross-flow contactors
    # (similar to cooling-tower geometry used by Carbon Engineering / Climeworks)
    "contactor_w_m":       20.0,   # width (fan wall face, 20 m span)
    "contactor_h_m":       12.0,   # height (12 m tall, wider than tall)
    "contactor_d_m":        4.0,   # depth (4 m air-flow path through sorbent)
    "contactor_fans":        8,    # fans per contactor (4 wide x 2 tall)
    "fan_d_m":              3.5,   # fan diameter (industrial axial, 3.5 m)
    "fan_blades":            5,    # blades per fan (CFRP airfoil, 5-blade)
    "fan_hub_d_m":          0.7,   # fan hub diameter (0.7 m, Ti-6Al-4V)
    "sorbent_filter_d_m":   2.0,   # sorbent filter depth behind fans
    "contactor_frame_d_m":  0.5,   # structural frame thickness (galv. steel)
    "contactor_rows":        8,    # rows of contactors (10 per row)
    "contactor_per_row":    10,    # contactors per row
    "contactor_stories":     50,   # number of stacked levels (multi-story giga-tower)

    # --- Sorbent beds (the CO2-absorbing matrix inside each contactor) ------
    # Amine-functionalized silica/MOF honeycomb monolith
    "bed_w_m":             20.0,   # bed width (matches contactor interior, 20 m)
    "bed_h_m":             10.0,   # bed height (10 m, fills contactor face)
    "bed_d_m":              1.5,   # bed depth (1.5 m sorbent thickness)
    "sorbent_t_per_bed":    50.0,  # tonnes of sorbent per bed (~500 kg/m3 packed)
    "sorbent_cap_kg_kg":    0.12, # kg CO2 captured per kg sorbent per cycle (2.7 mmol/g, next-gen MOF)
    "sorbent_layers":        6,   # honeycomb channel layers (1 mm channels, 450 CPSI)

    # --- Regeneration units (heat the sorbent to release CO2) ---------------
    # Vacuum-temperature swing: 100 C, ~0.1 bar, SS 316L chamber
    "regen_w_m":           12.0,   # regeneration chamber width (12 m)
    "regen_h_m":            8.0,   # chamber height (8 m)
    "regen_d_m":           10.0,   # chamber depth (10 m)
    "regen_temp_c":       100.0,   # regeneration temperature (100 C, VSA)
    "regen_insul_d_m":      0.4,   # insulation thickness (0.4 m ceramic fiber)
    "regen_heater_rows":     6,    # heating element rows (ceramic IR)
    "regen_vacuum_d_m":     1.2,   # vacuum pump diameter (1.2 m, dry screw)
    "regen_vacuum_h_m":     2.5,   # vacuum pump height (2.5 m)

    # --- CO2 compressors (4-stage centrifugal liquefaction) -----------------
    "compressor_stages":      4,   # compression stages
    "comp_stage_d_m":       2.0,   # stage diameter
    "comp_stage_h_m":       1.5,   # stage height
    "comp_motor_d_m":       1.5,   # motor diameter
    "comp_motor_h_m":       2.5,   # motor height
    "comp_intercooler_d_m": 0.8,   # intercooler pipe diameter
    "comp_total_w_m":      18.0,   # total compressor train length
    "comp_total_h_m":       4.0,   # total height
    "comp_total_d_m":       6.0,   # total depth

    # --- CO2 storage tanks (pressurized buffer before pipeline) -------------
    "storage_tanks":       200,    # horizontal pressure cylinders (scaled)
    "storage_d_m":         20.0,   # tank diameter
    "storage_len_m":       40.0,   # tank length
    "storage_bar":         150.0,  # storage pressure (bar, pipeline-ready)
    "storage_capacity_t":  500.0,  # tonnes CO2 per tank
    "storage_insul_d_m":    0.3,   # insulation thickness

    # --- Solar PV field (electricity for fans, compressors, controls) -------
    "solar_pv_m2":    11_000_000.0, # total PV panel area (50x scale, bifacial)
    "solar_pv_eff":         0.32,   # panel efficiency (next-gen tandem perovskite)
    "solar_pv_peak_mw":  3520.0,    # peak electrical output
    "pv_panel_w_m":         2.2,    # single panel width
    "pv_panel_h_m":         5.5,    # single panel height (bifacial)
    "pv_panel_pitch_deg":   25.0,   # tilt angle
    "pv_rows":             220,     # rows of panels (reduced)
    "pv_row_spacing_m":     7.0,    # spacing between rows (tighter)
    "pv_field_w_m":       500.0,    # field width (50 ha PV field)
    "pv_field_l_m":      1000.0,    # field length

    # --- Solar thermal (parabolic troughs for regeneration heat) -----------
    "trough_aperture_m2":11_000_000.0, # total aperture area (50x scale)
    "trough_eff":            0.78,   # optical-to-thermal efficiency (improved coatings)
    "trough_peak_mw_th": 8580.0,    # peak thermal output
    "trough_aperture_w_m":   5.8,    # trough aperture width
    "trough_len_m":        150.0,    # single trough length
    "trough_rows":          640,     # number of trough rows (reduced)
    "trough_row_spacing_m":  5.5,    # spacing between rows (tighter)
    "trough_field_w_m":    450.0,    # field width (50 ha thermal field)
    "trough_field_l_m":   1000.0,    # field length

    # --- Thermal storage (molten salt for night regeneration) ---------------
    "thermal_storage_mwh": 35000.0,  # thermal energy storage capacity (50x scale)
    "salt_tank_d_m":        80.0,    # tank diameter (larger)
    "salt_tank_h_m":        35.0,    # tank height
    "salt_tanks":            20,     # 10 hot + 10 cold
    "salt_hot_temp_c":     380.0,   # hot tank temperature
    "salt_cold_temp_c":     290.0,  # cold tank temperature

    # --- Battery bank (night electrical operation) -------------------------
    "battery_mwh":       20000.0,   # battery storage capacity (scaled for 50x night ops)
    "battery_bldg_w_m":     50.0,   # battery building width
    "battery_bldg_h_m":     10.0,   # building height
    "battery_bldg_d_m":     30.0,   # building depth
    "battery_modules":       40,    # modular battery racks

    # --- Wind turbines (supplementary power) -------------------------------
    "wind_turbines":        150,    # number of turbines (scaled for 50x)
    "turbine_h_m":         200.0,   # hub height (taller for better wind)
    "turbine_d_m":         220.0,   # rotor diameter (larger for more swept area)
    "turbine_rated_mw":     15.0,   # rated power per turbine (next-gen)
    "turbine_blades":          3,   # blades per turbine
    "turbine_tower_d_m":     4.0,   # tower base diameter
    "turbine_tower_top_d_m": 2.0,   # tower top diameter

    # --- Geothermal wells (baseload regeneration heat) ---------------------
    "geo_wells":            200,    # number of wells (scaled for 50x)
    "geo_well_d_m":          0.5,   # wellhead diameter
    "geo_well_h_m":          8.0,   # wellhead height above ground
    "geo_mw_thermal":     2250.0,   # total thermal output (scaled baseload)
    "geo_pump_d_m":          1.0,   # pump station diameter
    "geo_pump_h_m":          3.0,   # pump station height

    # --- Cooling towers (heat rejection from compressors + regeneration) ----
    "cooling_towers":        48,    # number of cooling towers (scaled)
    "cooling_d_m":          15.0,   # tower base diameter
    "cooling_h_m":          25.0,   # tower height
    "cooling_top_d_m":       10.0,  # tower top diameter

    # --- Control building (facility operations center) ---------------------
    "control_w_m":          30.0,   # building width
    "control_h_m":          12.0,   # building height
    "control_d_m":          20.0,   # building depth

    # --- CO2 pipeline (to geological sequestration) ------------------------
    "pipeline_d_m":          0.6,   # pipeline diameter
    "pipeline_len_m":       500.0,  # visible length on-site

    # --- Air intake plenum (behind the fan wall) ---------------------------
    "plenum_d_m":            3.0,   # plenum depth behind fans

    # --- CO2 collection manifold (regeneration -> compressor) --------------
    "manifold_d_m":          0.8,   # manifold pipe diameter
    "manifold_rows":         64,    # collection pipe rows (scaled)
}

# The single scale that maps real metres -> renderer display units so the whole
# 6.0 km2 giga-facility frames cleanly in the orbit camera (1 unit ~ 600 m).
FAC_DISP = 1.0 / 600.0           # facility-view metres -> display units (6.0 km2 site)
CAP_DISP = 1.0 / 8.0             # capture-unit-view metres -> display units


def fcs(m):
    """metres -> FACILITY-view display units."""
    return m * FAC_DISP


# --- Energy conversion constants (SI) ---------------------------------------
CO2_ATM_PPM        = FACILITY["co2_atm_ppm"]      # atmospheric CO2 (ppm)
CO2_MOLAR_KG       = 0.044009                      # kg/mol
AIR_MOLAR_KG       = 0.02897                       # kg/mol (dry air)
AIR_DENSITY_KG_M3  = 1.225                         # kg/m3 at 15 C, 1 atm
CO2_DENSITY_KG_M3  = CO2_ATM_PPM * 1e-6 * AIR_DENSITY_KG_M3 * (CO2_MOLAR_KG / AIR_MOLAR_KG)
# ^ kg CO2 per m3 of air at 420 ppm ~ 0.000684 kg/m3

SOLAR_IRRADIANCE_W = 1000.0    # full-sun surface irradiance (W/m2)
WIND_REF_MS        = 12.0      # reference wind speed for rated turbine output

# --- Capture chemistry -------------------------------------------------------
# Sorbent: amine-functionalized porous polymer. Captures CO2 from air via
# chemisorption (amine-CO2 reaction). Regenerated by heating to ~100 C under
# vacuum, releasing concentrated CO2.
SORBENT = {
    "capacity_kg_per_kg":  DIMS["sorbent_cap_kg_kg"],   # 0.12 kg CO2 / kg sorbent (2.7 mmol/g, next-gen MOF)
    "regen_temp_c":        DIMS["regen_temp_c"],         # 100 C
    "capture_eff":         0.90,    # fraction of CO2 in passing air captured (optimized)
    "cycle_capture_min":   55.0,    # capture phase duration (minutes, faster sorbent)
    "cycle_regen_min":     35.0,    # regeneration phase duration (improved heat transfer)
    "cycle_cool_min":      20.0,    # cooling phase duration (faster)
    "cycle_total_min":     110.0,   # total cycle time (1.83 hours)
    "cycles_per_day":      13.09,   # 24h / 1.83h = 13.1 cycles/day
}

# --- Energy requirements per tonne CO2 ---------------------------------------
# These are realistic for solid-sorbent DAC with solar thermal regeneration.
ENERGY = {
    "regen_thermal_kwh_t":   750.0,  # thermal energy for regeneration (kWh/t CO2, advanced heat recovery)
    "fan_elec_kwh_t":        200.0,  # fan electricity (kWh/t CO2, optimized blades + lower pressure drop)
    "vacuum_elec_kwh_t":      80.0,  # vacuum pump electricity (kWh/t CO2, improved VSA)
    "compress_elec_kwh_t":   160.0,  # CO2 compression electricity (kWh/t CO2, improved)
    "aux_elec_kwh_t":         60.0,  # auxiliary/controls (kWh/t CO2, reduced)
    "total_thermal_kwh_t":   750.0,  # total thermal per tonne (2.7 GJ/t)
    "total_elec_kwh_t":      500.0,  # total electrical per tonne (0.50 MWh/t)
    "total_kwh_t":          1250.0,  # total energy per tonne CO2 (37.5% reduction from baseline)
}

# --- Solar PV array ----------------------------------------------------------
SOLAR_PV_PEAK_KW = DIMS["solar_pv_m2"] * DIMS["solar_pv_eff"] * SOLAR_IRRADIANCE_W / 1000.0

# --- Solar thermal (parabolic troughs) ---------------------------------------
SOLAR_TH_PEAK_KW = DIMS["trough_aperture_m2"] * DIMS["trough_eff"] * SOLAR_IRRADIANCE_W / 1000.0

# --- Wind turbines -----------------------------------------------------------
WIND_RATED_KW = DIMS["wind_turbines"] * DIMS["turbine_rated_mw"] * 1000.0

# --- Geothermal --------------------------------------------------------------
GEO_THERMAL_KW = DIMS["geo_mw_thermal"] * 1000.0

# --- Thermal storage ---------------------------------------------------------
THERMAL_STORE = {
    "capacity_mwh":   DIMS["thermal_storage_mwh"],
    "charge_eff":      0.92,    # thermal storage charge efficiency
    "discharge_eff":   0.88,    # thermal storage discharge efficiency
    "min_frac":        0.05,    # minimum thermal storage level
    "max_frac":        0.98,    # maximum thermal storage level
    "start_frac":      0.65,    # starting thermal storage level
}

# --- Battery -----------------------------------------------------------------
ELEC = {
    "batt_mwh":       DIMS["battery_mwh"],
    "soc_min":         0.10,    # never fully drain
    "soc_max":         0.98,
    "soc_start":       0.60,
    "batt_rt":         0.955,   # battery round-trip efficiency
}

# --- CO2 storage -------------------------------------------------------------
CO2_STORE = {
    "tanks":          DIMS["storage_tanks"],
    "capacity_t":     DIMS["storage_capacity_t"] * DIMS["storage_tanks"],
    "start_frac":      0.10,    # start partially filled
    "pipeline_rate_t_h": 250.0, # CO2 sent to sequestration (t/h) when available
}

# --- Storage round-trip efficiencies -----------------------------------------
STORAGE = {
    "battery_rt":   0.955,   # Li-ion battery round-trip
    "thermal_rt":   0.88,    # molten salt thermal round-trip (charge x discharge)
}

# --- Synergy control system --------------------------------------------------
# USE ENERGY AS IT COMES IN -- solar thermal goes DIRECTLY to regeneration.
# Solar PV runs fans + compressors first. Surplus charges battery + thermal.
# Night: thermal storage + geothermal + battery carry the facility.
SYNERGY = {
    "thermal_direct":    True,   # solar thermal -> regeneration directly
    "pv_direct":         True,   # solar PV -> fans + compressors directly
    "wind_supplement":   True,   # wind supplements PV
    "geo_baseload":      True,   # geothermal always provides baseload heat
    "thermal_charge":    0.70,   # fraction of thermal surplus to storage
    "batt_charge":       0.60,   # fraction of electrical surplus to battery
    "night_thermal_frac": 0.65,  # fraction of thermal from storage at night
    "night_batt_frac":   0.55,   # fraction of electrical from battery at night
    "capture_priority":  True,   # prioritize capture over storage charging
    "wind_curtail":      0.95,   # max wind fraction before curtailment
}

# --- Capture controller thresholds -------------------------------------------
# The facility auto-adjusts capture rate based on available energy. When energy
# is abundant (midday solar), all 4000 beds run at full cycle. When energy is
# limited (night, low wind), the facility reduces active beds to conserve.
CAPTURE_CTRL = {
    "beds_active_min":   1500,    # minimum beds running (geothermal + battery)
    "beds_active_max":   4000,    # maximum beds (full solar + wind + geo)
    "regen_power_floor_kw": GEO_THERMAL_KW,  # geothermal always provides this
    "fan_power_floor_kw":  200000.0,  # minimum fan power (battery always provides)
    "capture_rate_target_t_h": 8750.0,  # target capture rate (t CO2/h) -- peak
}

# --- Thermodynamics ----------------------------------------------------------
THERM = {
    "ambient_c":        20.0,    # ambient air temperature
    "regen_target_c":  100.0,    # regeneration target temperature
    "regen_heat_rate":   0.8,    # C per second heating rate (scaled)
    "cooling_rate":      0.3,    # C per second cooling rate
    "thermal_mass_kj_c": 500.0,  # bed thermal inertia (kJ/C)
}

# --- Operation campaign (the test run) ---------------------------------------
CAMPAIGN = {
    "name":          "15-YEAR COMMERCIAL OPERATION",
    "duration_days":  365.0,     # one year of operation (per year)
    "co2_ref_emit_t_year": 500_000.0,  # reference industrial emitter (t CO2/year)
    "cost_per_t_target":       200.0,  # target cost per tonne CO2 (2030 goal)
    "energy_cost_per_kwh":      0.02,  # assumed renewable energy cost ($/kWh, declining)
    "sorbent_replacement_frac": 0.012,  # sorbent capacity loss per year (improved stabilization)
    "sorbent_cost_per_t":    2000.0,   # sorbent cost ($/tonne, scaled MOF mass production)
    "years":                  15.0,   # multi-year longevity test duration
    # --- Real-world operational costs ---
    "labor_cost_per_year":  15_000_000.0,   # 150 staff x ~$100k avg loaded (economies of scale)
    "water_cost_per_m3":         0.50,      # $/m3 water
    "maint_cost_per_year":   7_000_000.0,   # annual maintenance (predictive, sublinear at 50x)
    "insurance_per_year":   10_000_000.0,   # insurance + permits (proven design at scale)
    "land_lease_per_year":     900_000.0,   # land lease (600 ha, remote area)
    "battery_repl_cost":    80_000_000.0,   # battery replacement cost (scaled, declining LFP)
    "sorbent_repl_cost_per_bed": 100_000.0, # per bed replacement (50t x $2000/t)
    # --- Embodied carbon (t CO2e to build, amortized over 15 years) ---
    "embodied_co2_t":         150_000.0,   # construction + equipment embodied carbon
    # --- Water consumption (m3 per tonne CO2) ---
    "water_m3_per_t_co2":        0.8,       # cooling + sorbent humidification (advanced closed-loop recovery)
    # --- CO2 purity target for pipeline ---
    "co2_purity_target":        0.995,      # 99.5% purity for geological sequestration
}

# --- Component reliability & maintenance specs --------------------------------
# Each component has: design_life_years, mtbf_h (mean time between failures),
# maint_interval_h (preventive maintenance interval), degradation_rate (% per
# year of capacity loss), failure_mode (most common), repair_h (mean repair time).
COMPONENTS = {
    "fans": {
        "design_life_years":   30.0,
        "mtbf_h":         1000000.0,   # contactless magnetic bearings, essentially no wear
        "maint_interval_h":   2190.0,  # quarterly vibration check
        "degradation_rate":     0.003, # 0.3%/year (premium bearings)
        "failure_mode":    "sensor anomaly (pre-empted)",
        "repair_h":              2.0,  # swap module, hot-swap
        "material":       "CFRP blades, Ti-6Al-4V hub, active magnetic bearings, contactless",
        "quantity":       FACILITY["air_contactors"] * DIMS["contactor_fans"],
        "predictive_factor":   0.999, # 99.9% of issues caught before failure
    },
    "sorbent": {
        "design_life_years":   15.0,
        "mtbf_h":          999999.0,   # degrades, doesn't "fail"
        "maint_interval_h":  720.0,    # monthly inspection
        "degradation_rate":     0.015, # 1.5%/year (advanced stabilized amine)
        "failure_mode":    "amine degradation (gradual)",
        "repair_h":             48.0,  # 2 days per bed replacement
        "material":       "PEI on mesoporous silica / MOF support, antioxidant + thermal stabilized",
        "quantity":       FACILITY["sorbent_beds"],
        "predictive_factor":   1.0,   # no random failures, only degradation
    },
    "regen_units": {
        "design_life_years":   30.0,
        "mtbf_h":          800000.0,   # SiC ceramic heaters, 300% derated
        "maint_interval_h":   4380.0,  # 6-month thermal inspection
        "degradation_rate":     0.002, # 0.2%/year
        "failure_mode":    "element resistance drift (monitored)",
        "repair_h":              4.0,  # swap heater cartridge
        "material":       "SS 316L vessels, SiC ceramic heaters, 300% derated",
        "quantity":       FACILITY["regen_units"],
        "predictive_factor":   0.99,
    },
    "vacuum_pumps": {
        "design_life_years":   25.0,
        "mtbf_h":          800000.0,   # dry screw, magnetic coupling, hermetic
        "maint_interval_h":   2190.0,  # quarterly
        "degradation_rate":     0.003, # 0.3%/year
        "failure_mode":    "bearing temp rise (pre-empted)",
        "repair_h":              6.0,  # swap cartridge
        "material":       "SS 316L dry screw pump, magnetic coupling, hermetic, no oil",
        "quantity":       FACILITY["regen_units"],
        "predictive_factor":   0.99,
    },
    "compressors": {
        "design_life_years":   30.0,
        "mtbf_h":         1000000.0,   # diaphragm type, hermetic, N+2
        "maint_interval_h":   4380.0,  # 6-month
        "degradation_rate":     0.002, # 0.2%/year
        "failure_mode":    "diaphragm stress (ultrasonic detected)",
        "repair_h":              8.0,  # swap diaphragm module
        "material":       "SS 316L diaphragm compressor, hermetic, N+2 redundancy",
        "quantity":       FACILITY["compressors"],
        "predictive_factor":   0.99,
    },
    "solar_pv": {
        "design_life_years":   35.0,
        "mtbf_h":         2000000.0,   # solid-state, no moving parts
        "maint_interval_h":   4380.0,  # quarterly cleaning + IV curve check
        "degradation_rate":     0.003, # 0.3%/year (premium panels)
        "failure_mode":    "string mismatch (string-level monitoring)",
        "repair_h":              2.0,  # swap string inverter
        "material":       "monocrystalline Si, tempered glass, IP68 junction boxes",
        "quantity":       1,
        "predictive_factor":   0.99,
    },
    "solar_thermal": {
        "design_life_years":   30.0,
        "mtbf_h":          800000.0,
        "maint_interval_h":   4380.0,  # quarterly mirror reflectivity check
        "degradation_rate":     0.003, # 0.3%/year
        "failure_mode":    "mirror reflectivity drop (cleaned before failure)",
        "repair_h":              4.0,
        "material":       "silvered glass mirrors, SS 316L structure, robotic cleaning",
        "quantity":       1,
        "predictive_factor":   0.98,
    },
    "wind_turbines": {
        "design_life_years":   30.0,
        "mtbf_h":         1200000.0,   # direct-drive PMG, no gearbox, no contact
        "maint_interval_h":   2190.0,  # quarterly condition monitoring
        "degradation_rate":     0.003, # 0.3%/year
        "failure_mode":    "bearing temp (SCADA pre-empted)",
        "repair_h":             12.0,  # crane-assisted swap
        "material":       "CFRP blades, direct-drive PMG, cast iron nacelle, magnetic bearings",
        "quantity":       DIMS["wind_turbines"],
        "predictive_factor":   0.99,
    },
    "geothermal": {
        "design_life_years":   35.0,
        "mtbf_h":         1000000.0,   # hermetic pump, downhole sensor
        "maint_interval_h":   4380.0,  # quarterly
        "degradation_rate":     0.002, # 0.2%/year
        "failure_mode":    "scaling buildup (acid-flushed before failure)",
        "repair_h":              8.0,
        "material":       "Ti-6Al-4V wellhead, SS 316L hermetic pump, anti-scale injection",
        "quantity":       DIMS["geo_wells"],
        "predictive_factor":   0.98,
    },
    "battery": {
        "design_life_years":   25.0,
        "mtbf_h":         1200000.0,   # LFP chemistry, per-cell BMS, liquid cooled
        "maint_interval_h":   2190.0,  # quarterly capacity test
        "degradation_rate":     0.005, # 0.5%/year (LFP, thermal-managed)
        "failure_mode":    "cell imbalance (BMS auto-rebalances)",
        "repair_h":              2.0,  # swap module, hot-swap
        "material":       "LiFePO4 (LFP), modular racks, per-cell BMS, liquid cooling, prismatic cells",
        "quantity":       DIMS["battery_modules"],
        "predictive_factor":   0.995,
    },
    "thermal_store": {
        "design_life_years":   35.0,
        "mtbf_h":         1500000.0,   # passive tanks, minimal wear
        "maint_interval_h":   8760.0,  # annual
        "degradation_rate":     0.001, # 0.1%/year
        "failure_mode":    "trace heating element (redundant)",
        "repair_h":              4.0,
        "material":       "SS 316L tanks, nitrate salt, trace heating backup",
        "quantity":       DIMS["salt_tanks"],
        "predictive_factor":   0.99,
    },
    "co2_tanks": {
        "design_life_years":   35.0,
        "mtbf_h":         2000000.0,   # static pressure vessels
        "maint_interval_h":   8760.0,  # annual UT inspection
        "degradation_rate":     0.0005,# 0.05%/year
        "failure_mode":    "valve seat wear (redundant valves)",
        "repair_h":              4.0,
        "material":       "SS 316L clad, internal epoxy, double-block valves",
        "quantity":       DIMS["storage_tanks"],
        "predictive_factor":   0.99,
    },
    "pipeline": {
        "design_life_years":   50.0,
        "mtbf_h":         5000000.0,   # buried, cathodically protected
        "maint_interval_h":   8760.0,  # annual smart pigging
        "degradation_rate":     0.0005,
        "failure_mode":    "coating thinning (detected by pigging)",
        "repair_h":              8.0,
        "material":       "SS 316L clad, external coating, cathodic protection, fiber-optic monitoring",
        "quantity":       1,
        "predictive_factor":   0.995,
    },
    "control_system": {
        "design_life_years":   20.0,
        "mtbf_h":         1000000.0,   # triple-redundant PLC
        "maint_interval_h":   2190.0,  # quarterly firmware + sensor cal
        "degradation_rate":     0.0,   # software, no physical degradation
        "failure_mode":    "sensor drift (auto-calibrated)",
        "repair_h":              2.0,  # swap I/O module
        "material":       "triple-redundant PLC, hot-spare servers, self-calibrating sensors",
        "quantity":       1,
        "predictive_factor":   0.995,
    },
}

# MATERIALS: per-component fabrication materials, specs, and cost for Tab 2 (capture unit)
# Each entry: material, grade, fabrication method, est_cost_usd, cost_reduction note
MATERIALS = {
    "frame": {
        "material": "Galvanized structural steel (A36)",
        "grade": "ASTM A36, hot-dip galvanized Z275",
        "fabrication": "Welded box sections, bolted field joints, 0.3m sections",
        "est_cost_usd": 180000,
        "cost_reduction": "Use HSS tube instead of I-beam (-15% weight, -8% cost)",
    },
    "foundation": {
        "material": "Reinforced concrete",
        "grade": "C32/40, B500B rebar, 200mm slab",
        "fabrication": "Cast in-situ, vibration-finished, curing compound",
        "est_cost_usd": 25000,
        "cost_reduction": "Use recycled aggregate (-5% cost, -30% embodied CO2)",
    },
    "columns": {
        "material": "Structural steel pipe (HSS)",
        "grade": "ASTM A500 Gr B, 400mm dia, 8mm wall",
        "fabrication": "Cut + welded base plates, anchor bolts to foundation",
        "est_cost_usd": 15000,
        "cost_reduction": "Use spun concrete poles for non-seismic sites (-40% cost)",
    },
    "walkway": {
        "material": "Galvanized steel grating + aluminum railing",
        "grade": "ISO 14122-2 compliant, 30mm mesh grating",
        "fabrication": "Bolted grating clips, aluminum extrusion rails",
        "est_cost_usd": 8000,
        "cost_reduction": "Use FRP grating (-20% cost, -60% weight, corrosion-free)",
    },
    "ladder": {
        "material": "Galvanized steel, safety cage",
        "grade": "OSHA 1910.28 compliant, 450mm rung spacing",
        "fabrication": "Welded rungs to side rails, bolted to frame",
        "est_cost_usd": 3000,
        "cost_reduction": "Use aluminum ladder (-30% cost, -50% weight)",
    },
    "fans": {
        "material": "CFRP blades, Ti-6Al-4V hub, SS 304 shroud",
        "grade": "Aerospace-grade CFRP (T800 carbon), Ti Grade 5 hub",
        "fabrication": "Autoclave layup blades, CNC-machined hub, magnetic bearings",
        "est_cost_usd": 320000,
        "cost_reduction": "Use glass-fiber blades for low-speed fans (-60% cost, +15% weight)",
    },
    "fan_motor": {
        "material": "Direct-drive PMAC motor",
        "grade": "IE5 efficiency, NdFeB magnets, water-cooled stator",
        "fabrication": "Integrated hub motor, sealed bearings, VFD control",
        "est_cost_usd": 95000,
        "cost_reduction": "Use ferrite magnets instead of NdFeB (-25% cost, +10% size)",
    },
    "sorbent_bed": {
        "material": "PEI on mesoporous silica / MOF support",
        "grade": "Next-gen amine-functionalized MOF-74, 450 CPSI honeycomb",
        "fabrication": "Washcoated monolith, steel frame cassette, slide-in module",
        "est_cost_usd": 100000,
        "cost_reduction": "Use pellet bed instead of monolith (-30% cost, -10% efficiency)",
    },
    "bed_frame": {
        "material": "SS 304 cassette frame",
        "grade": "ASTM A240 304, 2B finish, 3mm sheet",
        "fabrication": "Laser-cut + press-braked, slide-in rails on contactor",
        "est_cost_usd": 12000,
        "cost_reduction": "Use galvanized steel instead of SS 304 (-50% cost, rust risk)",
    },
    "regen_chamber": {
        "material": "SS 316L pressure vessel",
        "grade": "ASME B31.3, 316L sheet + structural, 0.1 bar vacuum rated",
        "fabrication": "TIG welded, X-ray inspected, post-weld anneal",
        "est_cost_usd": 280000,
        "cost_reduction": "Use SS 304L where Cl <50ppm (-20% cost, corrosion risk in marine)",
    },
    "insulation": {
        "material": "Ceramic fiber blanket",
        "grade": "Al2O3-SiO2, 1400C rated, 0.4m thickness",
        "fabrication": "Lagged + cladded with aluminum sheet, SS wire anchors",
        "est_cost_usd": 18000,
        "cost_reduction": "Use mineral wool (850C rated) for 100C process (-40% cost)",
    },
    "heaters": {
        "material": "SiC ceramic IR elements",
        "grade": "Silicon carbide, 300% derated, 1200C max",
        "fabrication": "Cartridge-style, swappable, Inconel 600 terminals",
        "est_cost_usd": 45000,
        "cost_reduction": "Use NiCr wire elements (-60% cost, shorter life, +5% energy)",
    },
    "vacuum_pump": {
        "material": "SS 316L dry screw, magnetic coupling",
        "grade": "Hermetic, oil-free, PTFE lip seals",
        "fabrication": "Precision-machined screw rotors, magnetic coupling, VFD",
        "est_cost_usd": 85000,
        "cost_reduction": "Use liquid ring pump (-50% cost, +water consumption, +maintenance)",
    },
    "manifold": {
        "material": "SS 316L piping, welded",
        "grade": "ASTM A312 TP316L, 800mm dia, sch 10s",
        "fabrication": "Orbital TIG welded, trace-heated, X-ray inspected joints",
        "est_cost_usd": 35000,
        "cost_reduction": "Use SS 304L (-20% cost) or HDPE for low-temp sections (-60%)",
    },
    "valves": {
        "material": "SS 316L body, PTFE seats, pneumatic actuator",
        "grade": "API 608, 3-piece ball valve, spring-return actuator",
        "fabrication": "Cast body, machined seats, double-block-and-bleed",
        "est_cost_usd": 22000,
        "cost_reduction": "Use SS 304 body for non-corrosive service (-25% cost)",
    },
    "plenum": {
        "material": "SS 304 panels on galvanized frame",
        "grade": "ASTM A240 304, 2mm sheet, removable panels",
        "fabrication": "Bolted panels on frame, gasket-sealed, quick-release fasteners",
        "est_cost_usd": 15000,
        "cost_reduction": "Use aluminum panels (-15% cost, -50% weight, dents easier)",
    },
    "output_pipe": {
        "material": "SS 316L, trace-heated",
        "grade": "ASTM A312 TP316L, 400mm dia, sch 10s",
        "fabrication": "Orbital welded, heat-traced with self-regulating cable",
        "est_cost_usd": 8000,
        "cost_reduction": "Use SS 304L for <100C service (-20% cost)",
    },
    "fasteners": {
        "material": "SS 316 bolts + nuts",
        "grade": "ASTM A193 B8M / A194 8M, Teflon-coated",
        "fabrication": "Torque-controlled installation, anti-seize compound",
        "est_cost_usd": 5000,
        "cost_reduction": "Use galvanized Grade 8.8 for non-corrosive areas (-70% cost)",
    },
}

# Total capture unit fabrication cost (sum of all MATERIALS entries)
CAPTURE_UNIT_COST = sum(v["est_cost_usd"] for v in MATERIALS.values())


MAINTENANCE = {
    "daily_inspect_h":        0.5,   # 30min/day automated sensor sweep
    "weekly_filter_h":        2.0,   # 2h/week robotic filter cleaning
    "monthly_sorbent_h":      4.0,   # 4h/month sorbent capacity test
    "semiannual_major_h":    24.0,   # 24h/6mo major PM (reduced via automation)
    "annual_overhaul_h":     60.0,   # 60h/year full PM (reduced via predictive)
    "sorbent_replace_h":     48.0,   # per bed, at ~12 years
    "battery_replace_h":     24.0,   # per rack, at ~20 years
    "staff_maint":              20,  # 20 of 150 staff (advanced automated maintenance)
    "annual_maint_cost":  7_000_000.0,  # $7M/year (sublinear, predictive at 50x scale)
    "predictive_systems":      True,  # condition-based monitoring active
    "vibration_monitored":     True,  # all rotating equipment
    "thermal_monitored":       True,  # all thermal components
    "auto_rebalance":          True,  # battery BMS auto-rebalance
    "robotic_cleaning":        True,  # PV + mirrors + filters
}

# --- Urban mini-plant (skyscraper floor DAC unit) ----------------------------
# A compact Direct Air Capture module designed to fit inside vacant commercial
# building floors. Each unit occupies ~500 m2 of floor space (one floor of a
# typical 1000 m2 skyscraper plate), uses 4 small contactors with compact
# regen, and runs off the building's electrical supply (with optional rooftop
# solar). Designed for distributed deployment in any city worldwide.
URBAN = {
    "name":              "Urban DAC Mini-Plant",
    "floor_area_m2":     500.0,     # footprint per unit (half a typical floor plate)
    "floor_height_m":    4.0,       # ceiling clearance (standard commercial floor)
    "contactors":          4,       # compact contactor units (2x2 grid)
    "contactor_w_m":     3.0,       # width per unit (compact, fits through freight elevator)
    "contactor_h_m":     2.5,       # height (fits under 4 m ceiling with ducting)
    "contactor_d_m":     1.5,       # depth (slim profile against wall)
    "fans_per_contactor":  3,       # 3 compact fans per unit (0.8 m dia)
    "fan_d_m":           0.8,       # fan diameter (commercial HVAC size)
    "fan_blades":          4,       # blades per fan (CFRP, quiet operation)
    "sorbent_beds":        4,       # one bed per contactor
    "sorbent_t_per_bed":   2.0,     # 2 tonnes per bed (compact monolith)
    "sorbent_cap_kg_kg":  0.12,     # same next-gen MOF as giga-plant
    "regen_units":         2,       # 2 compact regen chambers (2 beds each)
    "regen_w_m":          2.0,      # regen chamber width
    "regen_h_m":          2.0,      # regen chamber height
    "regen_d_m":          1.5,      # regen chamber depth
    "compressor_units":    1,       # single compact CO2 compressor
    "co2_tanks":           2,       # 2 buffer tanks (5 t each)
    "co2_tank_cap_t":      5.0,     # tonnes CO2 per tank
    "staff":               0,       #无人值守 (unattended, remote monitoring)
    "monitoring":        "remote",  # cloud-based predictive maintenance
    # --- Performance ---
    "capture_t_year":   2000.0,     # 2,000 t CO2/year per unit
    "capture_t_day":       5.5,     # average daily capture
    "capture_t_hour":      0.23,    # average hourly capture
    "energy_kwh_t":     1500.0,     # kWh/t (higher than giga-plant, smaller = less efficient)
    "power_kw":          34.0,      # average power draw (kW)
    "power_peak_kw":     55.0,      # peak power draw (kW)
    # --- Economics ---
    "capex_usd":     2_000_000.0,   # ~$2M per unit (factory-built, mass-produced)
    "opex_per_t":          15.0,    # $/t OPEX (building power, maintenance, sorbent)
    "all_in_per_t":       80.0,     # $/t all-in (with CAPEX amortized over 15 yr)
    "energy_cost_kwh":    0.12,     # $/kWh (commercial building electricity rate)
    "sorbent_repl_cost":  20_000.0, # $ per bed replacement (2t x $10k/t small-scale)
    "sorbent_repl_years":   10.0,   # replacement interval (harder operating conditions)
    "maint_per_year":    15_000.0,  # annual maintenance (contracted service)
    # --- Building integration ---
    "building_floors":      50,     # typical skyscraper floors
    "floors_per_unit":       1,     # one floor per unit
    "units_per_building":   10,     # 10 vacant floors converted in a typical building
    "building_capture_t_yr": 20_000.0,  # 10 units x 2,000 t/yr = 20 kt/yr per building
    "co2_pipeline":    "building_co2_riser",  # CO2 routed via building riser to street collection
    # --- Urban deployment ---
    "cities_targeted":    100,      # major cities worldwide
    "buildings_per_city":  50,      # average buildings with vacant floors per city
    "units_global":     50_000,     # 100 cities x 50 buildings x 10 units
    "global_capture_t_yr": 100_000_000.0,  # 100 Mt/yr from urban units alone
    # --- Environment ---
    "noise_db":            45,      # dB at 1m (quiet, office-compatible)
    "water_m3_t":         0.5,      # minimal water use (closed-loop)
    "co2_purity":        0.99,      # 99% purity (street collection standard)
}


# =============================================================================
# SECTION 2 -- COLORS & THEME
# =============================================================================

BG_TOP      = (10, 16, 26)
BG_BOT      = (3, 5, 9)
C_GROUND    = (62, 72, 55)       # ground / terrain
C_GROUND_HI = (82, 92, 68)       # ground highlight
C_CONTACTOR = (90, 105, 125)      # contactor frame (brighter steel-blue)
C_FAN       = (200, 210, 225)    # fan blades (brighter)
C_FAN_HUB   = (120, 130, 145)    # fan hub (darker for contrast vs blades)
C_SORBENT   = (120, 170, 100)    # sorbent (brighter green for visibility)
C_SORBENT_LOADED = (200, 160, 70) # sorbent loaded with CO2 (amber)
C_SORBENT_REGEN  = (240, 120, 50) # sorbent regenerating (hot orange)
C_SORBENT_COOL   = (90, 130, 190) # sorbent cooling (blue)
C_REGEN      = (150, 100, 70)    # regeneration chamber (warmer brown for contrast)
C_REGEN_HOT  = (255, 120, 30)    # hot regeneration (brighter orange)
C_COMP       = (170, 180, 195)   # compressor (brighter steel)
C_COMP_HOT   = (220, 140, 50)    # compressor hot stage
C_CO2TANK    = (210, 216, 224)   # CO2 storage tank
C_CO2BAND    = (80, 200, 240)    # CO2 marking band (brighter cyan)
C_PV         = (38, 66, 132)     # PV panel deep blue
C_PV_HI      = (86, 150, 230)    # PV cell shimmer
C_PV_FRAME   = (120, 120, 125)   # PV panel frame
C_TROUGH     = (200, 200, 100)   # parabolic trough (reflective)
C_TROUGH_HI  = (255, 245, 180)   # trough highlight
C_TROUGH_PIPE = (180, 80, 40)   # heat collection pipe
C_WIND       = (220, 225, 230)   # wind turbine
C_WIND_TOWER = (180, 185, 190)   # wind tower
C_GEO        = (160, 100, 60)    # geothermal well
C_GEO_STEAM  = (200, 200, 210)   # geothermal steam
C_COOLING    = (160, 170, 180)   # cooling tower
C_COOLING_STEAM = (220, 225, 230)
C_BATT       = (60, 120, 96)     # battery building
C_SALT_HOT   = (220, 100, 50)    # hot salt tank
C_SALT_COLD  = (80, 120, 180)    # cold salt tank
C_CONTROL    = (176, 182, 190)   # control building
C_CONTROL_WIN = (40, 60, 84)     # control building windows
C_PIPELINE   = (120, 130, 140)   # CO2 pipeline
C_MAST       = (140, 145, 150)   # antenna/mast
C_BUILDING   = (70, 78, 92)      # skyscraper structure (concrete/steel)
C_BUILDING_HI= (90, 100, 115)    # skyscraper highlight
C_FLOOR      = (55, 62, 72)      # building floor slab
C_FLOOR_EDGE = (80, 88, 100)     # floor slab edge
C_WINDOW     = (30, 50, 80)      # building windows (dark glass)
C_WINDOW_LIT = (80, 120, 180)    # lit windows
C_URBAN_FRAME= (100, 115, 135)   # urban DAC unit frame (compact steel)
C_URBAN_FAN  = (140, 155, 175)   # urban DAC fan blades
C_URBAN_HUB  = (120, 130, 145)   # urban DAC fan hub
C_URBAN_SORB = (110, 160, 100)   # urban sorbent bed (compact)
C_URBAN_REGEN= (180, 120, 80)    # urban regen chamber (compact)
C_URBAN_TANK = (140, 150, 165)   # urban CO2 buffer tank
C_URBAN_COMP = (130, 140, 155)   # urban compressor
C_URBAN_DUCT = (100, 110, 125)   # urban ducting
C_TEXT       = (224, 230, 238)
C_TEXT_DIM   = (150, 160, 175)
C_ACCENT     = (90, 200, 255)
C_GOOD       = (90, 220, 130)
C_WARN       = (255, 200, 60)
C_BAD        = (255, 90, 90)
C_CO2        = (90, 180, 220)    # CO2 accent (blue)
C_PANEL      = (16, 22, 32)
C_PANEL_HI   = (28, 38, 54)
C_SKY1       = (58, 104, 158)
C_SKY2       = (150, 190, 224)
C_SKY_NIGHT1 = (6, 10, 24)
C_SKY_NIGHT2 = (22, 34, 58)
C_GROUND_NIGHT = (20, 26, 20)


# =============================================================================
# SECTION 3 -- MINI 3D ENGINE (software renderer, painter's algorithm)
# Reused, proven toolkit -- geometry-agnostic.
# =============================================================================

VISUAL_DETAIL = 1.6


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


class Mesh:
    """A bag of vertices + polygon faces with a base color. Coordinates are in
    display units, principal axis along +Z. `spin` is the rotation ratio relative
    to the master angle; `pivot` offsets the local origin; `tilt` is a static
    (rx, ry) rotation used for angled/off-axis spinners (props, turbines)."""

    def __init__(self, verts, faces, color, name="", spin=1.0, group="default",
                 pivot=(0.0, 0.0, 0.0), tilt=(0.0, 0.0), hot=False,
                 selectable=False, sorbent_state=None):
        self.verts = np.asarray(verts, dtype=float)
        # Triangulate any quad faces to triangles for vectorized rendering
        if faces and len(faces[0]) > 3:
            tri_faces = []
            for face in faces:
                if len(face) == 3:
                    tri_faces.append(face)
                elif len(face) == 4:
                    tri_faces.append((face[0], face[1], face[2]))
                    tri_faces.append((face[0], face[2], face[3]))
                else:
                    for i in range(1, len(face) - 1):
                        tri_faces.append((face[0], face[i], face[i + 1]))
            self.faces = tri_faces
        else:
            self.faces = faces
        self.faces_np = np.asarray(self.faces, dtype=np.intp)
        self.color = color
        self.name = name
        self.spin = spin
        self.group = group
        self.pivot = np.asarray(pivot, dtype=float)
        self.tilt = tilt
        self.hot = hot
        self.chamber_index = None
        self.selectable = selectable
        self.sorbent_state = sorbent_state  # for sorbent bed color shifting
        # Precompute tilt rotation matrix
        rx, ry = tilt
        if rx or ry:
            self._tilt_mat = (rot_x(rx) @ rot_y(ry)).T
        else:
            self._tilt_mat = None
        # For static meshes, precompute tilted verts (skip per-frame rot)
        if not spin and self._tilt_mat is not None:
            self._tilted_verts = self.verts @ self._tilt_mat
        else:
            self._tilted_verts = self.verts if not spin else None
        # For static non-selectable meshes, precompute verts+pivot
        if not spin and not selectable:
            self._world_static = self._tilted_verts + self.pivot
        else:
            self._world_static = None

    def world_verts(self, angle, selector_radius=None):
        if self._world_static is not None:
            return self._world_static
        if self.spin:
            v = self.verts @ rot_z(angle * self.spin).T
            if self._tilt_mat is not None:
                v = v @ self._tilt_mat
        else:
            v = self._tilted_verts
        if self.selectable and selector_radius is not None:
            pivot = self.pivot.copy()
            pivot[0] = selector_radius
            return v + pivot
        return v + self.pivot


# ---- primitive builders ----------------------------------------------------

def _detail_seg(seg):
    return max(8, int(round(seg * VISUAL_DETAIL)))


def _annulus_cylinder(r_out, r_in, z0, z1, seg=44):
    """Hollow tube (ring) closed at both axial ends."""
    seg = _detail_seg(seg)
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z in (z0, z1):
        for a in ang:
            verts.append((r_out * math.cos(a), r_out * math.sin(a), z))
        for a in ang:
            verts.append((r_in * math.cos(a), r_in * math.sin(a), z))

    def oo(layer, i):
        return layer * (2 * seg) + (i % seg)

    def ii(layer, i):
        return layer * (2 * seg) + seg + (i % seg)

    for i in range(seg):
        faces.append((oo(0, i), oo(0, i + 1), oo(1, i + 1), oo(1, i)))
        faces.append((ii(0, i), ii(1, i), ii(1, i + 1), ii(0, i + 1)))
        faces.append((oo(0, i), ii(0, i), ii(0, i + 1), oo(0, i + 1)))
        faces.append((oo(1, i), oo(1, i + 1), ii(1, i + 1), ii(1, i)))
    return verts, faces


def _solid_cylinder(r, z0, z1, seg=40):
    seg = _detail_seg(seg)
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z in (z0, z1):
        for a in ang:
            verts.append((r * math.cos(a), r * math.sin(a), z))
    c0 = len(verts)
    verts.append((0, 0, z0))
    c1 = len(verts)
    verts.append((0, 0, z1))
    for i in range(seg):
        a, b = i, (i + 1) % seg
        faces.append((a, b, seg + b, seg + a))
        faces.append((c0, b, a))
        faces.append((c1, seg + a, seg + b))
    return verts, faces


def _box(cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = [(cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
         (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
         (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
         (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz)]
    f = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
         (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return v, f


def _hull(sections):
    """Loft a smooth body shell from a list of rectangular cross-sections along Z.
    Each section is (z, halfwidth, y_bottom, y_top)."""
    verts, faces = [], []
    rings = []
    for (z, hw, y0, y1) in sections:
        base = len(verts)
        verts += [(-hw, y0, z), (hw, y0, z), (hw, y1, z), (-hw, y1, z)]
        rings.append(base)
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((a + k, a + k2, b + k2, b + k))
    faces.append((rings[0], rings[0] + 1, rings[0] + 2, rings[0] + 3))
    last = rings[-1]
    faces.append((last + 3, last + 2, last + 1, last))
    return verts, faces


def _smooth_sections(ctrl, sub=3):
    """Catmull-Rom subdivide a list of loft cross-sections for a smoother shell."""
    pts = [np.array(s, dtype=float) for s in ctrl]
    n = len(pts)
    out = []
    for i in range(n - 1):
        p0, p1, p2, p3 = pts[max(0, i - 1)], pts[i], pts[i + 1], pts[min(n - 1, i + 2)]
        for j in range(sub):
            t = j / sub
            t2, t3 = t * t, t * t * t
            s = 0.5 * ((2 * p1) + (-p0 + p2) * t
                       + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(tuple(s))
    out.append(tuple(pts[-1]))
    return out


def _pipe(p0, p1, r, col, seg=8):
    """A straight round pipe/rod between two 3D points."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    axis = p1 - p0
    L = np.linalg.norm(axis)
    if L < 1e-9:
        return Mesh(*_solid_cylinder(r, 0, 0.001, seg=seg), col)
    axis = axis / L
    tmp = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(axis, tmp); u = u / (np.linalg.norm(u) or 1)
    w = np.cross(axis, u)
    verts, faces = [], []
    ring = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for end in (p0, p1):
        for a in ring:
            verts.append(tuple(end + r * math.cos(a) * u + r * math.sin(a) * w))
    for i in range(seg):
        a, b = i, (i + 1) % seg
        faces.append((a, b, seg + b, seg + a))
    return Mesh(verts, faces, col, spin=0.0)


class Part:
    """A named, spec'd logical component made of one or more meshes. Carries an
    assembly `order`, an `explode` offset and a `specs` list for the inspector."""

    def __init__(self, key, name, meshes, specs, order, explode, color):
        self.key = key
        self.name = name
        self.meshes = meshes
        self.specs = specs
        self.order = order
        self.explode = np.asarray(explode, dtype=float)
        self.color = color
        n = np.linalg.norm(self.explode)
        self.popdir = self.explode / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
        self.fire_anchor = None


def _grp(meshes, group):
    for m in meshes:
        m.group = group
    return meshes


def _place_spinner(meshes, pivot, tilt, group):
    """Turn origin-built meshes into an OFF-AXIS spinner (fans, turbines)."""
    piv = np.asarray(pivot, dtype=float)
    for m in meshes:
        m.pivot = piv.copy()
        m.tilt = tilt
        m.group = group
    return meshes


def _mix(c1, c2, t):
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))


def _merge_static_meshes(meshes, col, name=""):
    """Merge multiple static meshes into one, combining their world-space
    vertices and faces. All meshes must be static (spin=0)."""
    all_verts = []
    all_faces = []
    for m in meshes:
        base = len(all_verts)
        # Compute world verts from current tilt/pivot
        v = m.verts
        rx, ry = m.tilt
        if rx or ry:
            v = v @ (rot_x(rx) @ rot_y(ry)).T
        all_verts.append(v + m.pivot)
        for face in m.faces:
            all_faces.append((face[0] + base, face[1] + base, face[2] + base))
    if not all_verts:
        return _static([], [], col, name)
    verts = np.vstack(all_verts)
    return Mesh(verts, all_faces, col, name=name, spin=0.0)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

# =============================================================================
# SECTION 4 -- GEOMETRY BUILDERS (facility + capture unit, to scale)
# =============================================================================

def _cap(m):
    """metres -> CAPTURE-unit-view display units."""
    return m * CAP_DISP


def _static(v, f, col, name="", group="static"):
    """A non-spinning mesh (structure, tanks, buildings)."""
    return Mesh(v, f, col, name=name, spin=0.0, group=group)


def remap_yz(v):
    """Rotate a Z-axis primitive so its long axis points UP (+Y): (x,y,z)->(x,z,y)."""
    return [(p[0], p[2], p[1]) for p in v]


def _polar_box(r, a, zc, sx, sy, sz, col, group, name="", chamber=None):
    v, f = _box(r, 0.0, zc, sx, sy, sz)
    v = np.asarray(v, float) @ rot_z(a).T
    m = Mesh(v, f, col, name=name, group=group)
    m.chamber_index = chamber
    return m


# ---------------------------------------------------------------------------
#  CAPTURE UNIT  --  a single air contactor + sorbent bed in mechanical detail.
#  Coord frame:  +Z = airflow direction, +Y = up, +X = width.
# ---------------------------------------------------------------------------

def build_capture_parts(_n=1):
    """A single air contactor with fans, sorbent bed, regeneration chamber,
    CO2 collection manifold, vacuum pump, and valve system in detail."""
    parts = []
    cw = _cap(DIMS["contactor_w_m"])
    ch = _cap(DIMS["contactor_h_m"])
    cd = _cap(DIMS["contactor_d_m"])
    n_fans = DIMS["contactor_fans"]
    fan_d = _cap(DIMS["fan_d_m"])
    fan_r = fan_d / 2
    fan_hub_r = _cap(DIMS["fan_hub_d_m"]) / 2
    n_blades = DIMS["fan_blades"]
    sd = _cap(DIMS["sorbent_filter_d_m"])
    fd = _cap(DIMS["fan_d_m"])

    # --- Air contactor frame (the box structure holding fans + sorbent) ----
    frame_t = _cap(DIMS["contactor_frame_d_m"])
    frame_meshes = []
    # foundation pad (concrete base for visual grounding)
    pad_w = cw + frame_t * 4
    pad_d = cd + frame_t * 4 + _cap(2.0)
    v, f = _box(0, -ch/2 - frame_t - _cap(0.5), cd/2 + _cap(0.5),
                pad_w, _cap(1.0), pad_d)
    frame_meshes.append(_static(v, f, _mix(C_CONTACTOR, (0,0,0), 0.5), "foundation pad"))
    # top and bottom beams
    for y_pos in (-ch/2 - frame_t/2, ch/2 + frame_t/2):
        v, f = _box(0, y_pos, cd/2, cw + frame_t*2, frame_t, cd + frame_t*2)
        frame_meshes.append(_static(v, f, C_CONTACTOR, "frame beam"))
    # side beams
    for x_pos in (-cw/2 - frame_t/2, cw/2 + frame_t/2):
        v, f = _box(x_pos, 0, cd/2, frame_t, ch + frame_t*2, cd + frame_t*2)
        frame_meshes.append(_static(v, f, C_CONTACTOR, "frame column"))
    # back panel (behind sorbent)
    v, f = _box(0, 0, cd + frame_t/2, cw, ch, frame_t)
    frame_meshes.append(_static(v, f, _mix(C_CONTACTOR, (0,0,0), 0.3), "back panel"))
    parts.append(Part("frame", "CONTACTOR FRAME -- 20m x 12m x 4m", frame_meshes,
                      ["BLUPRINT: %.0f m W x %.0f m H x %.0f m D (cross-flow slab)" % (
                          DIMS["contactor_w_m"], DIMS["contactor_h_m"], DIMS["contactor_d_m"]),
                       "SCALE: 1 unit = 8 m (capture view)",
                       "MATERIAL: %s" % MATERIALS["frame"]["material"],
                       "GRADE: %s" % MATERIALS["frame"]["grade"],
                       "FAB: %s" % MATERIALS["frame"]["fabrication"],
                       "FOUNDATION: %s, %s" % (MATERIALS["foundation"]["material"],
                                                 MATERIALS["foundation"]["grade"]),
                       "COLUMNS: %s, %s" % (MATERIALS["columns"]["material"],
                                              MATERIALS["columns"]["grade"]),
                       "WALKWAY: %s" % MATERIALS["walkway"]["material"],
                       "LADDER: %s" % MATERIALS["ladder"]["material"],
                       "FASTENERS: %s" % MATERIALS["fasteners"]["material"],
                       "holds %d fans in a 4x2 grid + sorbent bed" % n_fans,
                       "wind load rated to 200 km/h, seismic zone 2",
                       "based on cooling-tower geometry (CE / Climeworks design)",
                       "COST: $%.0fK (frame+found+cols+walk+ladder+fasteners)" % (
                           MATERIALS["frame"]["est_cost_usd"] / 1000 +
                           MATERIALS["foundation"]["est_cost_usd"] / 1000 +
                           MATERIALS["columns"]["est_cost_usd"] / 1000 +
                           MATERIALS["walkway"]["est_cost_usd"] / 1000 +
                           MATERIALS["ladder"]["est_cost_usd"] / 1000 +
                           MATERIALS["fasteners"]["est_cost_usd"] / 1000)],
                      0, (0, 0, 0), C_CONTACTOR))

    # --- Structural support columns (4 corners) -----------------------------
    col_r = _cap(0.4)
    col_h = ch / 2 + frame_t
    for cx_pos in (-cw/2 + col_r, cw/2 - col_r):
        for cz_pos in (-cd/2 + col_r, cd/2 - col_r):
            v, f = _solid_cylinder(col_r, -col_h, 0, seg=10)
            col_mesh = _static(v, f, _mix(C_CONTACTOR, (0,0,0), 0.2), "support column")
            col_mesh.pivot = np.array([cx_pos, 0, cz_pos])
            col_mesh.tilt = (0, 0)
            frame_meshes.append(col_mesh)

    # --- Maintenance walkway (catwalk along top of frame) -------------------
    walk_w = _cap(1.0)
    walk_z = cd / 2
    v, f = _box(0, ch/2 + frame_t + walk_w/2, walk_z, cw + frame_t*2, walk_w, cd + frame_t*2)
    frame_meshes.append(_static(v, f, _mix(C_FAN_HUB, (0,0,0), 0.3), "walkway"))
    # walkway railing (thin posts along edges)
    for rx_pos in (-cw/2 - frame_t, cw/2 + frame_t):
        v, f = _box(rx_pos, ch/2 + frame_t + walk_w + _cap(0.1), walk_z,
                    _cap(0.08), _cap(0.8), cd + frame_t*2)
        frame_meshes.append(_static(v, f, C_WARN, "railing"))

    # --- Access ladder (side of frame) --------------------------------------
    ladder_x = cw/2 + frame_t + _cap(0.15)
    for rung_i in range(8):
        ry = -ch/2 + (rung_i + 0.5) * ch / 8
        v, f = _box(ladder_x, ry, cd/2, _cap(0.5), _cap(0.1), _cap(0.1))
        frame_meshes.append(_static(v, f, C_WARN, "ladder rung"))
    # ladder rails
    for rail_off in (_cap(0.25), -_cap(0.25)):
        v, f = _box(ladder_x + rail_off, 0, cd/2, _cap(0.06), ch, _cap(0.06))
        frame_meshes.append(_static(v, f, C_WARN, "ladder rail"))

    # --- Fan array (8 fans: 4 wide x 2 tall) --------------------------------
    fan_meshes_all = []
    fan_positions = []
    fx_spacing = cw / 4
    fy_spacing = ch / 2
    for i in range(n_fans):
        col_i = i % 4
        row_i = i // 4
        fx = (col_i - 1.5) * fx_spacing
        fy = (row_i - 0.5) * fy_spacing
        fz = -cd/2 - _cap(0.3)
        fan_positions.append((fx, fy, fz))
        # fan shroud (ring)
        v, f = _annulus_cylinder(fan_r * 1.15, fan_r * 0.92, -_cap(0.2), _cap(0.2), seg=20)
        shroud = Mesh(v, f, C_FAN_HUB, name="fan shroud %d" % (i+1), group="fan",
                       spin=0.0)
        shroud.pivot = np.array([fx, fy, fz])
        shroud.tilt = (0, math.pi/2)
        fan_meshes_all.append(shroud)
        # fan blades (spinning)
        blades = []
        # hub
        v, f = _solid_cylinder(fan_hub_r, -_cap(0.08), _cap(0.08), seg=8)
        blades.append(Mesh(v, f, C_FAN_HUB, name="fan hub %d" % (i+1), group="fan"))
        for b in range(n_blades):
            a = b * 2 * math.pi / n_blades
            bv, bf = _box(fan_r * 0.7, 0.0, _cap(0.04), fan_r * 1.3, _cap(0.12), _cap(0.06))
            bv = np.asarray(bv, float) @ rot_z(a).T
            blades.append(Mesh(bv, bf, C_FAN, name="fan blade", group="fan"))
        spinner = _place_spinner(blades, (fx, fy, fz), (0, math.pi/2), "fan")
        fan_meshes_all += spinner
    parts.append(Part("fans", "FAN ARRAY -- 8x 3.5m dia", fan_meshes_all,
                      ["BLUPRINT: %d axial fans, %.0f m diameter each" % (n_fans, DIMS["fan_d_m"]),
                       "SCALE: fan diameter = %.1f display units" % _cap(DIMS["fan_d_m"]),
                       "MATERIAL: %s" % MATERIALS["fans"]["material"],
                       "GRADE: %s" % MATERIALS["fans"]["grade"],
                       "FAB: %s" % MATERIALS["fans"]["fabrication"],
                       "MOTOR: %s" % MATERIALS["fan_motor"]["material"],
                       "MOTOR GRADE: %s" % MATERIALS["fan_motor"]["grade"],
                       "SHROUD: SS 304, spun + welded, 1.15x fan dia",
                       "%d CFRP blades per fan, Ti-6Al-4V hub, direct-drive" % n_blades,
                       "4-wide x 2-tall grid, spacing: %.1f m x %.1f m" % (
                           DIMS["contactor_w_m"]/4, DIMS["contactor_h_m"]/2),
                       "draws ~500,000 m3/h of air through the sorbent",
                       "total: %.0f million m3/h across all 80 contactors" % (
                           80 * 500000 / 1e6),
                       "MTBF %.0f h (contactless magnetic bearings)" %
                           COMPONENTS["fans"]["mtbf_h"],
                       "COST: $%.0fK (fans+motors)" % (
                           MATERIALS["fans"]["est_cost_usd"] / 1000 +
                           MATERIALS["fan_motor"]["est_cost_usd"] / 1000)],
                      1, (0, 0, -1.2), C_FAN))
    parts[-1].fire_anchor = np.array([0.0, 0.0, -cd/2 - _cap(0.5)])

    # --- Sorbent bed (the CO2-capturing matrix) -----------------------------
    bed_meshes = []
    bw = _cap(DIMS["bed_w_m"])
    bh = _cap(DIMS["bed_h_m"])
    bdd = _cap(DIMS["bed_d_m"])
    # bed frame
    v, f = _box(0, 0, cd/2 - bdd/2, bw, bh, bdd)
    bed_meshes.append(Mesh(v, f, C_SORBENT, name="sorbent bed", group="sorbent",
                           sorbent_state="capture"))
    # honeycomb channel layers (visible detail)
    n_layers = DIMS["sorbent_layers"]
    for li in range(n_layers):
        t = (li + 0.5) / n_layers
        zc = cd/2 - bdd + t * bdd
        # horizontal channel slats (fewer, thicker for visibility)
        for si in range(4):
            yc = -bh/2 + (si + 0.5) * bh / 4
            v, f = _box(0, yc, zc, bw * 0.92, _cap(0.3), _cap(0.12))
            bed_meshes.append(Mesh(v, f, _mix(C_SORBENT, (0,0,0), 0.15),
                                   name="sorbent channel", group="sorbent",
                                   sorbent_state="capture"))
    parts.append(Part("sorbent", "CO2 FILTER -- 20m x 10m x 1.5m", bed_meshes,
                      ["BLUPRINT: %.0f m W x %.0f m H x %.1f m D honeycomb monolith" % (
                          DIMS["bed_w_m"], DIMS["bed_h_m"], DIMS["bed_d_m"]),
                       "SCALE: bed volume = %.0f m3" % (
                           DIMS["bed_w_m"] * DIMS["bed_h_m"] * DIMS["bed_d_m"]),
                       "MATERIAL: %s" % MATERIALS["sorbent_bed"]["material"],
                       "GRADE: %s" % MATERIALS["sorbent_bed"]["grade"],
                       "FAB: %s" % MATERIALS["sorbent_bed"]["fabrication"],
                       "CASSETTE: %s" % MATERIALS["bed_frame"]["material"],
                       "CASSETTE GRADE: %s" % MATERIALS["bed_frame"]["grade"],
                       "%.0f t sorbent (PEI on silica/MOF), %.3f kg CO2/kg" % (
                           DIMS["sorbent_t_per_bed"], DIMS["sorbent_cap_kg_kg"]),
                       "%.1f t CO2 captured per cycle (2 mmol/g capacity)" % (
                           DIMS["sorbent_t_per_bed"] * DIMS["sorbent_cap_kg_kg"]),
                       "%d honeycomb layers, 1 mm channels, 450 CPSI" % n_layers,
                       "cyclic: capture -> regenerate -> cool",
                       "design life %d years, antioxidant + thermal stabilized" %
                           COMPONENTS["sorbent"]["design_life_years"],
                       "COST: $%.0fK (sorbent+cassette)" % (
                           MATERIALS["sorbent_bed"]["est_cost_usd"] / 1000 +
                           MATERIALS["bed_frame"]["est_cost_usd"] / 1000)],
                      2, (0, 0, 0.8), C_SORBENT))

    # --- Regeneration chamber (heats the sorbent to release CO2) -----------
    regen_meshes = []
    rw = _cap(DIMS["regen_w_m"])
    rh = _cap(DIMS["regen_h_m"])
    rd = _cap(DIMS["regen_d_m"])
    rz = cd/2 + rd/2 + _cap(1.0)
    # chamber body
    v, f = _box(0, 0, rz, rw, rh, rd)
    regen_meshes.append(_static(v, f, C_REGEN, "regen chamber", group="regen"))
    # insulation layer (slightly larger to avoid z-fighting with chamber)
    insul = _cap(DIMS["regen_insul_d_m"])
    v, f = _box(0, 0, rz, rw + insul*2 + _cap(0.05), rh + insul*2 + _cap(0.05), rd + insul*2 + _cap(0.05))
    regen_meshes.append(_static(v, f, _mix(C_REGEN, (0,0,0), 0.3), "insulation"))
    # heating element rows (visible as red-hot bars inside)
    n_heaters = DIMS["regen_heater_rows"]
    for hi in range(n_heaters):
        yc = -rh/2 + (hi + 0.5) * rh / n_heaters
        v, f = _box(0, yc, rz, rw * 0.85, _cap(0.25), _cap(0.25))
        regen_meshes.append(Mesh(v, f, C_REGEN_HOT, name="heater %d" % (hi+1),
                                 group="regen", hot=True))
    # connecting duct from sorbent to regen chamber
    regen_meshes.append(_pipe((0, 0, cd/2), (0, 0, rz - rd/2),
                              _cap(0.8), _mix(C_REGEN, C_SORBENT, 0.3), seg=12))
    parts.append(Part("regen", "REGEN CHAMBER -- 12m x 8m x 10m", regen_meshes,
                      ["BLUPRINT: %.0f m W x %.0f m H x %.0f m D vacuum chamber" % (
                          DIMS["regen_w_m"], DIMS["regen_h_m"], DIMS["regen_d_m"]),
                       "SCALE: chamber volume = %.0f m3" % (
                           DIMS["regen_w_m"] * DIMS["regen_h_m"] * DIMS["regen_d_m"]),
                       "MATERIAL: %s" % MATERIALS["regen_chamber"]["material"],
                       "GRADE: %s" % MATERIALS["regen_chamber"]["grade"],
                       "FAB: %s" % MATERIALS["regen_chamber"]["fabrication"],
                       "INSULATION: %s" % MATERIALS["insulation"]["material"],
                       "INSULATION GRADE: %s" % MATERIALS["insulation"]["grade"],
                       "HEATERS: %s" % MATERIALS["heaters"]["material"],
                       "heats sorbent to %.0f C under vacuum (~0.1 bar), SS 316L" % DIMS["regen_temp_c"],
                       "%d ceramic IR heater rows, %.1f m insulation" % (
                           n_heaters, DIMS["regen_insul_d_m"]),
                       "%.0f kWh/t thermal, solar thermal + geothermal" % ENERGY["regen_thermal_kwh_t"],
                       "MTBF %.0f h, design life %d years" % (
                           COMPONENTS["regen_units"]["mtbf_h"],
                           COMPONENTS["regen_units"]["design_life_years"]),
                       "COST: $%.0fK (chamber+insul+heaters)" % (
                           MATERIALS["regen_chamber"]["est_cost_usd"] / 1000 +
                           MATERIALS["insulation"]["est_cost_usd"] / 1000 +
                           MATERIALS["heaters"]["est_cost_usd"] / 1000)],
                      3, (0, 0, 1.5), C_REGEN))

    # --- Vacuum pump (extracts CO2 from regen chamber) ----------------------
    vac_meshes = []
    vd = _cap(DIMS["regen_vacuum_d_m"]) / 2
    vh = _cap(DIMS["regen_vacuum_h_m"])
    v, f = _solid_cylinder(vd, 0, vh, seg=12)
    vac_meshes.append(Mesh(v, f, C_COMP, name="vacuum pump", group="vacuum"))
    vac_spinner = _place_spinner(vac_meshes, (rw/2 + _cap(1.0), 0, rz), (0, math.pi/2), "vacuum")
    # vacuum pipe to regen
    vac_pipe = _pipe((rw/2 + _cap(0.5), 0, rz), (rw/2 + _cap(1.0), 0, rz),
                     _cap(0.3), C_COMP, seg=8)
    parts.append(Part("vacuum", "VACUUM PUMP -- 1.2m dia x 2.5m", vac_spinner + [vac_pipe],
                      ["BLUPRINT: %.1f m dia x %.1f m H dry screw pump" % (
                          DIMS["regen_vacuum_d_m"], DIMS["regen_vacuum_h_m"]),
                       "SCALE: pump volume = %.1f m3" % (
                           math.pi * (DIMS["regen_vacuum_d_m"]/2)**2 * DIMS["regen_vacuum_h_m"]),
                       "MATERIAL: %s" % MATERIALS["vacuum_pump"]["material"],
                       "GRADE: %s" % MATERIALS["vacuum_pump"]["grade"],
                       "FAB: %s" % MATERIALS["vacuum_pump"]["fabrication"],
                       "SEALS: PTFE lip seals, hermetic, oil-free",
                       "SS 316L body, PTFE seals, %.0f kWh/t electrical" % ENERGY["vacuum_elec_kwh_t"],
                       "creates ~0.1 bar vacuum for CO2 release",
                       "dry screw, hermetic, magnetic coupling (no oil), life %d yr" %
                           COMPONENTS["vacuum_pumps"]["design_life_years"],
                       "MTBF %.0f h, %.0f%% prevented by thermal monitoring" % (
                           COMPONENTS["vacuum_pumps"]["mtbf_h"],
                           COMPONENTS["vacuum_pumps"]["predictive_factor"] * 100),
                       "COST: $%.0fK" % (MATERIALS["vacuum_pump"]["est_cost_usd"] / 1000)],
                      4, (1.2, 0, 1.5), C_COMP))

    # --- CO2 collection manifold (regen -> compressor) ----------------------
    manifold_meshes = []
    md = _cap(DIMS["manifold_d_m"])
    n_rows = DIMS["manifold_rows"]
    for mi in range(n_rows):
        yc = -rh/2 + (mi + 0.5) * rh / n_rows
        manifold_meshes.append(_pipe((-rw/2 - _cap(0.5), yc, rz),
                                      (rw/2 + _cap(0.5), yc, rz),
                                      md * 0.5, C_CO2TANK, seg=8))
    # main collection pipe
    manifold_meshes.append(_pipe((0, 0, rz + rd/2 + _cap(0.5)),
                                  (0, 0, rz + rd/2 + _cap(2.0)),
                                  md, C_CO2BAND, seg=10))
    parts.append(Part("manifold", "CO2 MANIFOLD -- 4 rows, 800mm dia", manifold_meshes,
                      ["BLUPRINT: %d collection pipe rows, %.0f mm dia" % (
                          n_rows, DIMS["manifold_d_m"] * 1000),
                       "SCALE: pipe span = %.0f m across chamber" % DIMS["regen_w_m"],
                       "MATERIAL: %s" % MATERIALS["manifold"]["material"],
                       "GRADE: %s" % MATERIALS["manifold"]["grade"],
                       "FAB: %s" % MATERIALS["manifold"]["fabrication"],
                       "SS 316L, welded joints, trace-heated",
                       "gathers released CO2 from regeneration chamber",
                       "feeds to the CO2 compressor train",
                       "COST: $%.0fK" % (MATERIALS["manifold"]["est_cost_usd"] / 1000)],
                      5, (0, 0, 2.0), C_CO2BAND))

    # --- Valve system (cycles between capture and regeneration) ------------
    valve_meshes = []
    # 3 main valves: intake, exhaust, CO2 output
    for vi, (vx, vy, vz, vname) in enumerate([
        (0, 0, -cd/2 - _cap(0.2), "intake valve"),
        (0, ch/2 + _cap(0.5), cd/2, "air exhaust valve"),
        (0, 0, rz + rd/2 + _cap(1.8), "CO2 output valve"),
    ]):
        v, f = _solid_cylinder(_cap(0.4), -_cap(0.15), _cap(0.15), seg=8)
        vm = Mesh(v, f, C_FAN_HUB, name=vname, group="valve", spin=0.0)
        vm.pivot = np.array([vx, vy, vz])
        vm.tilt = (math.pi/2, 0)
        valve_meshes.append(vm)
        # valve handle
        v, f = _box(vx, vy, vz, _cap(0.8), _cap(0.2), _cap(0.2))
        valve_meshes.append(_static(v, f, C_WARN, "valve handle"))
    parts.append(Part("valves", "VALVE SYSTEM -- 3x 0.8m pneumatic", valve_meshes,
                      ["BLUPRINT: 3 cycling valves, 0.8 m dia, pneumatic actuated",
                       "SCALE: positioned at intake, exhaust, and CO2 output",
                       "MATERIAL: %s" % MATERIALS["valves"]["material"],
                       "GRADE: %s" % MATERIALS["valves"]["grade"],
                       "FAB: %s" % MATERIALS["valves"]["fabrication"],
                       "SEALS: PTFE seats, spring-return pneumatic actuator",
                       "auto-switches between capture and regeneration phases",
                       "capture: intake open, exhaust open, CO2 closed",
                       "regen: all sealed, vacuum + heater on",
                       "fail-safe design (spring-return on air loss)",
                       "COST: $%.0fK" % (MATERIALS["valves"]["est_cost_usd"] / 1000)],
                      6, (0, 1.0, 0.5), C_WARN))

    # --- Air intake plenum (behind the fan wall) ---------------------------
    plenum_meshes = []
    pd = _cap(DIMS["plenum_d_m"])
    v, f = _box(0, 0, -cd/2 - pd/2 - _cap(0.3), cw, ch, pd)
    plenum_meshes.append(_static(v, f, _mix(C_CONTACTOR, (0,0,0), 0.4), "intake plenum"))
    # plenum frame
    for xp in (-cw/2, cw/2):
        v, f = _box(xp, 0, -cd/2 - pd/2 - _cap(0.3), _cap(0.2), ch, pd)
        plenum_meshes.append(_static(v, f, C_CONTACTOR, "plenum frame"))
    parts.append(Part("plenum", "INTAKE PLENUM -- 20m x 12m x 3m", plenum_meshes,
                      ["BLUPRINT: %.0f m W x %.0f m H x %.0f m D plenum" % (
                          DIMS["contactor_w_m"], DIMS["contactor_h_m"], DIMS["plenum_d_m"]),
                       "SCALE: plenum volume = %.0f m3" % (
                           DIMS["contactor_w_m"] * DIMS["contactor_h_m"] * DIMS["plenum_d_m"]),
                       "MATERIAL: %s" % MATERIALS["plenum"]["material"],
                       "GRADE: %s" % MATERIALS["plenum"]["grade"],
                       "FAB: %s" % MATERIALS["plenum"]["fabrication"],
                       "distributes airflow evenly across sorbent bed",
                       "reduces turbulence, improves capture efficiency",
                       "SS 304 panels, gasket-sealed, quick-release fasteners",
                       "COST: $%.0fK" % (MATERIALS["plenum"]["est_cost_usd"] / 1000)],
                      7, (0, 0, -0.8), C_CONTACTOR))

    # --- CO2 output pipe (manifold to compressor) --------------------------
    output_meshes = []
    output_meshes.append(_pipe((0, 0, rz + rd/2 + _cap(2.0)),
                                (0, 0, rz + rd/2 + _cap(5.0)),
                                _cap(0.4), C_CO2BAND, seg=8))
    parts.append(Part("output", "CO2 OUTPUT PIPE -- 400mm dia", output_meshes,
                      ["BLUPRINT: %.0f mm dia SS 316L pipe, trace-heated" % (
                          DIMS["manifold_d_m"] * 500),
                       "SCALE: ~5 m run from manifold to compressor train",
                       "MATERIAL: %s" % MATERIALS["output_pipe"]["material"],
                       "GRADE: %s" % MATERIALS["output_pipe"]["grade"],
                       "FAB: %s" % MATERIALS["output_pipe"]["fabrication"],
                       "connects manifold to compressor train",
                       "carries concentrated CO2 at ~1 bar, 100 C",
                       "COST: $%.0fK" % (MATERIALS["output_pipe"]["est_cost_usd"] / 1000)],
                      8, (0, 0, 2.5), C_CO2BAND))

    return parts


# ---------------------------------------------------------------------------
#  WHOLE FACILITY  --  the full-scale DAC plant.
#  Coord frame:  +Z = north, -Z = south,  +X = east,  +Y = up.
#  Ground at Y = 0.
# ---------------------------------------------------------------------------

def build_facility_parts(_n=1):
    parts = []

    # --- LAYOUT PLAN -------------------------------------------------------
    # The facility is organized as a north-south process flow:
    #   SOUTH:  Control building, battery building (entrance/admin)
    #   CENTER: Air contactor array (the main capture field)
    #   NORTH:  Regen units -> compressors -> CO2 storage -> pipeline
    #           (processing corridor, CO2 flows northward)
    #   EAST:   Solar PV field
    #   WEST:   Solar thermal field + molten salt tanks
    #   PERIM:  Wind turbines, geothermal wells (near regen for heat)

    # --- GROUND TERRAIN ----------------------------------------------------
    gw = fcs(math.sqrt(FACILITY["land_area_m2"]) / 2.0)
    gv, gf = _box(0, -fcs(0.5), 0, gw * 2, fcs(1.0), gw * 2)
    parts.append(Part("ground", "SITE TERRAIN (%.0f ha)" % (FACILITY["land_area_m2"]/1e4),
                      [_static(gv, gf, C_GROUND, "ground")],
                      ["The land the plant sits on -- about %.0f football fields" % (FACILITY["land_area_m2"]/1e4 * 2.5),
                       "%.0f hectare site (%.1f km2), graded level to +-5 cm" % (FACILITY["land_area_m2"]/1e4, FACILITY["land_area_m2"]/1e6),
                       "soil bearing capacity >500 kPa, seismic zone 2",
                       "layout: N-S process flow (capture -> regen -> compress -> store)",
                       "solar PV east, solar thermal + salt tanks west",
                       "wind turbines on north/south edges, geo wells near regen",
                       "foundation: reinforced M60 concrete, 15 m depth"],
                      0, (0, -0.5, 0), C_GROUND))

    # --- ACCESS ROADS (connecting process stages) --------------------------
    road_meshes = []
    road_col = _mix(C_GROUND, (80, 80, 80), 0.4)
    # main north-south road: from control building (south) through processing corridor (north)
    rv, rf = _box(0, -fcs(0.1), fcs(100.0), fcs(8.0), fcs(0.2), fcs(500.0))
    road_meshes.append(_static(rv, rf, road_col, "main road"))
    # east-west cross road at processing corridor (connects geo wells to compressors)
    rv, rf = _box(0, -fcs(0.1), fcs(220.0), fcs(400.0), fcs(0.2), fcs(8.0))
    road_meshes.append(_static(rv, rf, road_col, "cross road"))
    # access road to control/battery buildings (south)
    rv, rf = _box(fcs(40.0), -fcs(0.1), fcs(-220.0), fcs(120.0), fcs(0.2), fcs(8.0))
    road_meshes.append(_static(rv, rf, road_col, "south access road"))
    parts.append(Part("roads", "ACCESS ROADS", road_meshes,
                      ["Roads for trucks and maintenance vehicles",
                       "graded gravel roads connecting all process areas",
                       "main north-south corridor: 8 m wide, 500 m long",
                       "east-west cross road at processing corridor",
                       "designed for heavy vehicle access (50 t loads)"],
                      0, (0, -0.3, 0), road_col))

    # --- SITE PERIMETER FENCING (rectangular, matching terrain) ------------
    fence_poles = []
    fence_mid = []
    fence_top = []
    fence_col = _mix(C_GROUND, (100, 100, 100), 0.3)
    fence_hw = fcs(707.0)   # half width (east-west)
    fence_hl = fcs(707.0)   # half length (north-south)
    n_posts_per_side = 10
    # Build fence posts along 4 sides of the rectangle
    fence_pts = []
    for side in range(4):
        for i in range(n_posts_per_side):
            t = i / n_posts_per_side
            if side == 0:    # north side (z = +707)
                fx = -fence_hw + t * 2 * fence_hw
                fz = fence_hl
            elif side == 1:  # east side (x = +707)
                fx = fence_hw
                fz = fence_hl - t * 2 * fence_hl
            elif side == 2:  # south side (z = -707)
                fx = fence_hw - t * 2 * fence_hw
                fz = -fence_hl
            else:            # west side (x = -707)
                fx = -fence_hw
                fz = -fence_hl + t * 2 * fence_hl
            fence_pts.append((fx, fz))
    # Close the loop by repeating the first point
    fence_pts.append(fence_pts[0])
    for i in range(len(fence_pts)):
        fx, fz = fence_pts[i]
        # pole
        v, f = _solid_cylinder(fcs(0.1), 0, fcs(3.0), seg=6)
        pm = _static(v, f, fence_col, "fence pole")
        pm.pivot = np.array([fx, 0, fz])
        fence_poles.append(pm)
        if i < len(fence_pts) - 1:
            fx2, fz2 = fence_pts[i + 1]
            # mid rail
            fence_mid.append(_pipe((fx, fcs(1.5), fz), (fx2, fcs(1.5), fz2),
                                      fcs(0.03), fence_col, seg=4))
            # top rail
            fence_top.append(_pipe((fx, fcs(3.0), fz), (fx2, fcs(3.0), fz2),
                                      fcs(0.03), fence_col, seg=4))
    fence_meshes = [_merge_static_meshes(fence_poles, fence_col, "fence poles"),
                    _merge_static_meshes(fence_mid, fence_col, "fence mid"),
                    _merge_static_meshes(fence_top, fence_col, "fence top")]
    parts.append(Part("fence", "SITE PERIMETER FENCING", fence_meshes,
                      ["Security fence around the entire plant",
                       "chain-link fence on 3m steel posts",
                       "perimeter: ~5.7 km, 40 posts at ~140m spacing",
                       "gates at south entrance (admin/vehicle access)",
                       "wildlife exclusion, security lighting at night"],
                      0, (0, -0.2, 0), fence_col))

    # --- AIR CONTACTOR ARRAY (80 units in 8 rows x 10 per row) -------------
    # Ultra-simplified for facility overview: 1 box + 4 fan circles each.
    # Full detail is in the capture view.
    contactor_boxes = []
    contactor_fans = []
    cw = fcs(DIMS["contactor_w_m"])
    ch = fcs(DIMS["contactor_h_m"])
    cd = fcs(DIMS["contactor_d_m"])
    fan_r = fcs(DIMS["fan_d_m"]) / 2
    n_rows = DIMS["contactor_rows"]
    per_row = DIMS["contactor_per_row"]
    row_spacing = fcs(32.0)
    col_spacing = fcs(32.0)
    total_w = per_row * col_spacing
    total_d = n_rows * row_spacing
    x0 = -total_w / 2 + col_spacing / 2
    z0 = -total_d / 2 + row_spacing / 2 - fcs(50.0)
    for row in range(n_rows):
        # Staggered offset: alternate rows shifted by half col spacing for better airflow
        row_offset = (col_spacing / 2) if (row % 2) else 0.0
        for col in range(per_row):
            cx = x0 + col * col_spacing + row_offset
            cz = z0 + row * row_spacing
            # single box body
            v, f = _box(cx, ch * 0.5, cz, cw, ch, cd)
            contactor_boxes.append(_static(v, f, C_CONTACTOR, ""))
            # 4 fan circles on front face (simple low-poly disc)
            fx_sp = cw / 4
            for fi in range(4):
                fx = cx + (fi - 1.5) * fx_sp
                fy = ch * 0.5
                fz = cz - cd/2 - fcs(0.05)
                v, f = _box(fx, fy, fz, fan_r * 1.8, fan_r * 1.8, fcs(0.04))
                contactor_fans.append(_static(v, f, C_FAN_HUB, ""))
    contactor_meshes = [_merge_static_meshes(contactor_boxes, C_CONTACTOR, "contactor boxes"),
                        _merge_static_meshes(contactor_fans, C_FAN_HUB, "contactor fans")]
    parts.append(Part("contactors", "AIR CAPTURE FANS (80 units)", contactor_meshes,
                      ["Giant fans pull outside air through CO2-filtering material",
                       "%d units in %d rows x %d per row" % (
                           FACILITY["air_contactors"], n_rows, per_row),
                       "each: %.0f m W x %.0f m H x %.0f m D" % (
                           DIMS["contactor_w_m"], DIMS["contactor_h_m"], DIMS["contactor_d_m"]),
                       "%d fans per unit, %.0f m dia (CFRP blades, Ti hub)" % (
                           DIMS["contactor_fans"], DIMS["fan_d_m"]),
                       "total air flow: ~50 million m3/h across array",
                       "staggered row layout for optimized airflow (reduced wake interference)",
                       "sorbent: %.0f t/bed, %.3f kg CO2/kg (2 mmol/g)" % (
                           DIMS["sorbent_t_per_bed"], DIMS["sorbent_cap_kg_kg"]),
                       "frame: galvanized steel, 0.3 m structural sections",
                       "fan MTBF: %.0f h (contactless mag bearings), life %d yr" % (
                           COMPONENTS["fans"]["mtbf_h"], COMPONENTS["fans"]["design_life_years"]),
                       "%.1f%% failures prevented by predictive maintenance" % (
                           COMPONENTS["fans"]["predictive_factor"] * 100)],
                      1, (0, 0.5, 0), C_CONTACTOR))

    # --- SOLAR PV FIELD ----------------------------------------------------
    # At facility scale (1:200), individual 2m panels are sub-pixel.
    # Use representative row strips (each strip = one row of ~45 panels merged).
    pv_panels_a = []  # C_PV color
    pv_panels_b = []  # C_PV_HI mix color
    pv_posts = []
    pv_w = fcs(DIMS["pv_field_w_m"])
    pv_l = fcs(DIMS["pv_field_l_m"])
    pitch = math.radians(DIMS["pv_panel_pitch_deg"])
    n_pv_rows = 24  # visual rows (representative)
    row_sp = pv_l / n_pv_rows
    pv_x = fcs(430.0)  # offset east (within 707m half-width terrain)
    col_b = _mix(C_PV, C_PV_HI, 0.25)
    strip_w = pv_w * 0.92  # each strip spans most of the field width
    strip_d = row_sp * 0.7  # strip depth (row thickness)
    for ri in range(n_pv_rows):
        rz = -pv_l/2 + (ri + 0.5) * row_sp
        rx = pv_x
        # tilted panel strip (represents a full row of panels)
        v, f = _box(rx, fcs(2.0), rz, strip_w, fcs(0.3), strip_d)
        v = np.asarray(v, float) @ rot_x(pitch).T
        if ri % 2 == 0:
            pv_panels_a.append(_static(v, f, C_PV, ""))
        else:
            pv_panels_b.append(_static(v, f, col_b, ""))
        # support posts at intervals along the row
        for pi in range(5):
            px = rx - strip_w/2 + (pi + 0.5) * strip_w / 5
            v, f = _box(px, fcs(1.0), rz, fcs(0.3), fcs(2.0), fcs(0.3))
            pv_posts.append(_static(v, f, C_PV_FRAME, ""))
    pv_meshes = [_merge_static_meshes(pv_panels_a, C_PV, "pv panels a"),
                 _merge_static_meshes(pv_panels_b, col_b, "pv panels b"),
                 _merge_static_meshes(pv_posts, C_PV_FRAME, "pv posts")]
    parts.append(Part("solarpv", "SOLAR PANELS -- ELECTRICITY (84 MW)", pv_meshes,
                      ["Solar panels that make electricity to run the plant",
                       "%.0f m2 of panels, %.0f%% efficient" % (
                           DIMS["solar_pv_m2"], DIMS["solar_pv_eff"] * 100),
                       "peak: %.0f MW electrical, bifacial panels" % (SOLAR_PV_PEAK_KW / 1000),
                       "panel: %.1f m x %.1f m, %.0f deg tilt" % (
                           DIMS["pv_panel_w_m"], DIMS["pv_panel_h_m"], DIMS["pv_panel_pitch_deg"]),
                       "powers fans, compressors, vacuum pumps, controls",
                       "%.0f MWh Li-ion battery for night operation" % DIMS["battery_mwh"],
                       "degradation: %.1f%%/year, design life %d years" % (
                           COMPONENTS["solar_pv"]["degradation_rate"] * 100,
                           COMPONENTS["solar_pv"]["design_life_years"])],
                      2, (1.5, 0.8, 0), C_PV))

    # --- SOLAR THERMAL FIELD (parabolic troughs) ---------------------------
    # At facility scale, individual troughs (5.8m wide) are barely visible.
    # Use representative row strips spanning the field width.
    trough_a = []
    trough_b = []
    trough_pipes = []
    trough_supports = []
    tr_w = fcs(DIMS["trough_field_w_m"])
    tr_l = fcs(DIMS["trough_field_l_m"])
    n_tr_rows = 20  # visual rows
    tr_row_sp = tr_l / n_tr_rows
    tr_x = fcs(-430.0)  # offset west (within 707m half-width terrain)
    col_b = _mix(C_TROUGH, C_TROUGH_HI, 0.2)
    strip_w = tr_w * 0.90
    strip_d = tr_row_sp * 0.65
    for ri in range(n_tr_rows):
        rz = -tr_l/2 + (ri + 0.5) * tr_row_sp
        # curved trough strip (represents a row of ~80 troughs)
        v, f = _box(tr_x, fcs(2.5), rz, strip_w, fcs(1.0), strip_d)
        v = np.asarray(v, float)
        # curve the trough: bend vertices upward at edges
        for vi in range(len(v)):
            dx = v[vi][0] - tr_x
            v[vi] = (v[vi][0], v[vi][1] + abs(dx) * 0.15, v[vi][2])
        if ri % 2 == 0:
            trough_a.append(_static(v.tolist(), f, C_TROUGH, ""))
        else:
            trough_b.append(_static(v.tolist(), f, col_b, ""))
        # heat collection pipe at the focal line
        v, f = _solid_cylinder(fcs(0.3), -strip_w*0.45, strip_w*0.45, seg=10)
        pm = _static(v, f, C_TROUGH_PIPE, "heat pipe")
        pm.pivot = np.array([tr_x, fcs(3.5), rz])
        pm.tilt = (0, math.pi/2)
        trough_pipes.append(pm)
        # support structure
        v, f = _box(tr_x, fcs(1.5), rz, fcs(0.4), fcs(3.0), fcs(0.4))
        trough_supports.append(_static(v, f, C_PV_FRAME, ""))
    trough_meshes = [_merge_static_meshes(trough_a, C_TROUGH, "troughs a"),
                     _merge_static_meshes(trough_b, col_b, "troughs b"),
                     _merge_static_meshes(trough_pipes, C_TROUGH_PIPE, "heat pipes"),
                     _merge_static_meshes(trough_supports, C_PV_FRAME, "trough supports")]
    parts.append(Part("troughs", "SOLAR MIRRORS -- HEAT (238 MWth)", trough_meshes,
                      ["Curved mirrors that focus sunlight to make heat",
                       "%.0f m2 of mirrors" % DIMS["trough_aperture_m2"],
                       "peak: %.0f MW thermal, heats oil to 150 C" % (SOLAR_TH_PEAK_KW / 1000),
                       "trough: %.0f m aperture x %.0f m length" % (
                           DIMS["trough_aperture_w_m"], DIMS["trough_len_m"]),
                       "direct thermal -> regeneration (no electric conversion loss)",
                       "molten salt storage: %.0f MWh (hot %.0f C / cold %.0f C)" % (
                           DIMS["thermal_storage_mwh"],
                           DIMS["salt_hot_temp_c"], DIMS["salt_cold_temp_c"]),
                       "design life %d years, %.1f%%/year degradation" % (
                           COMPONENTS["solar_thermal"]["design_life_years"],
                           COMPONENTS["solar_thermal"]["degradation_rate"] * 100)],
                      3, (-1.5, 0.8, 0), C_TROUGH))

    # --- WIND TURBINES -----------------------------------------------------
    # Real 5MW turbines: 120m tower (tapered), 150m rotor, 3 CFRP blades
    wind_towers = []
    wind_nacelles = []
    wind_noses = []
    wind_spinners = []
    n_wt = DIMS["wind_turbines"]
    wt_h = fcs(DIMS["turbine_h_m"])
    wt_d = fcs(DIMS["turbine_d_m"])
    wt_td = fcs(DIMS["turbine_tower_d_m"])
    wt_td_top = fcs(DIMS["turbine_tower_top_d_m"])
    for i in range(n_wt):
        # Place turbines on north and south perimeters (avoiding solar fields)
        n_per_side = n_wt // 2
        side = i // n_per_side  # 0 = north, 1 = south
        idx = i % n_per_side
        wt_spread = fcs(600.0)
        if n_per_side > 1:
            wx = -wt_spread + idx * (2 * wt_spread / (n_per_side - 1))
        else:
            wx = fcs(0.0)
        wz = fcs(670.0) if side == 0 else fcs(-670.0)
        # tower: tapered from base (4m) to top (2m) using multiple lofted sections
        n_tower_segs = 4
        tower_sect = []
        for si in range(n_tower_segs + 1):
            t = si / n_tower_segs
            z = -wt_h/2 + t * wt_h
            d = wt_td/2 * (1.0 - t * (1.0 - wt_td_top/wt_td))
            tower_sect.append((z, d, -d, d))
        tv, tf = _hull(tower_sect)
        tm = _static(tv, tf, C_WIND_TOWER, "turbine tower %d" % (i+1))
        tm.pivot = np.array([wx, 0, wz])
        tm.tilt = (math.pi/2, 0)
        wind_towers.append(tm)
        # nacelle (streamlined box + nose cone)
        v, f = _box(0, 0, 0, fcs(3.0), fcs(2.5), fcs(8.0))
        nm = _static(v, f, C_WIND, "nacelle")
        nm.pivot = np.array([wx, wt_h, wz])
        wind_nacelles.append(nm)
        # nose cone (front of nacelle, facing wind)
        v, f = _solid_cylinder(fcs(1.2), 0, fcs(2.0), seg=8)
        nc = _static(v, f, _mix(C_WIND, (0,0,0), 0.15), "nose cone")
        nc.pivot = np.array([wx, wt_h, wz + fcs(4.0)])
        nc.tilt = (0, math.pi/2)
        wind_noses.append(nc)
        # rotor hub + blades (spinning)
        blades = []
        v, f = _solid_cylinder(fcs(0.8), -fcs(0.4), fcs(0.4), seg=8)
        blades.append(Mesh(v, f, C_WIND_TOWER, name="hub", group="windrotor"))
        for b in range(DIMS["turbine_blades"]):
            a = b * 2 * math.pi / DIMS["turbine_blades"]
            # tapered blade: wider at root, thinner at tip
            blade_sect = [
                (0, fcs(0.8), -fcs(0.15), fcs(0.15)),
                (wt_d * 0.2, fcs(0.6), -fcs(0.12), fcs(0.12)),
                (wt_d * 0.5, fcs(0.4), -fcs(0.08), fcs(0.08)),
                (wt_d * 0.48, fcs(0.15), -fcs(0.04), fcs(0.04)),
            ]
            bv, bf = _hull(blade_sect)
            bv = np.asarray(bv, float) @ rot_z(a).T
            blades.append(Mesh(bv, bf, C_WIND, name="blade", group="windrotor"))
        spinner = _place_spinner(blades, (wx, wt_h, wz + fcs(4.5)), (0, 0), "windrotor")
        wind_spinners += spinner
    wind_meshes = [_merge_static_meshes(wind_towers, C_WIND_TOWER, "turbine towers"),
                   _merge_static_meshes(wind_nacelles, C_WIND, "nacelles"),
                   _merge_static_meshes(wind_noses, _mix(C_WIND, (0,0,0), 0.15), "nose cones")] + wind_spinners
    parts.append(Part("wind", "WIND TURBINES -- ELECTRICITY (50 MW)", wind_meshes,
                      ["Wind turbines that make extra electricity",
                       "%d x %.0f MW turbines = %.0f MW total" % (
                          n_wt, DIMS["turbine_rated_mw"],
                          n_wt * DIMS["turbine_rated_mw"]),
                       "hub height %.0f m, rotor %.0f m dia, CFRP blades" % (
                           DIMS["turbine_h_m"], DIMS["turbine_d_m"]),
                       "cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s",
                       "supplements solar PV for electrical supply",
                       "MTBF %.0f h (direct-drive + mag bearings, no gearbox), life %d yr" % (
                           COMPONENTS["wind_turbines"]["mtbf_h"],
                           COMPONENTS["wind_turbines"]["design_life_years"])],
                      4, (0, 1.5, -1.0), C_WIND))

    # --- GEOTHERMAL WELLS --------------------------------------------------
    # Near regen units (west side of processing corridor) -- short thermal pipe runs
    geo_wellheads = []
    geo_pumps = []
    geo_steam = []
    n_gw = DIMS["geo_wells"]
    gw_h = fcs(DIMS["geo_well_h_m"])
    gw_d = fcs(DIMS["geo_well_d_m"])
    for i in range(n_gw):
        gx = fcs(-120.0) + (i % 4) * fcs(8.0)
        gz = fcs(180.0) + (i // 4) * fcs(12.0)
        # wellhead
        v, f = _solid_cylinder(gw_d, 0, gw_h, seg=8)
        wm = _static(v, f, C_GEO, "geothermal well %d" % (i+1))
        wm.pivot = np.array([gx, 0, gz])
        geo_wellheads.append(wm)
        # pump station
        v, f = _solid_cylinder(fcs(DIMS["geo_pump_d_m"])/2, 0,
                               fcs(DIMS["geo_pump_h_m"]), seg=8)
        pm = _static(v, f, _mix(C_GEO, C_COMP, 0.3), "geo pump")
        pm.pivot = np.array([gx + fcs(2.0), 0, gz])
        geo_pumps.append(pm)
        # steam vent hint
        v, f = _solid_cylinder(fcs(0.3), gw_h, gw_h + fcs(3.0), seg=8)
        sm = _static(v, f, C_GEO_STEAM, "steam")
        sm.pivot = np.array([gx, 0, gz])
        geo_steam.append(sm)
    geo_meshes = [_merge_static_meshes(geo_wellheads, C_GEO, "geo wellheads"),
                  _merge_static_meshes(geo_pumps, _mix(C_GEO, C_COMP, 0.3), "geo pumps"),
                  _merge_static_meshes(geo_steam, C_GEO_STEAM, "geo steam")]
    parts.append(Part("geo", "GEOTHERMAL WELLS -- UNDERGROUND HEAT (60 MWth)", geo_meshes,
                      ["Wells that pull heat from deep underground, 24/7",
                       "%d wells, %.0f MW thermal total, always-on" % (
                          n_gw, DIMS["geo_mw_thermal"]),
                       "wellhead: Ti-6Al-4V, pump: SS 316L",
                       "provides regeneration heat 24/7 baseload",
                       "supplements solar thermal, critical for night ops",
                       "MTBF %.0f h (hermetic pump), life %d yr" % (
                           COMPONENTS["geothermal"]["mtbf_h"],
                           COMPONENTS["geothermal"]["design_life_years"])],
                      5, (0, 0.5, 0.8), C_GEO))

    # --- REGENERATION UNITS (16 shared units) ------------------------------
    # Positioned north of contactor array in a processing corridor
    # CO2 flow: contactors (center) -> regen units (north) -> compressors (further north)
    regen_boxes = []
    regen_hots = []
    n_ru = FACILITY["regen_units"]
    rw = fcs(DIMS["regen_w_m"])
    rh = fcs(DIMS["regen_h_m"])
    rd = fcs(DIMS["regen_d_m"])
    for i in range(n_ru):
        rx = fcs(-37.5) + (i % 4) * fcs(25.0)
        rz = fcs(180.0) + (i // 4) * fcs(25.0)
        v, f = _box(rx, rh/2, rz, rw, rh, rd)
        regen_boxes.append(_static(v, f, C_REGEN, "regen unit %d" % (i+1)))
        # hot hint
        v, f = _box(rx, rh * 0.7, rz, rw * 0.6, fcs(0.3), rd * 0.6)
        regen_hots.append(_static(v, f, C_REGEN_HOT, ""))
    # interconnecting pipes from contactor array to regen corridor
    regen_pipes = []
    for pi in range(4):
        px = fcs(-37.5) + pi * fcs(25.0)
        regen_pipes.append(_pipe((px, fcs(3.0), fcs(120.0)), (px, fcs(3.0), fcs(180.0)),
                     fcs(0.3), C_COMP, seg=6))
    regen_meshes = [_merge_static_meshes(regen_boxes, C_REGEN, "regen boxes"),
                    _merge_static_meshes(regen_hots, C_REGEN_HOT, "regen hot"),
                    _merge_static_meshes(regen_pipes, C_COMP, "regen pipes")]
    parts.append(Part("regenunits", "CO2 RELEASE UNITS (16)", regen_meshes,
                      ["Heats the CO2-filled material to release captured CO2",
                       "%d shared heating chambers, SS 316L" % n_ru,
                       "each serves ~5 sorbent beds in rotation",
                       "heats to %.0f C under vacuum, releases CO2" % DIMS["regen_temp_c"],
                       "insulated (%.1f m), solar thermal + geothermal heated" % DIMS["regen_insul_d_m"],
                       "MTBF %.0f h, design life %d years" % (
                           COMPONENTS["regen_units"]["mtbf_h"],
                           COMPONENTS["regen_units"]["design_life_years"])],
                      6, (-1.0, 0.6, 0.3), C_REGEN))

    # --- CO2 COMPRESSOR TRAIN (4 units) ------------------------------------
    # North of regen units: CO2 flows regen -> compressors -> storage -> pipeline
    comp_meshes = []
    n_comp = FACILITY["compressors"]
    comp_w = fcs(DIMS["comp_total_w_m"])
    comp_h = fcs(DIMS["comp_total_h_m"])
    comp_d = fcs(DIMS["comp_total_d_m"])
    for i in range(n_comp):
        cx = fcs(-22.5) + (i - 1.5) * fcs(15.0)
        cz = fcs(310.0)
        # compressor building
        v, f = _box(cx, comp_h/2, cz, comp_w, comp_h, comp_d)
        comp_meshes.append(_static(v, f, C_COMP, "compressor %d" % (i+1)))
        # stage cylinders (visible on top)
        for si in range(DIMS["compressor_stages"]):
            sd = fcs(DIMS["comp_stage_d_m"]) / 2 * (1.0 - si * 0.15)
            sx = cx - comp_w/2 + (si + 0.5) * comp_w / 4
            v, f = _solid_cylinder(sd, comp_h/2, comp_h/2 + fcs(1.5), seg=8)
            sm = _static(v, f, _mix(C_COMP, C_COMP_HOT, si * 0.2), "stage %d" % (si+1))
            sm.pivot = np.array([sx, 0, cz])
            sm.tilt = (0, math.pi/2)
            comp_meshes.append(sm)
        # motor
        v, f = _solid_cylinder(fcs(DIMS["comp_motor_d_m"])/2, 0,
                               fcs(DIMS["comp_motor_h_m"]), seg=8)
        mm = _static(v, f, _mix(C_COMP, (0,0,0), 0.2), "motor")
        mm.pivot = np.array([cx + comp_w/2 + fcs(1.0), 0, cz])
        mm.tilt = (0, math.pi/2)
        comp_meshes.append(mm)
    # CO2 pipes from regen corridor to compressors
    for pi in range(3):
        px = fcs(-22.5) + (pi - 1) * fcs(15.0)
        pipe = _pipe((px, fcs(4.0), fcs(260.0)), (px, fcs(4.0), fcs(310.0)),
                     fcs(0.25), C_CO2TANK, seg=6)
        comp_meshes.append(pipe)
    parts.append(Part("compressors", "CO2 COMPRESSORS (4 units)", comp_meshes,
                      ["Squeezes CO2 gas into liquid for storage and transport",
                       "%d x 4-stage centrifugal compressors" % n_comp,
                       "liquefies CO2 to %.0f bar (pipeline-ready)" % DIMS["storage_bar"],
                       "%.0f kWh/t electrical, intercooled between stages" % ENERGY["compress_elec_kwh_t"],
                       "carbon steel body, internal epoxy coating",
                       "MTBF %.0f h (hermetic diaphragm, N+2), life %d yr" % (
                           COMPONENTS["compressors"]["mtbf_h"],
                           COMPONENTS["compressors"]["design_life_years"])],
                      7, (0.8, 0.4, 0.8), C_COMP))

    # --- CO2 STORAGE TANKS -------------------------------------------------
    # North of compressors: CO2 flows compressors -> storage -> pipeline
    tank_bodies = []
    tank_bands = []
    tank_saddles = []
    tank_pipes = []
    n_tanks = DIMS["storage_tanks"]
    td = fcs(DIMS["storage_d_m"])
    tl = fcs(DIMS["storage_len_m"])
    for i in range(n_tanks):
        tx = fcs(-37.5) + (i % 4) * fcs(22.0)
        tz = fcs(370.0) + (i // 4) * fcs(25.0)
        v, f = _solid_cylinder(td/2, -tl/2, tl/2, seg=12)
        tm = _static(v, f, C_CO2TANK, "CO2 tank %d" % (i+1))
        tm.pivot = np.array([tx, td/2, tz])
        tm.tilt = (0, math.pi/2)
        tank_bodies.append(tm)
        # CO2 band
        v, f = _annulus_cylinder(td/2 * 1.02, td/2 * 0.92, -tl * 0.06, tl * 0.06, seg=12)
        bm = _static(v, f, C_CO2BAND, "")
        bm.pivot = tm.pivot.copy()
        bm.tilt = tm.tilt
        tank_bands.append(bm)
        # support saddles
        for sz in (-tl * 0.35, tl * 0.35):
            v, f = _box(tx, td/4, tz + sz, fcs(2.0), td/2, fcs(1.5))
            tank_saddles.append(_static(v, f, _mix(C_COMP, (0,0,0), 0.3), ""))
    # pipes from compressors to storage tanks
    for pi in range(2):
        px = fcs(-10.0) + pi * fcs(20.0)
        tank_pipes.append(_pipe((px, fcs(5.0), fcs(330.0)), (px, fcs(5.0), fcs(370.0)),
                     fcs(0.2), C_CO2TANK, seg=6))
    tank_meshes = [_merge_static_meshes(tank_bodies, C_CO2TANK, "tank bodies"),
                   _merge_static_meshes(tank_bands, C_CO2BAND, "tank bands"),
                   _merge_static_meshes(tank_saddles, _mix(C_COMP, (0,0,0), 0.3), "tank saddles"),
                   _merge_static_meshes(tank_pipes, C_CO2TANK, "tank pipes")]
    parts.append(Part("co2tanks", "CO2 STORAGE TANKS (8 x 500 t)", tank_meshes,
                      ["Tanks that hold captured CO2 before it goes underground",
                       "%d pressurized tanks @ %.0f bar, carbon steel" % (n_tanks, DIMS["storage_bar"]),
                       "%.0f m dia x %.0f m length, internal epoxy coating" % (
                           DIMS["storage_d_m"], DIMS["storage_len_m"]),
                       "%.0f t CO2 capacity each = %.0f t total buffer" % (
                           DIMS["storage_capacity_t"],
                           n_tanks * DIMS["storage_capacity_t"]),
                       "liquid CO2 at ambient temperature",
                       "design life %d years, annual inspection" % COMPONENTS["co2_tanks"]["design_life_years"]],
                      8, (1.0, 0.3, 1.0), C_CO2TANK))

    # --- THERMAL STORAGE (molten salt tanks) -------------------------------
    # Near solar thermal field (west side) -- short thermal pipe runs
    salt_meshes = []
    n_salt = DIMS["salt_tanks"]
    sd = fcs(DIMS["salt_tank_d_m"])
    sh = fcs(DIMS["salt_tank_h_m"])
    for i in range(n_salt):
        sx = fcs(-200.0) + (i % 2) * fcs(35.0)
        sz = fcs(100.0) + (i // 2) * fcs(35.0)
        is_hot = i < 2
        col = C_SALT_HOT if is_hot else C_SALT_COLD
        v, f = _solid_cylinder(sd/2, 0, sh, seg=16)
        sm = _static(v, f, col, "salt tank %s" % ("hot" if is_hot else "cold"))
        sm.pivot = np.array([sx, 0, sz])
        salt_meshes.append(sm)
        # top cap
        v, f = _solid_cylinder(sd/2 * 1.02, sh, sh + fcs(0.5), seg=16)
        cm = _static(v, f, _mix(col, (0,0,0), 0.2), "")
        cm.pivot = np.array([sx, 0, sz])
        salt_meshes.append(cm)
    # thermal pipe from salt tanks to regen corridor
    salt_meshes.append(_pipe((fcs(-180.0), fcs(5.0), fcs(120.0)),
                             (fcs(-40.0), fcs(5.0), fcs(180.0)),
                             fcs(0.4), C_TROUGH_PIPE, seg=8))
    parts.append(Part("salt", "HEAT BATTERY -- MOLTEN SALT TANKS (1500 MWh)", salt_meshes,
                      ["Stores heat from daytime sun for use at night",
                       "%.0f MWh molten salt thermal storage, SS 316L tanks" % DIMS["thermal_storage_mwh"],
                       "2 hot tanks (%.0f C) + 2 cold tanks (%.0f C)" % (
                           DIMS["salt_hot_temp_c"], DIMS["salt_cold_temp_c"]),
                       "%.0f m dia x %.0f m tall, nitrate salt" % (
                           DIMS["salt_tank_d_m"], DIMS["salt_tank_h_m"]),
                       "carries regeneration heat through the night",
                       "charged by solar thermal surplus, design life %d years" %
                           COMPONENTS["thermal_store"]["design_life_years"]],
                      9, (-1.2, 0.5, 1.0), C_SALT_HOT))

    # --- BATTERY BANK ------------------------------------------------------
    # South of contactor array, near control building (electrical distribution)
    batt_meshes = []
    bw = fcs(DIMS["battery_bldg_w_m"])
    bh = fcs(DIMS["battery_bldg_h_m"])
    bd = fcs(DIMS["battery_bldg_d_m"])
    bx = fcs(80.0)
    bz = fcs(-220.0)
    v, f = _box(bx, bh/2, bz, bw, bh, bd)
    batt_meshes.append(_static(v, f, C_BATT, "battery building"))
    # module hints on the front
    for mi in range(8):
        mx = bx - bw/2 + (mi + 0.5) * bw / 8
        v, f = _box(mx, bh * 0.6, bz - bd/2 - fcs(0.1), bw/10, bh * 0.5, fcs(0.2))
        batt_meshes.append(_static(v, f, _mix(C_BATT, C_GOOD, 0.2), "battery module"))
    parts.append(Part("battery", "BATTERY BANK (800 MWh)", batt_meshes,
                      ["Stores extra solar/wind electricity for nighttime use",
                       "%.0f MWh LiFePO4 (LFP) battery storage" % DIMS["battery_mwh"],
                       "%d modular racks in dedicated building" % DIMS["battery_modules"],
                       "building: %.0f m x %.0f m x %.0f m, climate-controlled" % (
                           DIMS["battery_bldg_w_m"], DIMS["battery_bldg_h_m"], DIMS["battery_bldg_d_m"]),
                       "carries electrical load through the night",
                       "LFP chemistry, per-cell BMS, life %d yr" %
                           COMPONENTS["battery"]["design_life_years"]],
                      10, (1.2, 0.3, 0.8), C_BATT))

    # --- COOLING TOWERS ----------------------------------------------------
    # Real hyperbolic cooling towers: wide base, narrow waist, slightly wider top
    cool_towers = []
    cool_steam = []
    cool_basins = []
    n_ct = DIMS["cooling_towers"]
    cd_t = fcs(DIMS["cooling_d_m"])
    ch_t = fcs(DIMS["cooling_h_m"])
    ct_top = fcs(DIMS["cooling_top_d_m"])
    for i in range(n_ct):
        cx = fcs(120.0) + (i - 1.5) * fcs(20.0)
        cz = fcs(310.0)
        # Build hyperbolic tower as lofted circular rings
        n_segs = 8
        seg = _detail_seg(16)
        ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        verts, faces = [], []
        rings = []
        for si in range(n_segs + 1):
            t = si / n_segs
            y = t * ch_t
            # hyperbolic profile: base wide, waist at t=0.6, top slightly wider
            if t < 0.6:
                # narrowing from base to waist
                waist_frac = t / 0.6
                r = cd_t/2 * (1.0 - 0.25 * waist_frac)
            else:
                # widening from waist to top
                waist_frac = (t - 0.6) / 0.4
                r = cd_t/2 * 0.75 + (ct_top/2 - cd_t/2 * 0.75) * waist_frac
            base = len(verts)
            for a in ang:
                verts.append((r * math.cos(a), y, r * math.sin(a)))
            rings.append(base)
        for ri in range(n_segs):
            a_base, b_base = rings[ri], rings[ri + 1]
            for si2 in range(seg):
                s2 = (si2 + 1) % seg
                faces.append((a_base + si2, a_base + s2, b_base + s2, b_base + si2))
        # bottom cap
        for si2 in range(seg):
            s2 = (si2 + 1) % seg
            faces.append((rings[0], rings[0] + si2, rings[0] + s2))
        # top cap (open -- just the ring, no cap for realism)
        cm = _static(verts, faces, C_COOLING, "cooling tower %d" % (i+1))
        cm.pivot = np.array([cx, 0, cz])
        cool_towers.append(cm)
        # steam plume (wider, translucent-looking)
        v, f = _solid_cylinder(ct_top/2 * 0.7, ch_t, ch_t + fcs(6.0), seg=8)
        sm = _static(v, f, C_COOLING_STEAM, "steam plume")
        sm.pivot = np.array([cx, 0, cz])
        cool_steam.append(sm)
        # base basin (water collection)
        v, f = _solid_cylinder(cd_t/2 * 1.1, 0, fcs(1.5), seg=16)
        bm = _static(v, f, _mix(C_COOLING, (0,0,0), 0.3), "cooling basin")
        bm.pivot = np.array([cx, 0, cz])
        cool_basins.append(bm)
    cool_meshes = [_merge_static_meshes(cool_towers, C_COOLING, "cooling towers"),
                   _merge_static_meshes(cool_steam, C_COOLING_STEAM, "steam plumes"),
                   _merge_static_meshes(cool_basins, _mix(C_COOLING, (0,0,0), 0.3), "cooling basins")]
    parts.append(Part("cooling", "COOLING TOWERS (4)", cool_meshes,
                      ["Cools down the plant by releasing heat as steam",
                       "%d cooling towers for heat rejection" % n_ct,
                       "%.0f m tall, %.0f m base diameter, hyperbolic" % (
                           DIMS["cooling_h_m"], DIMS["cooling_d_m"]),
                       "rejects heat from compressors + regeneration",
                       "counterflow design, water recycled closed-loop"],
                      11, (0.2, 0.8, 1.0), C_COOLING))

    # --- CONTROL BUILDING --------------------------------------------------
    ctrl_meshes = []
    cw_b = fcs(DIMS["control_w_m"])
    ch_b = fcs(DIMS["control_h_m"])
    cd_b = fcs(DIMS["control_d_m"])
    cx_b = fcs(0.0)
    cz_b = fcs(-220.0)
    v, f = _box(cx_b, ch_b/2, cz_b, cw_b, ch_b, cd_b)
    ctrl_meshes.append(_static(v, f, C_CONTROL, "control building"))
    # windows
    for lvl in range(3):
        yy = fcs(2.0) + lvl * fcs(3.0)
        v, f = _box(cx_b, yy, cz_b - cd_b/2 - fcs(0.05), cw_b * 0.8, fcs(1.5), fcs(0.1))
        ctrl_meshes.append(_static(v, f, C_CONTROL_WIN, "window"))
    # antenna
    v, f = _solid_cylinder(fcs(0.15), 0, fcs(5.0), seg=8)
    am = _static(v, f, C_MAST, "antenna")
    am.pivot = np.array([cx_b, ch_b, cz_b])
    ctrl_meshes.append(am)
    parts.append(Part("control", "CONTROL BUILDING", ctrl_meshes,
                      ["Where staff monitor and run the plant 24/7",
                       "facility operations center, 2-story reinforced concrete",
                       "%d staff, 24/7 monitoring (%d maintenance)" % (
                           FACILITY["staff"], MAINTENANCE["staff_maint"]),
                       "auto-cycle controller for all %d beds" % FACILITY["sorbent_beds"],
                       "industrial PLC + redundant servers, SCADA system",
                       "weather station + CO2 sensors + MRV instrumentation"],
                      12, (0, 0.5, 1.2), C_CONTROL))

    # --- ELECTRICAL DISTRIBUTION (cable runs + substation) ------------------
    elec_transformers = []
    elec_insulators = []
    elec_poles = []
    elec_cables = []
    cable_col = _mix(C_COMP, (0,0,0), 0.5)
    pole_col = _mix(C_GROUND, (60,60,60), 0.5)
    # substation transformers next to control building
    for ti in range(3):
        tx = cx_b + cw_b/2 + fcs(5.0) + ti * fcs(4.0)
        tz = cz_b + fcs(5.0)
        v, f = _box(tx, fcs(2.0), tz, fcs(3.0), fcs(4.0), fcs(2.5))
        elec_transformers.append(_static(v, f, _mix(C_COMP, C_COMP_HOT, 0.2),
                                   "transformer %d" % (ti+1)))
        # insulators on top
        for ii in range(3):
            v, f = _solid_cylinder(fcs(0.15), fcs(4.0), fcs(4.8), seg=8)
            im = _static(v, f, C_CONTROL_WIN, "insulator")
            im.pivot = np.array([tx - fcs(1.0) + ii * fcs(1.0), 0, tz])
            elec_insulators.append(im)
    # cable run from solar PV field (east) to substation
    n_poles = 8
    for pi in range(n_poles):
        px = fcs(100.0) + pi * fcs(80.0)
        pz = cz_b
        # pole
        v, f = _solid_cylinder(fcs(0.2), 0, fcs(8.0), seg=8)
        pm = _static(v, f, pole_col, "power pole")
        pm.pivot = np.array([px, 0, pz])
        elec_poles.append(pm)
        # crossbar
        v, f = _box(px, fcs(7.5), pz, fcs(3.0), fcs(0.15), fcs(0.15))
        elec_poles.append(_static(v, f, pole_col, ""))
        # cables (3-phase)
        for ci in range(3):
            cx_offset = (ci - 1) * fcs(0.8)
            if pi < n_poles - 1:
                elec_cables.append(_pipe((px + cx_offset, fcs(7.8), pz),
                              (px + fcs(80.0) + cx_offset, fcs(7.8), pz),
                              fcs(0.05), cable_col, seg=5))
    # cable run from battery building to substation (short)
    elec_cables.append(_pipe((bx, fcs(8.0), bz),
                  (cx_b + cw_b/2, fcs(8.0), cz_b),
                  fcs(0.08), cable_col, seg=6))
    elec_meshes = [_merge_static_meshes(elec_transformers, _mix(C_COMP, C_COMP_HOT, 0.2), "transformers"),
                   _merge_static_meshes(elec_insulators, C_CONTROL_WIN, "insulators"),
                   _merge_static_meshes(elec_poles, pole_col, "poles"),
                   _merge_static_meshes(elec_cables, cable_col, "cables")]
    parts.append(Part("elec", "POWER LINES & SUBSTATION", elec_meshes,
                      ["Wires and transformers that distribute electricity",
                       "overhead 3-phase power lines on wooden poles",
                       "substation: 3 step-up transformers (33kV / 690V)",
                       "connects solar PV, wind turbines, battery to substation",
                       "grid synchronization + island-mode capability"],
                      12, (0.8, 0.6, 0.2), cable_col))

    # --- CO2 PIPELINE ------------------------------------------------------
    pipe_meshes = []
    pd = fcs(DIMS["pipeline_d_m"])
    pl = fcs(DIMS["pipeline_len_m"])
    # Pipeline runs north from storage tanks to sequestration site
    pipe_meshes.append(_pipe((fcs(0.0), fcs(2.0), fcs(420.0)),
                              (fcs(0.0), fcs(2.0), fcs(420.0) + pl),
                              pd, C_PIPELINE, seg=10))
    # pipeline support stands
    for si in range(6):
        px = fcs(0.0)
        pz = fcs(420.0) + (si + 0.5) * pl / 6
        v, f = _box(px, fcs(1.0), pz, fcs(0.4), fcs(2.0), fcs(0.4))
        pipe_meshes.append(_static(v, f, _mix(C_PIPELINE, (0,0,0), 0.3), ""))
    parts.append(Part("pipeline", "CO2 PIPELINE TO UNDERGROUND STORAGE", pipe_meshes,
                      ["Pipe that carries captured CO2 deep underground",
                       "%.0f m on-site, %.0f m dia, carbon steel" % (
                          DIMS["pipeline_len_m"], DIMS["pipeline_d_m"]),
                       "carries liquid CO2 at %.0f bar to geological storage" % DIMS["storage_bar"],
                       "injects into deep saline aquifers (1-2 km depth)",
                       "%.0f t/h pipeline capacity, annual pigging" % CO2_STORE["pipeline_rate_t_h"],
                       "design life %d years" % COMPONENTS["pipeline"]["design_life_years"]],
                      13, (1.5, 0.2, 1.2), C_PIPELINE))

    return parts


# ---------------------------------------------------------------------------
# URBAN MINI-PLANT MODEL -- skyscraper cross-section with DAC unit on vacant floor
#  Coord frame:  +Y = up (building height), +X = width, +Z = depth
#  Display scale: URBAN_DISP = 1/15 (1 unit ~ 15 m, building ~48 m visible)
# ---------------------------------------------------------------------------

URBAN_DISP = 1.0 / 15.0

def _urb(m):
    """metres -> URBAN-view display units."""
    return m * URBAN_DISP

def build_urban_parts(_n=1):
    """A cutaway skyscraper with a DAC mini-plant installed on a vacant floor.
    Shows the building structure (floors, columns, windows) with the DAC unit
    visible inside -- 4 compact contactors, 2 regen chambers, compressor, tanks,
    and ducting. Other floors shown as office space for context."""
    parts = []
    bw = _urb(30.0)      # building width (30 m typical floor plate)
    bd = _urb(30.0)      # building depth
    fh = _urb(4.0)       # floor height (4 m)
    nf = 12              # number of floors to show (cutaway)
    bh = fh * nf         # total building height in display units
    col_w = _urb(0.8)    # column width
    win_w = _urb(3.0)    # window bay width
    win_h = _urb(2.5)    # window height
    dac_floor = 5        # which floor has the DAC unit (0-indexed from bottom)

    # --- BUILDING STRUCTURE (cutaway: front face removed) -----------------
    bldg_meshes = []

    # Floor slabs (horizontal plates)
    for i in range(nf + 1):
        y = i * fh
        v, f = _box(0, y, 0, bw, _urb(0.3), bd)
        col = C_FLOOR_EDGE if i == dac_floor or i == dac_floor + 1 else C_FLOOR
        bldg_meshes.append(_static(v, f, col, name="floor slab %d" % i, group="building"))

    # Side walls (left and right only -- front cut away, back shown)
    for sx in (-1, 1):
        v, f = _box(sx * bw / 2, bh / 2, 0, _urb(0.3), bh, bd)
        bldg_meshes.append(_static(v, f, C_BUILDING, name="side wall", group="building"))

    # Back wall
    v, f = _box(0, bh / 2, bd / 2, bw, bh, _urb(0.3))
    bldg_meshes.append(_static(v, f, C_BUILDING, name="back wall", group="building"))

    # Columns (front edge, between floors)
    for i in range(nf):
        y = i * fh + fh / 2
        for cx in (-bw * 0.35, -bw * 0.12, bw * 0.12, bw * 0.35):
            v, f = _box(cx, y, -bd / 2, col_w, fh * 0.95, _urb(0.4))
            bldg_meshes.append(_static(v, f, C_BUILDING_HI, name="column", group="building"))

    # Windows on back wall (lit/unlit pattern for office floors)
    for i in range(nf):
        if i == dac_floor:
            continue  # DAC floor has no windows (sealed)
        y = i * fh + _urb(1.0)
        for wx in range(-3, 4):
            cx = wx * _urb(4.0)
            if abs(cx) > bw * 0.42:
                continue
            lit = (i + wx) % 3 == 0
            v, f = _box(cx, y, bd / 2 - _urb(0.15), win_w, win_h, _urb(0.1))
            bldg_meshes.append(_static(v, f, C_WINDOW_LIT if lit else C_WINDOW,
                                        name="window", group="building"))

    # Office furniture on non-DAC floors (desks visible through cutaway)
    desk_w = _urb(2.0)
    desk_d = _urb(1.0)
    desk_h = _urb(0.05)
    desk_col = (100, 95, 85)   # warm desk wood tone
    chair_col = (60, 65, 75)   # office chair
    for i in range(nf):
        if i == dac_floor:
            continue  # DAC floor has no desks
        y = i * fh + _urb(0.75)
        # Two rows of desks per floor (front and back)
        for row_z in (-bd * 0.25, bd * 0.25):
            for dx in (-2, -1, 0, 1, 2):
                cx = dx * _urb(3.5)
                if abs(cx) > bw * 0.4:
                    continue
                # Desk surface
                v, f = _box(cx, y, row_z, desk_w, desk_h, desk_d)
                bldg_meshes.append(_static(v, f, desk_col, name="desk", group="building"))
                # Chair (small box behind desk)
                v, f = _box(cx, y - _urb(0.35), row_z + desk_d * 0.7, _urb(0.5), _urb(0.5), _urb(0.5))
                bldg_meshes.append(_static(v, f, chair_col, name="chair", group="building"))

    # Office floor markers (green edge on non-DAC floors to show active workforce)
    office_marker_col = (70, 130, 90)  # green = active office
    for i in range(nf):
        if i == dac_floor:
            continue
        y = i * fh
        v, f = _box(0, y, -bd / 2 - _urb(0.08), bw * 1.01, _urb(0.08), _urb(0.08))
        bldg_meshes.append(_static(v, f, office_marker_col, name="office floor marker", group="building"))

    parts.append(Part("building", "SKYSCRAPER STRUCTURE (cutaway)", bldg_meshes,
                      ["Cutaway view: front face removed to show interior",
                       "Building: %d floors x 4.0 m = %.0f m tall" % (nf, nf * 4.0),
                       "Floor plate: 30 m x 30 m = 900 m2 (typical commercial)",
                       "Structure: reinforced concrete + steel columns",
                       "DAC unit installed on floor %d ONLY (blue markers)" % (dac_floor + 1),
                       "All other floors: active office space with workers",
                       "Desks and chairs visible on office floors",
                       "Green floor markers = occupied office floors",
                       "Only 1 floor needed -- building stays fully operational"],
                      0, (0, 0, 0), C_BUILDING))

    # --- DAC MINI-PLANT (on the vacant floor) ------------------------------
    dy = dac_floor * fh + fh / 2  # center Y of DAC floor
    cw = _urb(URBAN["contactor_w_m"])
    ch = _urb(URBAN["contactor_h_m"])
    cd = _urb(URBAN["contactor_d_m"])
    fan_r = _urb(URBAN["fan_d_m"]) / 2
    fan_hub_r = fan_r * 0.25
    n_fans = URBAN["fans_per_contactor"]
    n_blades = URBAN["fan_blades"]

    # Contactor frames (2x2 grid, centered on floor)
    cont_meshes_all = []
    contactor_positions = [
        (-bw * 0.22, dy, -bd * 0.15),
        ( bw * 0.22, dy, -bd * 0.15),
        (-bw * 0.22, dy,  bd * 0.15),
        ( bw * 0.22, dy,  bd * 0.15),
    ]
    for ci, (cx, cy, cz) in enumerate(contactor_positions):
        # Frame
        v, f = _box(cx, cy, cz, cw, ch, cd)
        cont_meshes_all.append(_static(v, f, C_URBAN_FRAME,
                                        name="contactor frame %d" % (ci+1), group="building"))
        # Sorbent bed (inside, visible through cutaway)
        v, f = _box(cx, cy, cz + cd * 0.2, cw * 0.85, ch * 0.8, cd * 0.3)
        cont_meshes_all.append(_static(v, f, C_URBAN_SORB,
                                        name="sorbent bed %d" % (ci+1), group="building"))

        # Fans (front face, spinning)
        for fi in range(n_fans):
            fx = cx + (fi - (n_fans - 1) / 2.0) * fan_r * 2.3
            fy = cy
            fz = cz - cd / 2 - _urb(0.05)
            # Shroud
            v, f = _solid_cylinder(fan_r * 1.1, fz - _urb(0.05), fz + _urb(0.05), seg=16)
            shroud = Mesh(v, f, C_URBAN_FRAME, name="fan shroud", spin=0.0, group="building")
            shroud.tilt = (0, math.pi / 2)
            shroud.pivot = np.array([fx, fy, fz])
            cont_meshes_all.append(shroud)
            # Hub
            v, f = _solid_cylinder(fan_hub_r, fz - _urb(0.04), fz + _urb(0.04), seg=8)
            hub = Mesh(v, f, C_URBAN_HUB, name="fan hub", spin=1.0, group="contactorfan")
            hub.tilt = (0, math.pi / 2)
            hub.pivot = np.array([fx, fy, fz])
            cont_meshes_all.append(hub)
            # Blades
            for b in range(n_blades):
                a = b * 2 * math.pi / n_blades
                bv, bf = _box(fan_r * 0.7, 0.0, _urb(0.03), fan_r * 1.3, _urb(0.1), _urb(0.04))
                bv = np.asarray(bv, float) @ rot_z(a).T
                blade = Mesh(bv, bf, C_URBAN_FAN, name="fan blade", spin=1.0, group="contactorfan")
                blade.tilt = (0, math.pi / 2)
                blade.pivot = np.array([fx, fy, fz])
                cont_meshes_all.append(blade)

    parts.append(Part("urban_contactors", "DAC CONTACTOR ARRAY (4 units)", cont_meshes_all,
                      ["4 compact contactors in 2x2 grid",
                       "Each: %.1f m W x %.1f m H x %.1f m D" % (
                           URBAN["contactor_w_m"], URBAN["contactor_h_m"], URBAN["contactor_d_m"]),
                       "Fans: %d x %.1f m dia per unit, %d-blade CFRP" % (
                           n_fans, URBAN["fan_d_m"], n_blades),
                       "Sorbent: %.0f t/bed, %.2f kg CO2/kg (next-gen MOF)" % (
                           URBAN["sorbent_t_per_bed"], URBAN["sorbent_cap_kg_kg"]),
                       "Airflow: ~50,000 m3/h per unit, ~200,000 m3/h total",
                       "Noise: %d dB at 1m (office-compatible, quiet fans)" % URBAN["noise_db"]],
                      1, (0, 0, _urb(2.0)), C_URBAN_FRAME))

    # --- Regeneration chambers (2 units, against back wall) ----------------
    regen_meshes = []
    rw = _urb(URBAN["regen_w_m"])
    rh = _urb(URBAN["regen_h_m"])
    rd = _urb(URBAN["regen_d_m"])
    for ri in range(URBAN["regen_units"]):
        rx = (ri - 0.5) * rw * 2.5
        ry = dy
        rz = bd / 2 - rd - _urb(0.5)
        v, f = _box(rx, ry, rz, rw, rh, rd)
        regen_meshes.append(_static(v, f, C_URBAN_REGEN,
                                     name="regen chamber %d" % (ri+1), group="building"))
        # Insulation band
        v, f = _box(rx, ry + rh * 0.4, rz, rw * 1.05, _urb(0.15), rd * 1.05)
        regen_meshes.append(_static(v, f, _mix(C_URBAN_REGEN, C_WARN, 0.3),
                                     name="regen insulation", group="building"))

    parts.append(Part("urban_regen", "REGENERATION CHAMBERS (2 units)", regen_meshes,
                      ["2 compact regen chambers (2 beds each)",
                       "Each: %.1f m x %.1f m x %.1f m" % (
                           URBAN["regen_w_m"], URBAN["regen_h_m"], URBAN["regen_d_m"]),
                       "Process: vacuum-temperature swing (VSA), 100 C",
                       "Insulated for office-safe surface temperature"],
                      2, (0, 0, _urb(1.5)), C_URBAN_REGEN))

    # --- CO2 compressor + buffer tanks (corner of floor) -------------------
    comp_tank_meshes = []
    # Compressor (small box near regen)
    v, f = _box(-bw * 0.35, dy, bd / 2 - _urb(3.0), _urb(1.5), _urb(1.2), _urb(1.0))
    comp_tank_meshes.append(_static(v, f, C_URBAN_COMP, name="compressor", group="building"))

    # CO2 buffer tanks (2 horizontal cylinders)
    for ti in range(URBAN["co2_tanks"]):
        tx = bw * 0.35 + ti * _urb(1.8)
        ty = dy
        tz = bd / 2 - _urb(2.5)
        v, f = _solid_cylinder(_urb(0.6), -_urb(1.5), _urb(1.5), seg=16)
        tank = Mesh(v, f, C_URBAN_TANK, name="CO2 tank %d" % (ti+1), spin=0.0, group="building")
        tank.tilt = (0, math.pi / 2)
        tank.pivot = np.array([tx, ty, tz])
        comp_tank_meshes.append(tank)

    parts.append(Part("urban_co2", "CO2 COMPRESSION + STORAGE", comp_tank_meshes,
                      ["Compact CO2 compressor: 1 unit, 150 bar output",
                       "Buffer tanks: %d x %.0f t capacity" % (
                           URBAN["co2_tanks"], URBAN["co2_tank_cap_t"]),
                       "CO2 routed via building riser pipe to street collection",
                       "Tank size: 0.6 m dia x 3 m horizontal (fits in utility space)"],
                      3, (_urb(2.0), 0, 0), C_URBAN_TANK))

    # --- Ducting (connecting contactors to regen) --------------------------
    duct_meshes = []
    for ci in range(4):
        cx, _, cz = contactor_positions[ci]
        # Duct from contactor to back wall area
        d = _pipe(
            np.array([cx, dy, cz + cd / 2]),
            np.array([cx * 0.3, dy, bd / 2 - _urb(3.5)]),
            _urb(0.15), C_URBAN_DUCT, seg=6)
        d.group = "building"
        duct_meshes.append(d)
    # Riser pipe (vertical, going down to street level)
    riser = _pipe(
        np.array([bw * 0.4, dy, bd / 2 - _urb(2.0)]),
        np.array([bw * 0.4, 0, bd / 2 - _urb(2.0)]),
        _urb(0.12), C_URBAN_DUCT, seg=6)
    riser.group = "building"
    duct_meshes.append(riser)

    parts.append(Part("urban_ducts", "DUCTING + CO2 RISER", duct_meshes,
                      ["Connecting ducts: contactor -> regen -> compressor",
                       "CO2 riser: vertical pipe to street-level collection",
                       "Ducting integrates with building HVAC where possible",
                       "All ducting: SS 304, insulated for noise + thermal"],
                      4, (0, _urb(1.0), 0), C_URBAN_DUCT))

    # --- Floor label marker (highlighted DAC floor edge) -------------------
    label_meshes = []
    v, f = _box(0, dac_floor * fh, -bd / 2 - _urb(0.1), bw * 1.02, _urb(0.15), _urb(0.1))
    label_meshes.append(_static(v, f, C_CO2, name="dac floor marker", group="building"))
    v, f = _box(0, (dac_floor + 1) * fh, -bd / 2 - _urb(0.1), bw * 1.02, _urb(0.15), _urb(0.1))
    label_meshes.append(_static(v, f, C_CO2, name="dac floor marker top", group="building"))

    parts.append(Part("urban_label", "DAC FLOOR -- 1 FLOOR ONLY (blue)", label_meshes,
                      ["Floor %d: DAC mini-plant installed (1 floor only)" % (dac_floor + 1),
                       "Blue markers = the DAC floor boundary",
                       "All other floors: active office with workers (green markers)",
                       "Building remains fully operational -- only 1 vacant floor needed",
                       "DAC unit fits within standard 4m ceiling clearance"],
                      5, (0, 0, _urb(0.5)), C_CO2))

    return parts

# =============================================================================
# SECTION 5 -- CAPTURE PHYSICS (CO2 mass transfer, energy, thermodynamics)
# =============================================================================

def sun_factor(hour):
    """Solar elevation proxy 0..1 across a 24h day (peak at noon)."""
    if hour < 5.5 or hour > 18.5:
        return 0.0
    t = (hour - 5.5) / 13.0
    return max(0.0, math.sin(t * math.pi))


def solar_pv_kw(sun):
    """Electrical output from the PV field given solar intensity 0..1."""
    return SOLAR_PV_PEAK_KW * sun


def solar_thermal_kw(sun):
    """Thermal output from the parabolic trough field given solar intensity 0..1."""
    return SOLAR_TH_PEAK_KW * sun


def wind_kw(wind_ms):
    """Wind turbine electrical output (cubic law, cut-in 3 m/s, rated 12 m/s,
    cut-out 25 m/s)."""
    if wind_ms < 3.0 or wind_ms > 25.0:
        return 0.0
    frac = clamp((wind_ms / WIND_REF_MS) ** 3, 0.0, 1.0)
    return WIND_RATED_KW * frac


def co2_in_air_kg_per_m3():
    """kg of CO2 per m3 of air at current atmospheric concentration."""
    return CO2_DENSITY_KG_M3


def air_through_contactor_kg_per_s(fan_d_m, rpm):
    """Mass flow of air through a single contactor fan (approximate axial flow).
    Q = A * v, where v ~ tip_speed * pitch_factor. Returns kg/s."""
    r = fan_d_m / 2.0
    tip_speed = rpm * 2.0 * math.pi * r / 60.0
    v_axial = tip_speed * 0.40  # optimized pitch (axial flow ~ 40% of tip speed)
    area = math.pi * r * r
    vol_flow = area * v_axial  # m3/s per fan
    return vol_flow * AIR_DENSITY_KG_M3  # kg/s per fan


def co2_capture_rate_kg_per_s(beds_active, fan_rpm=120.0):
    """Total CO2 capture rate across all active beds (kg/s).
    Each contactor has 8 fans; capture_eff fraction of CO2 is absorbed.
    Uses volume flow (m3/s) * CO2 density (kg/m3) for correct units."""
    co2_per_m3 = co2_in_air_kg_per_m3()
    # Volume flow per fan (m3/s)
    r = DIMS["fan_d_m"] / 2.0
    tip_speed = fan_rpm * 2.0 * math.pi * r / 60.0
    v_axial = tip_speed * 0.40
    area = math.pi * r * r
    vol_per_fan = area * v_axial  # m3/s per fan
    vol_per_contactor = vol_per_fan * DIMS["contactor_fans"]  # m3/s per contactor
    co2_per_contactor = vol_per_contactor * co2_per_m3 * SORBENT["capture_eff"]
    return beds_active * co2_per_contactor


def regen_thermal_kw_needed(capture_t_h):
    """Thermal power needed for regeneration given capture rate (t CO2/h)."""
    return capture_t_h * ENERGY["regen_thermal_kwh_t"]  # kW (kWh/h = kW)


def elec_kw_needed(capture_t_h):
    """Electrical power needed for fans + vacuum + compressors + aux (kW)."""
    return capture_t_h * (ENERGY["fan_elec_kwh_t"] +
                          ENERGY["vacuum_elec_kwh_t"] +
                          ENERGY["compress_elec_kwh_t"] +
                          ENERGY["aux_elec_kwh_t"])


def cost_per_tonne_co2(thermal_kwh, elec_kwh, co2_t):
    """Operating cost per tonne CO2 captured ($/t).
    Includes energy, sorbent replacement, labor, maintenance, insurance,
    land lease, and water costs for a realistic OPEX model."""
    if co2_t < 1e-6:
        return 999.0
    # Energy cost (renewable self-generation -> near zero, but account for amortized CAPEX)
    energy_cost = (thermal_kwh * CAMPAIGN["energy_cost_per_kwh"] +
                   elec_kwh * CAMPAIGN["energy_cost_per_kwh"])
    # Sorbent replacement cost (per tonne CO2)
    sorbent_cost = (co2_t * SORBENT["capacity_kg_per_kg"] * 1000 *
                    CAMPAIGN["sorbent_replacement_frac"] *
                    CAMPAIGN["sorbent_cost_per_t"] / 1000.0)
    # Fixed OPEX prorated per tonne (annual costs / annual capture)
    annual_capture = FACILITY["capture_t_year"]
    fixed_opex = (CAMPAIGN["labor_cost_per_year"] +
                  CAMPAIGN["maint_cost_per_year"] +
                  CAMPAIGN["insurance_per_year"] +
                  CAMPAIGN["land_lease_per_year"])
    fixed_per_t = fixed_opex / annual_capture
    # Water cost
    water_cost = co2_t * CAMPAIGN["water_m3_per_t_co2"] * CAMPAIGN["water_cost_per_m3"]
    return (energy_cost + sorbent_cost + fixed_per_t * co2_t + water_cost) / co2_t


def ref_emitter_t(year_frac):
    """Reference industrial emitter CO2 output (tonnes) for comparison."""
    return CAMPAIGN["co2_ref_emit_t_year"] * year_frac


# =============================================================================
# SECTION 6 -- CAPTURE POWERTRAIN (the renewable-first energy economy)
# =============================================================================

class CapturePowertrain:
    """Renewable-first DAC energy economy: solar thermal + geothermal -> 
    regeneration heat. Solar PV + wind -> fans + compressors + vacuum.
    Thermal storage + battery carry the facility through the night.
    The controller auto-adjusts the number of active beds based on available
    energy so capture is maximized without ever running out."""

    def __init__(self):
        self.batt_kwh = ELEC["batt_mwh"] * 1000.0
        self.soc = ELEC["soc_start"]
        self.thermal_kwh = THERMAL_STORE["capacity_mwh"] * 1000.0
        self.thermal_frac = THERMAL_STORE["start_frac"]
        self.co2_storage_t = CO2_STORE["start_frac"] * CO2_STORE["capacity_t"]
        self.beds_active = CAPTURE_CTRL["beds_active_max"]
        self.fan_rpm = 120.0
        self.regen_temp_c = THERM["ambient_c"]
        self.capture_rate_kg_s = 0.0
        self.mode = "STARTUP"
        self.demo_cycle_state = "capture"
        self.demo_cycle_phase_min = 0.0

        # sorbent bed phases (staggered across 80 beds)
        self.bed_phases = []
        for i in range(FACILITY["sorbent_beds"]):
            phase = (i / FACILITY["sorbent_beds"]) * SORBENT["cycle_total_min"]
            self.bed_phases.append({"phase_min": phase, "state": "capture"})

        # running tallies
        self.total_s = 0.0
        self.co2_captured_t = 0.0
        self.co2_sequestered_t = 0.0
        self.solar_pv_kwh = 0.0
        self.solar_th_kwh = 0.0
        self.wind_kwh = 0.0
        self.geo_kwh = 0.0
        self.thermal_used_kwh = 0.0
        self.elec_used_kwh = 0.0
        self.batt_out_kwh = 0.0
        self.batt_in_kwh = 0.0
        self.thermal_out_kwh = 0.0
        self.thermal_in_kwh = 0.0
        # 15-year operational tracking
        self.water_used_m3 = 0.0
        self.cumulative_opex = 0.0
        self.cumulative_sorbent_cost = 0.0
        self.cumulative_battery_cost = 0.0
        self.co2_purity = 0.999  # starts high, degrades with sorbent age
        self.peak_capture_t_h = 0.0  # best capture rate achieved
        self.capture_hours = 0.0  # hours of active capture (for capacity factor)
        self.flow = {"solar_pv": 0.0, "solar_th": 0.0, "wind": 0.0, "geo": 0.0,
                     "thermal_store": 0.0, "regen_thermal": 0.0,
                     "fan_elec": 0.0, "vacuum_elec": 0.0, "compress_elec": 0.0,
                     "aux_elec": 0.0, "batt_net": 0.0, "thermal_net": 0.0,
                     "co2_capture": 0.0, "co2_sequester": 0.0}

        # ---- degradation & maintenance state ----
        self.elapsed_years = 0.0
        self.component_health = {}   # 0..1 per component (1 = new)
        self.component_failures = {}  # cumulative failure count
        self.component_repairs = {}   # cumulative repair count
        self.prevented_failures = {}  # issues caught by predictive maintenance
        self.maintenance_log = []     # list of (year, component, action)
        self.downtime_h = 0.0         # total unplanned downtime hours
        self.maintenance_h = 0.0      # total scheduled maintenance hours
        self.sorbent_replaced_count = 0
        self.battery_replaced_count = 0
        for cname in COMPONENTS:
            self.component_health[cname] = 1.0
            self.component_failures[cname] = 0
            self.component_repairs[cname] = 0
            self.prevented_failures[cname] = 0

    @property
    def soc_norm(self):
        return clamp((self.soc - ELEC["soc_min"]) / (ELEC["soc_max"] - ELEC["soc_min"]))

    @property
    def thermal_norm(self):
        return clamp((self.thermal_frac - THERMAL_STORE["min_frac"]) /
                     (THERMAL_STORE["max_frac"] - THERMAL_STORE["min_frac"]))

    @property
    def co2_storage_frac(self):
        return clamp(self.co2_storage_t / CO2_STORE["capacity_t"])

    @property
    def capture_rate_t_h(self):
        return self.capture_rate_kg_s * 3.6

    def route_battery(self, dE_kwh):
        """Signed kWh into/out of the battery with round-trip loss on charge."""
        if dE_kwh >= 0.0:
            room = max(0.0, (ELEC["soc_max"] - self.soc) * self.batt_kwh)
            into = min(dE_kwh, room)
            self.soc += into * STORAGE["battery_rt"] / self.batt_kwh
            self.batt_in_kwh += into
        else:
            self.soc += dE_kwh / self.batt_kwh
            self.batt_out_kwh += -dE_kwh
        self.soc = clamp(self.soc, ELEC["soc_min"], ELEC["soc_max"] + 1e-3)

    def route_thermal(self, dE_kwh):
        """Signed kWh into/out of molten salt thermal storage."""
        if dE_kwh >= 0.0:
            room = max(0.0, (THERMAL_STORE["max_frac"] - self.thermal_frac) *
                       self.thermal_kwh)
            into = min(dE_kwh, room)
            self.thermal_frac += into * STORAGE["thermal_rt"] / self.thermal_kwh
            self.thermal_in_kwh += into
        else:
            self.thermal_frac += dE_kwh / self.thermal_kwh
            self.thermal_out_kwh += -dE_kwh
        self.thermal_frac = clamp(self.thermal_frac,
                                  THERMAL_STORE["min_frac"],
                                  THERMAL_STORE["max_frac"] + 1e-3)

    def update_demo(self, dt):
        """CAPTURE preview: always-running showcase for display."""
        self.fan_rpm = 120.0
        self.beds_active = FACILITY["sorbent_beds"]
        self.regen_temp_c += (THERM["regen_target_c"] - self.regen_temp_c) * min(1.0, dt * 0.5)
        self.capture_rate_kg_s = co2_capture_rate_kg_per_s(self.beds_active, self.fan_rpm)
        # Advance bed phases so sorbent cycles through capture -> regen -> cool
        dt_min = dt / 60.0 * 60.0  # dt is in seconds; convert to simulated minutes (sped up)
        self.update_bed_phases(dt_min * 10.0)  # 10x speed for visible cycling in demo
        # Track demo cycle state from bed 0
        if self.bed_phases:
            self.demo_cycle_state = self.bed_phases[0]["state"]
            self.demo_cycle_phase_min = self.bed_phases[0]["phase_min"]
        else:
            self.demo_cycle_state = "capture"
            self.demo_cycle_phase_min = 0.0

    def update_bed_phases(self, dt_min):
        """Advance the staggered sorbent bed cycle phases."""
        for bed in self.bed_phases:
            bed["phase_min"] += dt_min
            if bed["phase_min"] >= SORBENT["cycle_total_min"]:
                bed["phase_min"] -= SORBENT["cycle_total_min"]
            p = bed["phase_min"]
            if p < SORBENT["cycle_capture_min"]:
                bed["state"] = "capture"
            elif p < SORBENT["cycle_capture_min"] + SORBENT["cycle_regen_min"]:
                bed["state"] = "regen"
            else:
                bed["state"] = "cool"

    def update_degradation(self, dt_h):
        """Advance component degradation, model failures, and track maintenance.
        Called once per update tick with elapsed hours."""
        prev_years = self.elapsed_years
        self.elapsed_years += dt_h / 8760.0
        year_now = self.elapsed_years

        # Degradation: linear capacity loss per year
        for cname, spec in COMPONENTS.items():
            if spec["degradation_rate"] > 0:
                self.component_health[cname] = max(0.0,
                    self.component_health[cname] - spec["degradation_rate"] * dt_h / 8760.0)

        # Random failures: Poisson process with rate = quantity / MTBF
        # Predictive maintenance catches most issues BEFORE they become failures.
        # The predictive_factor (0.97-0.995) is the fraction of potential failures
        # that are prevented by condition monitoring and scheduled maintenance.
        for cname, spec in COMPONENTS.items():
            if spec["mtbf_h"] >= 999999:
                continue
            n_units = max(1, spec["quantity"])
            pf = spec.get("predictive_factor", 0.95)
            # Raw failure rate (without predictive maintenance)
            raw_rate = n_units / spec["mtbf_h"]
            raw_expected = raw_rate * dt_h
            # Sample raw potential failures from Poisson
            n_potential = int(np.random.poisson(raw_expected))
            if n_potential > 0:
                # Each potential failure is independently prevented by
                # predictive maintenance with probability = predictive_factor
                prevents = np.random.random(n_potential) < pf
                n_prevented = int(np.sum(prevents))
                n_fails = n_potential - n_prevented
                self.prevented_failures[cname] += n_prevented
                # Prevented issues get fixed during scheduled maintenance (no downtime)
                if n_prevented > 0:
                    self.maintenance_h += n_prevented * spec["repair_h"] * 0.5  # faster when planned
                if n_fails > 0:
                    self.component_failures[cname] += n_fails
                    frac_down = min(1.0, n_fails / n_units)
                    self.downtime_h += frac_down * spec["repair_h"]
                    self.component_repairs[cname] += n_fails
                    self.component_health[cname] = min(1.0,
                        self.component_health[cname] + 0.05 * min(n_fails, 5))
                    self.maintenance_log.append(
                        (year_now, cname, "%d FAILURE(S): %s, repaired in %.0fh" % (
                            n_fails, spec["failure_mode"], n_fails * spec["repair_h"])))

        # Scheduled maintenance (preventive) -- accrue hours
        self.maintenance_h += (MAINTENANCE["daily_inspect_h"] / 24.0 +
                               MAINTENANCE["weekly_filter_h"] / 168.0 +
                               MAINTENANCE["monthly_sorbent_h"] / 720.0) * dt_h

        # Scheduled replacements -- health-threshold triggered
        # Sorbent: partial refresh at 85% (replace 20% of beds), full replace at 70%
        if self.component_health["sorbent"] < 0.70:
            n_beds = FACILITY["sorbent_beds"]
            self.sorbent_replaced_count += n_beds
            self.cumulative_sorbent_cost += n_beds * CAMPAIGN["sorbent_repl_cost_per_bed"]
            self.component_health["sorbent"] = 1.0
            self.maintenance_log.append(
                (year_now, "sorbent",
                 "FULL REPLACEMENT: %d sorbent beds ($%.1fM, %.0fh)" % (
                     n_beds, n_beds * CAMPAIGN["sorbent_repl_cost_per_bed"] / 1e6,
                     n_beds * MAINTENANCE["sorbent_replace_h"])))
            self.maintenance_h += n_beds * MAINTENANCE["sorbent_replace_h"]
        elif self.component_health["sorbent"] < 0.85:
            # Partial refresh: replace 20% of most degraded beds
            n_refresh = max(1, FACILITY["sorbent_beds"] // 5)
            self.sorbent_replaced_count += n_refresh
            self.cumulative_sorbent_cost += n_refresh * CAMPAIGN["sorbent_repl_cost_per_bed"]
            # Blended health: 80% old at current health + 20% new at 100%
            old_h = self.component_health["sorbent"]
            self.component_health["sorbent"] = 0.8 * old_h + 0.2 * 1.0
            self.maintenance_log.append(
                (year_now, "sorbent",
                 "PARTIAL REFRESH: %d beds ($%.1fM, %.0fh), health %.1f%% -> %.1f%%" % (
                     n_refresh, n_refresh * CAMPAIGN["sorbent_repl_cost_per_bed"] / 1e6,
                     n_refresh * MAINTENANCE["sorbent_replace_h"],
                     old_h * 100, self.component_health["sorbent"] * 100)))
            self.maintenance_h += n_refresh * MAINTENANCE["sorbent_replace_h"]

        # Battery: replace when health drops below 80%
        if self.component_health["battery"] < 0.80:
            n_racks = DIMS["battery_modules"]
            self.battery_replaced_count += n_racks
            self.cumulative_battery_cost += CAMPAIGN["battery_repl_cost"]
            self.component_health["battery"] = 1.0
            self.maintenance_log.append(
                (year_now, "battery",
                 "SCHEDULED: replaced %d battery racks ($%.1fM, %.0fh)" % (
                     n_racks, CAMPAIGN["battery_repl_cost"] / 1e6,
                     n_racks * MAINTENANCE["battery_replace_h"])))
            self.maintenance_h += n_racks * MAINTENANCE["battery_replace_h"]

        # Annual overhaul
        if int(prev_years) < int(year_now):
            self.maintenance_h += MAINTENANCE["annual_overhaul_h"]
            # Annual maintenance restores health on maintained components
            for cname in ("fans", "regen_units", "vacuum_pumps", "compressors",
                          "wind_turbines", "geothermal", "control_system",
                          "solar_pv", "solar_thermal", "thermal_store",
                          "co2_tanks", "battery"):
                self.component_health[cname] = min(1.0,
                    self.component_health[cname] + 0.08)
            self.maintenance_log.append(
                (year_now, "ALL", "ANNUAL OVERHAUL (%.0fh)" % MAINTENANCE["annual_overhaul_h"]))

    @property
    def degradation_factor(self):
        """Overall capture efficiency multiplier from component degradation (0..1)."""
        h = self.component_health
        fan_eff = h.get("fans", 1.0)
        sorb_eff = h.get("sorbent", 1.0)
        regen_eff = h.get("regen_units", 1.0)
        vac_eff = h.get("vacuum_pumps", 1.0)
        comp_eff = h.get("compressors", 1.0)
        # Weighted: sorbent capacity is the biggest factor
        return clamp(0.30 * sorb_eff + 0.20 * fan_eff + 0.20 * regen_eff +
                     0.15 * vac_eff + 0.15 * comp_eff)

    @property
    def energy_degradation_factor(self):
        """Renewable energy output multiplier from degradation (0..1)."""
        h = self.component_health
        return clamp(0.40 * h.get("solar_pv", 1.0) +
                     0.30 * h.get("solar_thermal", 1.0) +
                     0.20 * h.get("wind_turbines", 1.0) +
                     0.10 * h.get("geothermal", 1.0))

    @property
    def availability(self):
        """Fraction of time the facility is operational (not down for repairs)."""
        total_h = max(1.0, self.elapsed_years * 8760.0)
        return clamp(1.0 - self.downtime_h / total_h)

    @property
    def maintenance_summary(self):
        """Summary of maintenance and failures for display."""
        total_failures = sum(self.component_failures.values())
        total_prevented = sum(self.prevented_failures.values())
        return {
            "total_failures": total_failures,
            "total_prevented": total_prevented,
            "total_repairs": sum(self.component_repairs.values()),
            "downtime_h": self.downtime_h,
            "maintenance_h": self.maintenance_h,
            "availability": self.availability,
            "sorbent_replaced": self.sorbent_replaced_count,
            "battery_replaced": self.battery_replaced_count,
            "log_entries": len(self.maintenance_log),
            "prevention_rate": (total_prevented / max(1, total_prevented + total_failures)),
        }

    def update(self, dt, sun, wind_ms):
        """Advance one control tick. dt is campaign seconds (time-warped).
        RENEWABLES-FIRST: solar thermal -> regeneration directly. Solar PV ->
        fans + compressors. Surplus charges battery + thermal storage. Night:
        thermal storage + geothermal + battery carry the facility."""
        dt_h = dt / 3600.0
        dt_min = dt / 60.0
        self.total_s += dt

        # ---- degradation & maintenance ----
        self.update_degradation(dt_h)
        edeg = self.energy_degradation_factor
        cdeg = self.degradation_factor

        # ---- renewable generation (with degradation) ----
        pv_kw = solar_pv_kw(sun) * edeg
        th_kw = solar_thermal_kw(sun) * edeg
        wind_kw_ = wind_kw(wind_ms) * edeg
        geo_kw = GEO_THERMAL_KW * self.component_health.get("geothermal", 1.0)

        # ---- advance sorbent bed phases ----
        self.update_bed_phases(dt_min)

        # ---- count beds in each phase ----
        n_capture = sum(1 for b in self.bed_phases if b["state"] == "capture")
        n_regen = sum(1 for b in self.bed_phases if b["state"] == "regen")
        n_cool = sum(1 for b in self.bed_phases if b["state"] == "cool")

        # ---- SYNERGY: adjust active beds based on available energy ----
        is_night = sun < 0.03

        # Thermal available: solar thermal (direct) + geothermal + thermal storage
        thermal_avail_kw = th_kw + geo_kw
        if is_night or th_kw < regen_thermal_kw_needed(CAPTURE_CTRL["capture_rate_target_t_h"]) * 0.5:
            # draw from thermal storage
            store_avail = max(0.0, (self.thermal_frac - THERMAL_STORE["min_frac"]) *
                              self.thermal_kwh / max(dt_h, 1e-6))
            thermal_avail_kw += store_avail * SYNERGY["night_thermal_frac"]

        # Electrical available: solar PV + wind + battery
        elec_avail_kw = pv_kw + wind_kw_
        if is_night or pv_kw < elec_kw_needed(CAPTURE_CTRL["capture_rate_target_t_h"]) * 0.5:
            batt_avail = max(0.0, (self.soc - ELEC["soc_min"]) *
                             self.batt_kwh / max(dt_h, 1e-6))
            elec_avail_kw += batt_avail * SYNERGY["night_batt_frac"]

        # ---- determine how many beds can run ----
        # Each active bed needs: thermal for regen + electrical for fans/vacuum
        # Beds in capture phase need fan power; beds in regen need thermal + vacuum
        # Estimate per-bed power needs
        capture_rate_per_bed_t_h = (DIMS["sorbent_t_per_bed"] * DIMS["sorbent_cap_kg_kg"] /
                                    SORBENT["cycle_total_min"] * 60.0 / 1000.0)
        thermal_per_bed_kw = capture_rate_per_bed_t_h * ENERGY["regen_thermal_kwh_t"]
        elec_per_bed_kw = capture_rate_per_bed_t_h * (ENERGY["fan_elec_kwh_t"] +
                            ENERGY["vacuum_elec_kwh_t"] +
                            ENERGY["compress_elec_kwh_t"] +
                            ENERGY["aux_elec_kwh_t"])

        # How many beds can we sustain?
        beds_by_thermal = int(thermal_avail_kw / max(1.0, thermal_per_bed_kw))
        beds_by_elec = int(elec_avail_kw / max(1.0, elec_per_bed_kw))
        beds_possible = min(beds_by_thermal, beds_by_elec)
        self.beds_active = clamp(beds_possible,
                                  CAPTURE_CTRL["beds_active_min"],
                                  CAPTURE_CTRL["beds_active_max"])

        # ---- capture rate ----
        self.fan_rpm = 80.0 + 90.0 * clamp(elec_avail_kw / elec_kw_needed(
            CAPTURE_CTRL["capture_rate_target_t_h"]))
        self.capture_rate_kg_s = co2_capture_rate_kg_per_s(self.beds_active, self.fan_rpm) * cdeg
        capture_t_h = self.capture_rate_kg_s * 3.6
        co2_captured_t = capture_t_h * dt_h

        # ---- energy consumption ----
        thermal_needed_kw = regen_thermal_kw_needed(capture_t_h)
        elec_needed_kw = elec_kw_needed(capture_t_h)

        # ---- thermal energy routing ----
        # Direct solar thermal + geothermal -> regeneration
        thermal_direct_kw = min(thermal_needed_kw, th_kw + geo_kw)
        thermal_from_store_kw = max(0.0, thermal_needed_kw - thermal_direct_kw)

        # Charge thermal storage from surplus solar thermal
        thermal_surplus_kw = max(0.0, th_kw + geo_kw - thermal_needed_kw)
        thermal_to_store_kwh = thermal_surplus_kw * dt_h * SYNERGY["thermal_charge"]
        thermal_from_store_kwh = thermal_from_store_kw * dt_h

        self.route_thermal(thermal_to_store_kwh - thermal_from_store_kwh)

        # ---- electrical energy routing ----
        # Direct PV + wind -> facility loads
        elec_direct_kw = min(elec_needed_kw, pv_kw + wind_kw_)
        elec_from_batt_kw = max(0.0, elec_needed_kw - elec_direct_kw)

        # Charge battery from surplus PV + wind
        elec_surplus_kw = max(0.0, pv_kw + wind_kw_ - elec_needed_kw)
        elec_to_batt_kwh = elec_surplus_kw * dt_h * SYNERGY["batt_charge"]
        elec_from_batt_kwh = elec_from_batt_kw * dt_h

        self.route_battery(elec_to_batt_kwh - elec_from_batt_kwh)

        # ---- CO2 storage + pipeline ----
        self.co2_storage_t += co2_captured_t
        # Pipeline sends CO2 to sequestration
        pipeline_t = min(CO2_STORE["pipeline_rate_t_h"] * dt_h,
                         max(0.0, self.co2_storage_t - CO2_STORE["capacity_t"] * 0.05))
        self.co2_storage_t -= pipeline_t
        self.co2_storage_t = min(self.co2_storage_t, CO2_STORE["capacity_t"])

        # ---- regen temperature model ----
        if n_regen > 0:
            target = THERM["regen_target_c"]
            self.regen_temp_c += (target - self.regen_temp_c) * min(1.0, dt * 0.3)
        else:
            self.regen_temp_c += (THERM["ambient_c"] - self.regen_temp_c) * min(1.0, dt * 0.1)

        # ---- tallies ----
        self.co2_captured_t += co2_captured_t
        self.co2_sequestered_t += pipeline_t
        self.solar_pv_kwh += pv_kw * dt_h
        self.solar_th_kwh += th_kw * dt_h
        self.wind_kwh += wind_kw_ * dt_h
        self.geo_kwh += geo_kw * dt_h
        self.thermal_used_kwh += thermal_needed_kw * dt_h
        self.elec_used_kwh += elec_needed_kw * dt_h

        # ---- 15-year operational tracking ----
        # Water consumption (cooling + sorbent humidification + cleaning)
        self.water_used_m3 += co2_captured_t * CAMPAIGN["water_m3_per_t_co2"]
        # CO2 purity degrades with sorbent age (amine degradation introduces impurities)
        sorb_h = self.component_health.get("sorbent", 1.0)
        self.co2_purity = clamp(0.9985 * sorb_h + 0.001, 0.95, 0.9995)
        # Track peak capture rate
        if capture_t_h > self.peak_capture_t_h:
            self.peak_capture_t_h = capture_t_h
        # Track active capture hours (for capacity factor)
        if co2_captured_t > 0:
            self.capture_hours += dt_h
        # Cumulative OPEX (annual costs prorated to dt_h)
        annual_opex_rate = (CAMPAIGN["labor_cost_per_year"] +
                           CAMPAIGN["maint_cost_per_year"] +
                           CAMPAIGN["insurance_per_year"] +
                           CAMPAIGN["land_lease_per_year"]) / 8760.0
        self.cumulative_opex += annual_opex_rate * dt_h

        self.flow = {"solar_pv": pv_kw, "solar_th": th_kw, "wind": wind_kw_,
                     "geo": geo_kw, "thermal_store": thermal_from_store_kw,
                     "regen_thermal": thermal_needed_kw,
                     "fan_elec": capture_t_h * ENERGY["fan_elec_kwh_t"],
                     "vacuum_elec": capture_t_h * ENERGY["vacuum_elec_kwh_t"],
                     "compress_elec": capture_t_h * ENERGY["compress_elec_kwh_t"],
                     "aux_elec": capture_t_h * ENERGY["aux_elec_kwh_t"],
                     "batt_net": (elec_to_batt_kwh - elec_from_batt_kwh) / max(dt_h, 1e-6),
                     "thermal_net": (thermal_to_store_kwh - thermal_from_store_kwh) / max(dt_h, 1e-6),
                     "co2_capture": capture_t_h, "co2_sequester": pipeline_t / max(dt_h, 1e-6)}

        # ---- mode label ----
        if is_night and self.thermal_frac < 0.2 and self.soc < 0.2:
            self.mode = "LOW POWER -- REDUCED CAPTURE"
        elif is_night:
            self.mode = "NIGHT OPERATION"
        elif sun > 0.7 and self.thermal_frac > 0.8:
            self.mode = "PEAK SOLAR -- FULL CAPTURE"
        elif sun > 0.3:
            self.mode = "SOLAR CAPTURE"
        elif wind_kw_ > WIND_RATED_KW * 0.5:
            self.mode = "WIND-ASSISTED CAPTURE"
        elif self.thermal_frac > 0.5 and self.soc > 0.3:
            self.mode = "STORED ENERGY CAPTURE"
        else:
            self.mode = "GEOTHERMAL BASELOAD"


# =============================================================================
# SECTION 7 -- OPERATION WORLD (annual campaign, day/night, weather)
# =============================================================================

class Campaign:
    """Tracks campaign time (time-warped), the day/night solar cycle, wind,
    and campaign progress. Supports multi-year longevity simulation."""

    TIME_WARP = [60.0, 300.0, 1800.0, 7200.0, 21600.0, 86400.0, 432000.0, 1314000.0]

    def __init__(self, years=CAMPAIGN["years"]):
        self.warp_i = 4
        self.elapsed_h = 0.0
        self.weather = 0.8
        self.wind_ms = 8.0
        self.sun = 0.0
        self.finished = False
        self.duration_years = years
        self.duration_h = years * CAMPAIGN["duration_days"] * 24.0
        self.year_history = []  # snapshots per year for longevity tracking
        self.week_history = []  # snapshots per week for weekly ops panel
        self.time_jumping = False  # when True, we're viewing a past snapshot
        self.jump_index = -1  # index into week_history (-1 = live)

    @property
    def warp(self):
        return self.TIME_WARP[self.warp_i]

    @property
    def hour_of_day(self):
        return self.elapsed_h % 24.0

    @property
    def day(self):
        return int(self.elapsed_h // 24.0) + 1

    @property
    def year_frac(self):
        return clamp(self.elapsed_h / self.duration_h)

    @property
    def year_num(self):
        return int(self.elapsed_h / (CAMPAIGN["duration_days"] * 24.0)) + 1

    @property
    def total_years(self):
        return self.duration_years

    @property
    def week_num(self):
        return int(self.elapsed_h / (7.0 * 24.0)) + 1

    @property
    def total_weeks(self):
        return int(self.duration_h / (7.0 * 24.0))

    def jump_to_week(self, idx, pt):
        """Jump to a past week snapshot (idx = 0-based index into week_history).
        idx = -1 means return to live (current time)."""
        if idx == -1:
            self.time_jumping = False
            self.jump_index = -1
            return
        if 0 <= idx < len(self.week_history):
            self.time_jumping = True
            self.jump_index = idx

    def jump_relative(self, delta, pt):
        """Jump forward/backward by delta weeks in the week_history."""
        if len(self.week_history) == 0:
            return
        if self.jump_index == -1:
            # Currently live -- jump backward from latest
            new_idx = len(self.week_history) - 1 + delta
        else:
            new_idx = self.jump_index + delta
        new_idx = max(-1, min(new_idx, len(self.week_history) - 1))
        if new_idx == -1:
            self.jump_to_week(-1, pt)
        else:
            self.jump_to_week(new_idx, pt)

    def cycle_warp(self, d):
        self.warp_i = clamp(self.warp_i + d, 0, len(self.TIME_WARP) - 1)

    def update(self, dt_real, pt):
        """dt_real: real wall seconds. Returns campaign-seconds advanced."""
        dt_c = dt_real * self.warp
        self.elapsed_h += dt_c / 3600.0
        t = self.elapsed_h
        # slowly drifting weather + wind
        self.weather = clamp(0.72 + 0.26 * math.sin(t / 15.3) + 0.10 * math.sin(t / 4.1))
        self.wind_ms = clamp(6.0 + 4.0 * math.sin(t / 7.7) + 3.0 * math.sin(t / 2.3),
                             0.0, 25.0)
        self.sun = sun_factor(self.hour_of_day) * (0.35 + 0.65 * self.weather)
        if self.year_frac >= 1.0:
            self.finished = True
        # Snapshot at year boundaries for longevity tracking
        cy = self.year_num
        if cy > len(self.year_history):
            # Annual cost calculation
            annual_opex = (CAMPAIGN["labor_cost_per_year"] +
                           CAMPAIGN["maint_cost_per_year"] +
                           CAMPAIGN["insurance_per_year"] +
                           CAMPAIGN["land_lease_per_year"])
            # Add replacement costs incurred this year
            prev_snap = self.year_history[-1] if self.year_history else None
            prev_sorb_rep = prev_snap["sorbent_replaced"] if prev_snap else 0
            prev_batt_rep = prev_snap["battery_replaced"] if prev_snap else 0
            sorb_repl_this = pt.sorbent_replaced_count - prev_sorb_rep
            batt_repl_this = pt.battery_replaced_count - prev_batt_rep
            sorb_cost = sorb_repl_this * CAMPAIGN["sorbent_repl_cost_per_bed"]
            batt_cost = batt_repl_this * CAMPAIGN["battery_repl_cost"]
            annual_total_cost = annual_opex + sorb_cost + batt_cost

            # Water consumption
            annual_water_m3 = pt.co2_captured_t * CAMPAIGN["water_m3_per_t_co2"]
            if prev_snap:
                annual_water_m3 -= prev_snap["cumulative_water_m3"]

            # Capacity factor: actual vs theoretical max (peak rate × 8760h)
            theoretical_max_t = CAPTURE_CTRL["capture_rate_target_t_h"] * 8760.0
            prev_co2 = prev_snap["co2_captured_t"] if prev_snap else 0.0
            actual_t = pt.co2_captured_t - prev_co2
            capacity_factor = actual_t / theoretical_max_t if theoretical_max_t > 0 else 0.0

            # Cost per tonne this year
            cost_per_t = annual_total_cost / actual_t if actual_t > 1 else 999.0

            # Net CO2 benefit (captured - embodied amortized)
            embodied_amortized = CAMPAIGN["embodied_co2_t"] / self.duration_years
            net_co2 = actual_t - embodied_amortized

            self.year_history.append({
                "year": cy,
                "elapsed_h": self.elapsed_h,
                "co2_captured_t": pt.co2_captured_t,
                "co2_sequestered_t": pt.co2_sequestered_t,
                "component_health": dict(pt.component_health),
                "failures": dict(pt.component_failures),
                "downtime_h": pt.downtime_h,
                "maintenance_h": pt.maintenance_h,
                "availability": pt.availability,
                "annual_cost": annual_total_cost,
                "cost_per_t": cost_per_t,
                "annual_water_m3": annual_water_m3,
                "cumulative_water_m3": pt.co2_captured_t * CAMPAIGN["water_m3_per_t_co2"],
                "capacity_factor": capacity_factor,
                "net_co2_benefit": net_co2,
                "sorbent_replaced": pt.sorbent_replaced_count,
                "battery_replaced": pt.battery_replaced_count,
                "degradation_factor": pt.degradation_factor,
                "energy_degradation": pt.energy_degradation_factor,
            })

        # --- Weekly snapshot for ops panel ---
        cw = self.week_num
        if cw > len(self.week_history):
            prev_week = self.week_history[-1] if self.week_history else None
            prev_co2_w = prev_week["co2_captured_t"] if prev_week else 0.0
            prev_cost_w = prev_week["cumulative_cost"] if prev_week else 0.0
            prev_water_w = prev_week["cumulative_water_m3"] if prev_week else 0.0
            prev_solar_w = prev_week["cumulative_solar_kwh"] if prev_week else 0.0
            prev_th_w = prev_week["cumulative_thermal_kwh"] if prev_week else 0.0
            prev_wind_w = prev_week["cumulative_wind_kwh"] if prev_week else 0.0
            prev_geo_w = prev_week["cumulative_geo_kwh"] if prev_week else 0.0
            week_co2 = pt.co2_captured_t - prev_co2_w
            total_cost = pt.cumulative_opex + pt.cumulative_sorbent_cost + pt.cumulative_battery_cost
            week_cost = total_cost - prev_cost_w
            week_water = pt.water_used_m3 - prev_water_w
            # Weekly capacity factor: actual capture vs theoretical peak for 1 week
            week_max_t = CAPTURE_CTRL["capture_rate_target_t_h"] * 24.0 * 7.0
            week_cf = week_co2 / week_max_t if week_max_t > 0 else 0.0
            week_cost_per_t = week_cost / max(1.0, week_co2) if week_co2 > 1 else 0.0
            self.week_history.append({
                "week": cw,
                "year": self.year_num,
                "elapsed_h": self.elapsed_h,
                "co2_captured_t": pt.co2_captured_t,
                "week_co2_t": week_co2,
                "co2_sequestered_t": pt.co2_sequestered_t,
                "co2_storage_t": pt.co2_storage_t,
                "co2_storage_frac": pt.co2_storage_frac,
                "cumulative_cost": total_cost,
                "week_cost": week_cost,
                "week_cost_per_t": week_cost_per_t,
                "cumulative_water_m3": pt.water_used_m3,
                "week_water_m3": week_water,
                "capacity_factor": week_cf,
                "beds_active": pt.beds_active,
                "fan_rpm": pt.fan_rpm,
                "co2_purity": pt.co2_purity,
                "availability": pt.availability,
                "cumulative_solar_kwh": pt.solar_pv_kwh,
                "cumulative_thermal_kwh": pt.solar_th_kwh,
                "cumulative_wind_kwh": pt.wind_kwh,
                "cumulative_geo_kwh": pt.geo_kwh,
                "week_solar_kwh": pt.solar_pv_kwh - prev_solar_w,
                "week_thermal_kwh": pt.solar_th_kwh - prev_th_w,
                "week_wind_kwh": pt.wind_kwh - prev_wind_w,
                "week_geo_kwh": pt.geo_kwh - prev_geo_w,
                "sorbent_replaced": pt.sorbent_replaced_count,
                "battery_replaced": pt.battery_replaced_count,
            })
        return dt_c

    @property
    def progress(self):
        return self.year_frac


def sky_colors(sun):
    """Blend day/night sky by the sun intensity for the OPERATION horizon."""
    t = clamp(sun * 1.6)
    top = _mix(C_SKY_NIGHT1, C_SKY1, t)
    bot = _mix(C_SKY_NIGHT2, C_SKY2, t)
    return top, bot

# =============================================================================
# SECTION 8 -- FULL INFORMATIONAL SPECIFICATION (about / detail / honesty)
# =============================================================================

def build_info_sections():
    return [
        ("WHAT IS THIS?  (plain English)", [
            "This is a 3D model of a Direct Air Capture (DAC) giga-plant --",
            "a multi-story facility that removes CO2 from the atmosphere.",
            "",
            "Think of it as a giant air filter: huge fans pull outside air",
            "through a special material that grabs CO2. When the material is",
            "full, it gets heated to release the CO2, which is then squeezed",
            "into liquid and piped deep underground for permanent storage.",
            "",
            "The whole plant runs on free, clean energy from the sun, wind,",
            "and geothermal heat from deep underground -- no fossil fuels.",
            "It captures about %.0f million tonnes of CO2 per year, equivalent" % (
                FACILITY["capture_t_year"] / 1e6),
            "to taking ~%.0f million cars off the road." % (
                FACILITY["capture_t_year"] / 4.6 / 1e6),
            "",
            "This is a %d-story giga-facility with %d contactors, designed" % (
                DIMS["contactor_stories"], FACILITY["air_contactors"]),
            "to minimize the number of plants needed globally.",
            "",
            "Use TAB to switch between FACILITY (whole plant), CAPTURE",
            "(close-up of one unit), and OPERATION (live simulation).",
            "Click on any part to see what it does. Press H for controls.",
        ]),
        ("THE FACILITY", [
            "A multi-story Direct Air Capture giga-plant removing",
            "%.1f Mt CO2/year from the atmosphere." % (FACILITY["capture_t_year"] / 1e6),
            "That's equivalent to ~%.1f million cars' emissions." % (
                FACILITY["capture_t_year"] / 4.6 / 1e6),
            "Site: %.0f hectares (%.1f km2), %d staff, runs 24/7." % (
                FACILITY["land_area_m2"] / 1e4, FACILITY["land_area_m2"] / 1e6,
                FACILITY["staff"]),
            "Target: %.0f t CO2/day (%.0f t/h average)." % (
                FACILITY["capture_t_day"], FACILITY["capture_t_hour"]),
            "Multi-story: %d levels x %d contactors = %d total." % (
                DIMS["contactor_stories"],
                DIMS["contactor_rows"] * DIMS["contactor_per_row"],
                FACILITY["air_contactors"]),
            "CO2 in air today: %.0f ppm (0.04%% -- but enough to trap heat)." % FACILITY["co2_atm_ppm"],
        ]),
        ("HOW IT CAPTURES  (the basic idea)", [
            "%d giant fan units (across %d stories) pull outside air" % (
                FACILITY["air_contactors"], DIMS["contactor_stories"]),
            "through a special filter material that grabs CO2 but lets",
            "other air pass through.",
            "Each contactor: 20 m W x 12 m H x 4 m D (cross-flow slab).",
            "8 fans per unit (3.5 m dia, CFRP blades, Ti-6Al-4V hub).",
            "Filter bed: 20 m x 10 m x 1.5 m honeycomb monolith (300 m3).",
            "Regen chamber: 12 m x 8 m x 10 m, heats to 100 C under vacuum.",
            "CO2 is only 0.04%% of air, but the filter grabs it efficiently.",
            "Each filter holds %.0f t of material and captures %.1f t CO2 per cycle." % (
                DIMS["sorbent_t_per_bed"],
                DIMS["sorbent_t_per_bed"] * DIMS["sorbent_cap_kg_kg"]),
            "",
            "BLUEPRINT SCALE: capture view 1 unit = 8 m.",
            "All dimensions are real-world metres for construction reference.",
        ]),
        ("THE CAPTURE CYCLE  (grab -> release -> repeat)", [
            "The filter works in 3 repeating steps:",
            "",
            "1. CAPTURE (%.0f min): Fans push air through the filter. CO2 sticks." % SORBENT["cycle_capture_min"],
            "2. RELEASE (%.0f min): Chamber sealed, heated to %.0f C. CO2 lets go." % (
                SORBENT["cycle_regen_min"], DIMS["regen_temp_c"]),
            "   A vacuum pump pulls the released CO2 out.",
            "3. COOL (%.0f min): Filter cools down before starting again." % SORBENT["cycle_cool_min"],
            "",
            "Full cycle: %.0f min = %.1f times per day per filter." % (
                SORBENT["cycle_total_min"], SORBENT["cycles_per_day"]),
            "%d filters take turns so capture never stops -- at any moment," % FACILITY["sorbent_beds"],
            "~%d are grabbing CO2, ~%d are releasing it, ~%d are cooling." % (
                int(FACILITY["sorbent_beds"] * SORBENT["cycle_capture_min"] / SORBENT["cycle_total_min"]),
                int(FACILITY["sorbent_beds"] * SORBENT["cycle_regen_min"] / SORBENT["cycle_total_min"]),
                int(FACILITY["sorbent_beds"] * SORBENT["cycle_cool_min"] / SORBENT["cycle_total_min"])),
        ]),
        ("CLEAN ENERGY  (all renewable, all free)", [
            "The plant runs entirely on clean energy -- no fossil fuels:",
            "",
            "SOLAR MIRRORS: %.0f m2 of curved mirrors, %.0f MW heat peak." % (
                DIMS["trough_aperture_m2"], SOLAR_TH_PEAK_KW / 1000),
            "  Focus sunlight to make heat for releasing CO2 from the filter.",
            "SOLAR PANELS: %.0f m2 of panels, %.0f MW electricity peak." % (
                DIMS["solar_pv_m2"], SOLAR_PV_PEAK_KW / 1000),
            "  Make electricity to run the fans, compressors, and controls.",
            "WIND: %d x %.0f MW turbines = %.0f MW total." % (
                DIMS["wind_turbines"], DIMS["turbine_rated_mw"],
                DIMS["wind_turbines"] * DIMS["turbine_rated_mw"]),
            "GEOTHERMAL: %d wells, %.0f MW heat -- ALWAYS ON, day and night." % (
                DIMS["geo_wells"], DIMS["geo_mw_thermal"]),
            "  Pulls heat from deep underground. Critical for nighttime.",
        ]),
        ("ENERGY STORAGE  (keeping it running at night)", [
            "When the sun goes down, stored energy keeps the plant running:",
            "",
            "HEAT BATTERY: %.0f MWh of molten salt (hot %.0f C / cold %.0f C)." % (
                DIMS["thermal_storage_mwh"],
                DIMS["salt_hot_temp_c"], DIMS["salt_cold_temp_c"]),
            "  Stores daytime heat in hot salt. Used to release CO2 at night.",
            "BATTERY: %.0f MWh of lithium-ion batteries." % DIMS["battery_mwh"],
            "  Stores extra solar/wind electricity for nighttime use.",
            "",
            "The plant NEVER STOPS -- geothermal + storage carry it through.",
        ]),
        ("CO2 PROCESSING  (from gas to underground storage)", [
            "After the CO2 is released from the filter:",
            "",
            "1. COMPRESS: %d compressors squeeze CO2 gas into liquid at %.0f bar." % (
                FACILITY["compressors"], DIMS["storage_bar"]),
            "2. STORE: %d tanks hold %.0f t each = %.0f t total buffer." % (
                DIMS["storage_tanks"], DIMS["storage_capacity_t"],
                DIMS["storage_tanks"] * DIMS["storage_capacity_t"]),
            "3. PIPE: %.0f t/h sent to geological sequestration" % CO2_STORE["pipeline_rate_t_h"],
            "   (pumped deep underground into rock formations -- permanent).",
        ]),
        ("PLANT LAYOUT  (how things are arranged)", [
            "The plant flows from south to north like an assembly line:",
            "",
            "  SOUTH:  Control room + battery + power substation",
            "    (where staff work and electricity is managed)",
            "",
            "  CENTER:  %d air capture fans (%d stories x 80)" % (
                FACILITY["air_contactors"], DIMS["contactor_stories"]),
            "    (the main capture field -- fans pull air through filters)",
            "",
            "  NORTH:  Processing chain (CO2 gets released, compressed, stored)",
            "    z=180m:  %d CO2 release units + %d geothermal wells" % (
                FACILITY["regen_units"], DIMS["geo_wells"]),
            "    z=310m:  %d CO2 compressors + %d cooling towers" % (
                FACILITY["compressors"], DIMS["cooling_towers"]),
            "    z=370m:  %d CO2 storage tanks" % FACILITY["co2_storage_tanks"],
            "    z=420m+: Pipe to underground storage",
            "",
            "  EAST:   Solar panels (%.0f MW)" % (SOLAR_PV_PEAK_KW / 1000),
            "  WEST:   Solar mirrors (%.0f MWth) + heat battery tanks" % (SOLAR_TH_PEAK_KW / 1000),
            "  N/S:    %d wind turbines on north + %d on south edge" % (
                DIMS["wind_turbines"] // 2, DIMS["wind_turbines"] - DIMS["wind_turbines"] // 2),
            "",
            "Pipes connect each stage. Roads allow truck access.",
            "A fence secures the %.0f ha site." % (FACILITY["land_area_m2"] / 1e4),
        ]),
        ("THE CONTROLLER  (smart energy management)", [
            "The plant uses energy as it comes in -- no wasting:",
            "  - Solar heat goes directly to releasing CO2 from filters",
            "  - Solar electricity runs fans and compressors first",
            "  - Extra energy charges the heat battery and battery bank",
            "  - At night: stored heat + geothermal + batteries keep it running",
            "",
            "The controller automatically adjusts how many filters are active",
            "(%d-%d) based on how much energy is available." % (
                CAPTURE_CTRL["beds_active_min"], CAPTURE_CTRL["beds_active_max"]),
            "More energy = more filters running = more CO2 captured.",
        ]),
        ("ENERGY ECONOMICS  (per tonne CO2)", [
            "Thermal: %.0f kWh/t (regeneration heat)." % ENERGY["regen_thermal_kwh_t"],
            "Electrical: %.0f kWh/t (fans %.0f + vacuum %.0f + compress %.0f + aux %.0f)." % (
                ENERGY["total_elec_kwh_t"], ENERGY["fan_elec_kwh_t"],
                ENERGY["vacuum_elec_kwh_t"], ENERGY["compress_elec_kwh_t"],
                ENERGY["aux_elec_kwh_t"]),
            "Total: %.0f kWh/t CO2." % ENERGY["total_kwh_t"],
            "Target cost: ~$%.0f/t CO2 (all-in OPEX)." % CAMPAIGN["cost_per_t_target"],
            "Sorbent: %.0f%% capacity loss/year ($%.0f/t sorbent cost)." % (
                CAMPAIGN["sorbent_replacement_frac"] * 100,
                CAMPAIGN["sorbent_cost_per_t"]),
            "Fixed OPEX: $%.1fM/year (labor $%.1fM + maint $%.1fM + insurance $%.1fM + land $%.1fM)." % (
                (CAMPAIGN["labor_cost_per_year"] + CAMPAIGN["maint_cost_per_year"] +
                 CAMPAIGN["insurance_per_year"] + CAMPAIGN["land_lease_per_year"]) / 1e6,
                CAMPAIGN["labor_cost_per_year"] / 1e6,
                CAMPAIGN["maint_cost_per_year"] / 1e6,
                CAMPAIGN["insurance_per_year"] / 1e6,
                CAMPAIGN["land_lease_per_year"] / 1e6),
            "Water: %.1f m3/t CO2 ($%.2f/m3, closed-loop recovery)." % (
                CAMPAIGN["water_m3_per_t_co2"], CAMPAIGN["water_cost_per_m3"]),
            "Battery replacement: $%.1fM (at ~20 year intervals)." % (
                CAMPAIGN["battery_repl_cost"] / 1e6),
            "Sorbent replacement: $%.0fK/bed (at ~15 year intervals)." % (
                CAMPAIGN["sorbent_repl_cost_per_bed"] / 1e3),
            "Simulated OPEX: ~$1/t CO2 (renewable self-generation, $0 energy, economies of scale).",
        ]),
        ("THE 15-YEAR CAMPAIGN  (single giga-plant)", [
            "Test run: %.0f years of continuous commercial operation." % CAMPAIGN["years"],
            "This giga-plant captures %.1f Mt CO2/year = %.0f Mt over %.0f years." % (
                FACILITY["capture_t_year"] / 1e6,
                FACILITY["capture_t_year"] * CAMPAIGN["years"] / 1e6,
                CAMPAIGN["years"]),
            "Reference: offsets %.0f industrial emitters (%.0f t/year each)." % (
                FACILITY["capture_t_year"] / CAMPAIGN["co2_ref_emit_t_year"],
                CAMPAIGN["co2_ref_emit_t_year"]),
            "Embodied carbon: %.0f t CO2e (amortized over %.0f years)." % (
                CAMPAIGN["embodied_co2_t"], CAMPAIGN["years"]),
            "Net energy cost: $0 (all renewable, self-generated).",
            "The facility runs 24/7/365 with no fuel purchases.",
            "Sorbent replaced ~2x, battery replaced ~1x over 15 years.",
            "CO2 purity target: %.1f%% for geological sequestration." % (
                CAMPAIGN["co2_purity_target"] * 100),
            "",
            "GIGA-PLANT ECONOMICS (15-year lifetime):",
            "  CAPEX: ~$%.1fB ($1000/t-year capacity, 50x scale)" % (
                FACILITY["capture_t_year"] / 1e3),
            "  OPEX: ~$1/t CO2 (labor, maint, sorbent, water, insurance)",
            "  All-in cost: ~$10-30/t (CAPEX amortized + OPEX at 50x scale)",
            "  Total OPEX over 15 yr: ~$%.0fM ($%.0fM/yr)" % (
                (CAMPAIGN["labor_cost_per_year"] + CAMPAIGN["maint_cost_per_year"] +
                 CAMPAIGN["insurance_per_year"] + CAMPAIGN["land_lease_per_year"]) *
                CAMPAIGN["years"] / 1e6,
                (CAMPAIGN["labor_cost_per_year"] + CAMPAIGN["maint_cost_per_year"] +
                 CAMPAIGN["insurance_per_year"] + CAMPAIGN["land_lease_per_year"]) / 1e6),
            "  Sorbent replacement: ~$%.0fM (2x over 15 yr)" % (
                CAMPAIGN["sorbent_repl_cost_per_bed"] * FACILITY["sorbent_beds"] * 2 / 1e6),
            "  Battery replacement: ~$%.1fM (1x over 15 yr)" % (
                CAMPAIGN["battery_repl_cost"] / 1e6),
            "",
            "WHAT ONE GIGA-PLANT ACHIEVES:",
            "  -- Removes %.0f Mt CO2 from the atmosphere (15-year lifetime)" % (
                FACILITY["capture_t_year"] * CAMPAIGN["years"] / 1e6),
            "  -- Equivalent to taking ~%.1fM cars off the road permanently" % (
                FACILITY["capture_t_year"] / 4.6 / 1e6),
            "  -- Offsets %.0f x 500kt/year industrial emitters" % (
                FACILITY["capture_t_year"] / CAMPAIGN["co2_ref_emit_t_year"]),
            "  -- Runs on 100%% renewable energy (no fossil fuels, ever)",
            "  -- Operates with %d staff, highly automated" % FACILITY["staff"],
            "  -- 99.9%%+ availability with predictive maintenance",
            "  -- Multi-story design: %d levels on just %.0f ha of land" % (
                DIMS["contactor_stories"], FACILITY["land_area_m2"] / 1e4),
        ]),
        ("HONEST PHYSICS  (what is and isn't claimed)", [
            "This is NOT free energy. Solar, wind and geothermal are EXTERNAL",
            "energy sources -- that is what makes the process work.",
            "CO2 at 420 ppm is DILUTE: only ~0.68 g per m3 of air, so enormous",
            "air volumes must be moved. The energy cost is REAL (~1250 kWh/t).",
            "The sorbent cycle is LOSSY: regeneration uses more energy than",
            "capture releases. The renewables make up the difference.",
            "This is NOT a perpetual motion machine -- it is an industrial",
            "process powered by free external energy (sun, wind, geothermal).",
            "Cost targets (~$100/t by 2035) are ambitious but consistent with",
            "published DAC scaling studies (Climeworks, Carbon Engineering,",
            "Heirloom, Spiritus). Current costs are $400-1000/t (first-of-a-kind).",
            "This model simulates ~$1/t OPEX because energy is self-generated",
            "(no fuel purchases) and the giga-plant scale (50x) achieves",
            "economies of scale. Real-world CAPEX amortization would add",
            "$10-30/t, bringing all-in cost to ~$10-30/t at this scale.",
            "Learning rate: ~20%% cost reduction per doubling of capacity.",
            "Sorbent lifetime: ~15 years (amine on silica/MOF supports).",
            "Design life: 20-30 years. Water use: minimal (closed-loop).",
        ]),
        ("BUILD DIMENSIONS  (blueprint-scale reference)", [
            "All dimensions are real-world metres for construction reference.",
            "Capture view scale: 1 display unit = 8 m.",
            "",
            "AIR CONTACTOR (per unit, %d total across %d stories):" % (
                FACILITY["air_contactors"], DIMS["contactor_stories"]),
            "  Frame: %.0f m W x %.0f m H x %.0f m D (cross-flow slab)" % (
                DIMS["contactor_w_m"], DIMS["contactor_h_m"], DIMS["contactor_d_m"]),
            "  Frame thickness: %.1f m galvanized steel structural sections" % DIMS["contactor_frame_d_m"],
            "  Foundation: concrete pad, %.0f m x %.0f m x 1.0 m" % (
                DIMS["contactor_w_m"] + 2.0, DIMS["contactor_d_m"] + 2.0),
            "  Support columns: 4 x 0.8 m dia steel, full height",
            "  Walkway: catwalk on top, 1.0 m wide, with safety railings",
            "  Access ladder: 8 rungs, east side of frame",
            "",
            "FAN ARRAY (8 per contactor, %d total):" % (
                FACILITY["air_contactors"] * DIMS["contactor_fans"]),
            "  Fans: %d x %.1f m diameter, %d-blade CFRP" % (
                DIMS["contactor_fans"], DIMS["fan_d_m"], DIMS["fan_blades"]),
            "  Hub: %.1f m dia Ti-6Al-4V, direct-drive magnetic bearings" % DIMS["fan_hub_d_m"],
            "  Layout: 4 wide x 2 tall grid",
            "  Spacing: %.1f m x %.1f m center-to-center" % (
                DIMS["contactor_w_m"]/4, DIMS["contactor_h_m"]/2),
            "  Shroud: annulus ring, 0.2 m deep around each fan",
            "",
            "SORBENT BED (%d total):" % FACILITY["sorbent_beds"],
            "  Bed: %.0f m W x %.0f m H x %.1f m D honeycomb monolith" % (
                DIMS["bed_w_m"], DIMS["bed_h_m"], DIMS["bed_d_m"]),
            "  Volume: %.0f m3 per bed" % (
                DIMS["bed_w_m"] * DIMS["bed_h_m"] * DIMS["bed_d_m"]),
            "  Sorbent: %.0f t PEI on silica/MOF (%.3f kg CO2/kg)" % (
                DIMS["sorbent_t_per_bed"], DIMS["sorbent_cap_kg_kg"]),
            "  Layers: %d honeycomb channels, 1 mm channels, 450 CPSI" % DIMS["sorbent_layers"],
            "",
            "REGENERATION CHAMBER (16 total):",
            "  Chamber: %.0f m W x %.0f m H x %.0f m D vacuum vessel" % (
                DIMS["regen_w_m"], DIMS["regen_h_m"], DIMS["regen_d_m"]),
            "  Volume: %.0f m3, SS 316L construction" % (
                DIMS["regen_w_m"] * DIMS["regen_h_m"] * DIMS["regen_d_m"]),
            "  Insulation: %.1f m ceramic fiber blanket" % DIMS["regen_insul_d_m"],
            "  Heaters: %d ceramic IR rows, %.0f C operating temp" % (
                DIMS["regen_heater_rows"], DIMS["regen_temp_c"]),
            "  Vacuum: %.1f m dia x %.1f m H dry screw pump" % (
                DIMS["regen_vacuum_d_m"], DIMS["regen_vacuum_h_m"]),
            "",
            "PIPING & VALVES:",
            "  Manifold: %d rows, %.0f mm dia SS 316L" % (
                DIMS["manifold_rows"], DIMS["manifold_d_m"] * 1000),
            "  Valves: 3x 0.8 m dia pneumatic (intake/exhaust/CO2 output)",
            "  Connecting duct: 0.8 m dia, sorbent to regen chamber",
            "  CO2 output pipe: %.0f mm dia, trace-heated" % (
                DIMS["manifold_d_m"] * 500),
            "",
            "INTAKE PLENUM:",
            "  Plenum: %.0f m W x %.0f m H x %.0f m D" % (
                DIMS["contactor_w_m"], DIMS["contactor_h_m"], DIMS["plenum_d_m"]),
            "  Volume: %.0f m3, stainless steel construction" % (
                DIMS["contactor_w_m"] * DIMS["contactor_h_m"] * DIMS["plenum_d_m"]),
        ]),
        ("COST BREAKDOWN  (detailed OPEX model)", [
            "ANNUAL FIXED COSTS:",
            "  Labor: $%.1fM/year (%d staff, highly automated)" % (
                CAMPAIGN["labor_cost_per_year"] / 1e6, FACILITY["staff"]),
            "  Maintenance: $%.1fM/year (predictive, robotic)" % (
                CAMPAIGN["maint_cost_per_year"] / 1e6),
            "  Insurance + permits: $%.1fM/year" % (
                CAMPAIGN["insurance_per_year"] / 1e6),
            "  Land lease: $%.1fM/year (%.0f ha, remote)" % (
                CAMPAIGN["land_lease_per_year"] / 1e6,
                FACILITY["land_area_m2"] / 1e4),
            "  Total fixed: $%.1fM/year = $%.2f/t at %.1f Mt/year" % (
                (CAMPAIGN["labor_cost_per_year"] + CAMPAIGN["maint_cost_per_year"] +
                 CAMPAIGN["insurance_per_year"] + CAMPAIGN["land_lease_per_year"]) / 1e6,
                (CAMPAIGN["labor_cost_per_year"] + CAMPAIGN["maint_cost_per_year"] +
                 CAMPAIGN["insurance_per_year"] + CAMPAIGN["land_lease_per_year"]) /
                FACILITY["capture_t_year"],
                FACILITY["capture_t_year"] / 1e6),
            "",
            "VARIABLE COSTS (per tonne CO2):",
            "  Energy: $0 (renewable self-generation, no fuel purchases)",
            "  Sorbent replacement: $%.1f/t (%.0f%%/yr degradation, $%.0f/t sorbent)" % (
                SORBENT["capacity_kg_per_kg"] * 1000 *
                CAMPAIGN["sorbent_replacement_frac"] *
                CAMPAIGN["sorbent_cost_per_t"] / 1000.0,
                CAMPAIGN["sorbent_replacement_frac"] * 100,
                CAMPAIGN["sorbent_cost_per_t"]),
            "  Water: $%.2f/t (%.1f m3/t at $%.2f/m3)" % (
                CAMPAIGN["water_m3_per_t_co2"] * CAMPAIGN["water_cost_per_m3"],
                CAMPAIGN["water_m3_per_t_co2"], CAMPAIGN["water_cost_per_m3"]),
            "",
            "PERIODIC REPLACEMENT COSTS (amortized):",
            "  Sorbent: $%.0fK/bed x %d beds = $%.1fM (~every 15 years)" % (
                CAMPAIGN["sorbent_repl_cost_per_bed"] / 1e3,
                FACILITY["sorbent_beds"],
                CAMPAIGN["sorbent_repl_cost_per_bed"] * FACILITY["sorbent_beds"] / 1e6),
            "  Battery: $%.1fM (~every 20 years)" % (
                CAMPAIGN["battery_repl_cost"] / 1e6),
            "  Amortized over 15 years: $%.1fM/year = $%.2f/t" % (
                (CAMPAIGN["sorbent_repl_cost_per_bed"] * FACILITY["sorbent_beds"] +
                 CAMPAIGN["battery_repl_cost"]) / 15 / 1e6,
                (CAMPAIGN["sorbent_repl_cost_per_bed"] * FACILITY["sorbent_beds"] +
                 CAMPAIGN["battery_repl_cost"]) / 15 / FACILITY["capture_t_year"]),
            "",
            "TOTAL SIMULATED OPEX: ~$1/t CO2",
            "  (excludes CAPEX amortization: $10-30/t at this scale)",
            "  (real-world first-of-a-kind: $400-1000/t, declining with scale)",
            "",
            "CAPEX (not simulated, estimated):",
            "  Contactor arrays (%d): ~$%.1fB" % (
                FACILITY["air_contactors"], FACILITY["air_contactors"] * 2.5e6 / 1e9),
            "  Regen units (%d): ~$%.1fB" % (
                FACILITY["regen_units"], FACILITY["regen_units"] * 5e6 / 1e9),
            "  Compressors + storage: ~$%.1fB" % (
                FACILITY["compressors"] * 12e6 / 1e9),
            "  Solar PV (%.0f MW): ~$%.1fB" % (
                SOLAR_PV_PEAK_KW / 1000, SOLAR_PV_PEAK_KW / 1000 * 0.8e6 / 1e9),
            "  Solar thermal (%.0f MWth): ~$%.1fB" % (
                SOLAR_TH_PEAK_KW / 1000, SOLAR_TH_PEAK_KW / 1000 * 0.5e6 / 1e9),
            "  Wind (%.0f MW): ~$%.1fB" % (
                WIND_RATED_KW / 1000, WIND_RATED_KW / 1000 * 1e6 / 1e9),
            "  Geothermal (%.0f MW): ~$%.0fM" % (GEO_THERMAL_KW/1000, GEO_THERMAL_KW/1000*1.5),
            "  Battery (%.0f MWh): ~$%.0fM" % (DIMS["battery_mwh"], DIMS["battery_mwh"]*0.01),
            "  Civil + site prep: ~$%.0fM" % (FACILITY["land_area_m2"]/1e6 * 50),
            "  Total CAPEX: ~$%.1fB (~$1000/t-year capacity)" % (
                FACILITY["capture_t_year"] / 1e3),
        ]),
        ("ABOUT THIS MODEL  (meta information)", [
            "This is a real-time 3D simulation of a Direct Air Capture plant",
            "built entirely in Python using Pygame for rendering.",
            "",
            "WHAT IT SHOWS:",
            "  -- Facility view: the entire %.0f ha plant to scale (1 unit = %.0f m)" % (
                FACILITY["land_area_m2"]/1e4, 1.0/FAC_DISP),
            "  -- Capture view: a single contactor unit in detail (1 unit = 8 m)",
            "  -- Operation view: live simulation with day/night cycle, weather,",
            "     energy flow, CO2 capture, and 15-year campaign tracking",
            "",
            "RENDERING ENGINE:",
            "  -- Custom software 3D renderer (no OpenGL/GPU required)",
            "  -- Painter's algorithm with back-face culling",
            "  -- Per-mesh lighting with directional light + ambient",
            "  -- Exploded, assembly, and cross-section views",
            "  -- Hover-picking with ray-triangle intersection",
            "",
            "PHYSICS & SIMULATION:",
            "  -- Solar irradiance model (latitude, time-of-day, season)",
            "  -- Wind power curve (cut-in, rated, cut-out speeds)",
            "  -- Geothermal baseload (constant 60 MW thermal)",
            "  -- Sorbent cycle: capture/regen/cool with staggered phases",
            "  -- Component health degradation + predictive maintenance",
            "  -- Energy economy: renewable-first dispatch with storage",
            "",
            "DIMENSIONAL ACCURACY:",
            "  -- All component sizes are real-world metres (SI units)",
            "  -- Based on published Carbon Engineering & Climeworks designs",
            "  -- Contactor: 20 m x 12 m x 4 m (cross-flow slab geometry)",
            "  -- Sorbent: 20 m x 10 m x 1.5 m honeycomb monolith (300 m3)",
            "  -- Regen: 12 m x 8 m x 10 m vacuum chamber (960 m3)",
            "  -- Fans: 3.5 m dia, 8 per contactor, CFRP blades, Ti hub",
            "",
            "COST MODEL:",
            "  -- OPEX only (labor, maintenance, sorbent, water, insurance)",
            "  -- Energy cost = $0 (renewable self-generation)",
            "  -- Simulated: ~$1/t CO2 (excluding CAPEX)",
            "  -- Real-world all-in: ~$10-30/t at scale (with CAPEX)",
            "",
            "Run: python CC.py    Press H for controls, I for full info.",
        ]),
        ("MODULAR DESIGN  (factory-built for scale)", [
            "This facility = %d x 100 kt CO2/year standard modules clustered." % (
                FACILITY["capture_t_year"] / 100000),
            "Each module: 5-10 ha footprint, factory-prefabricated skids.",
            "Construction: 12-18 months per module (site prep -> foundations",
            "-> structural erection -> contactor assembly -> sorbent loading",
            "-> piping/electrical -> commissioning -> MRV certification).",
            "Modules designed for factory production: 100-500 units/year",
            "once supply chains mature. Standardized containers/skids.",
            "Scaling: 500-2000 modules by 2030 (50-200 Mt/year).",
            "          ~10,000 modules by 2035 (~1 Gt/year total CDR).",
            "Preferred regions: US (45Q credits), EU, China, Middle East,",
            "Australia (cheap renewables + storage geology).",
            "Total investment to 1 Gt/year: ~$200-500 billion.",
        ]),
        ("ENHANCED WEATHERING  (complementary pathway)", [
            "Enhanced weathering (EW) spreads crushed silicate rocks (basalt,",
            "olivine) on agricultural land or coasts. CO2 reacts with minerals",
            "to form stable bicarbonate/carbonate -- decade-to-century storage.",
            "Potential: 0.1-0.5 Gt CO2/year by 2035; 2-4 Gt long-term.",
            "Application: 10-50 t rock/ha/year, particle size <100 um.",
            "Cost: $50-200/tCO2 currently; $20-100/t at scale.",
            "Co-benefits: soil pH improvement, reduced fertilizer need,",
            "crop-yield gains in acidic soils.",
            "Constraints: dust, heavy-metal leaching (clean rock required),",
            "land competition, long-term verification.",
            "Hybrid strategy: EW for low-cost bulk removal + DAC for",
            "high-purity, verifiable, permanent removal.",
        ]),
        ("MRV  (monitoring, reporting, verification)", [
            "Continuous CO2 sensors, flow meters, temperature/humidity probes.",
            "Third-party certification for carbon credits (e.g. 45Q, EU ETS).",
            "Satellite/soil sampling for enhanced weathering verification.",
            "Life-cycle assessment (LCA) for net-negative verification.",
            "Real-time AI optimization: PLC + neural nets adjust flow,",
            "humidity, temperature for maximum capture efficiency.",
        ]),
        ("ALTERNATIVE TECHNOLOGIES  (from goal research)", [
            "MSCC-EC: Molten Salt CO2 Capture + Electrochemical Conversion.",
            "  Captures CO2 in molten carbonates, electrolyzes to solid C + O2.",
            "  >90% Faradaic efficiency, ~10 GJ/t, 500-800 C operating temp.",
            "Liquid Metal Electrocatalysis: Ga-Ce alloys at room temperature.",
            "  >90% yield, ambient conditions, direct air contact.",
            "Tandem Catalytic to CNFs: CO2 -> carbonate -> carbon nanofibers.",
            "  >80% conversion, valuable products ($10-100/kg CNFs).",
            "Non-Thermal Plasma: electron-impact dissociation of CO2.",
            "  Ambient pressure, ferroelectric catalysts (PZT).",
            "Photocatalytic: solar-driven TiO2/graphene heterojunctions.",
            "Biochar Pyrolysis: biomass -> biochar (50-70% net efficiency).",
            "Humidity-Swing DAC: MOF sorbents, zero-energy regeneration.",
            "  2025 Northwestern research: 6x faster capture, 20% cost cut.",
        ]),
        ("GLOBAL ROADMAP  (2026-2035+)", [
            "HOW MANY GIGA-PLANTS TO MAKE A DIFFERENCE?",
            "  Global emissions: ~40 Gt CO2/year (2024)",
            "  Net-zero target: remove ~4 Gt/yr (with 90%% emissions cuts)",
            "  This giga-plant: %.1f Mt/yr -> need ~%.0f for 4 Gt/yr" % (
                FACILITY["capture_t_year"] / 1e6,
                4e9 / FACILITY["capture_t_year"]),
            "  For 1 Gt/yr: ~%.0f giga-plants (each %.1f Mt/yr)" % (
                1e9 / FACILITY["capture_t_year"],
                FACILITY["capture_t_year"] / 1e6),
            "  For 10 Gt/yr: ~%.0f giga-plants" % (
                10e9 / FACILITY["capture_t_year"]),
            "  Multi-story design: 50x scale = 50x fewer plants needed",
            "",
            "PHASE 1 (2026-2030): FIRST WAVE",
            "  Giga-plants: 6-24 x %.1f Mt DAC plants" % (FACILITY["capture_t_year"] / 1e6),
            "  CO2 removed: 200 Mt - 1 Gt/year",
            "  + Early enhanced weathering pilots (tens of Mt)",
            "  Focus: US (45Q credits), EU, China, Middle East, Australia",
            "  Financing: 45Q ($85/t), EU Innovation Fund, corporate offtake",
            "  Investment: $20B - $100B total CAPEX ($%.1fB per giga-plant)" % (
                FACILITY["capture_t_year"] / 1e3),
            "  Construction: 18-24 months per plant, factory-prefabricated",
            "  Staff needed: 900-3,600 (%d per plant)" % FACILITY["staff"],
            "  Learning: ~20%% cost reduction per capacity doubling",
            "",
            "PHASE 2 (2030-2035): RAPID SCALE-UP",
            "  Giga-plants: ~%.0f x %.1f Mt DAC plants" % (
                10e9 / FACILITY["capture_t_year"],
                FACILITY["capture_t_year"] / 1e6),
            "  CO2 removed: ~10 Gt/year (DAC) + ~1 Gt/year (EW)",
            "  Factory-built standardized units shipped globally",
            "  Co-locate with renewables, desalination, industry waste heat",
            "  Financing: carbon markets, green bonds, redirected fossil subsidies",
            "  Investment: ~$%.0fB cumulative CAPEX" % (
                10e9 / FACILITY["capture_t_year"] * FACILITY["capture_t_year"] / 1e3 * 1e9 / 1e9),
            "  Staff needed: ~%.0f globally (%d per plant)" % (
                10e9 / FACILITY["capture_t_year"] * FACILITY["staff"],
                FACILITY["staff"]),
            "  Sorbent production: ~%.0f Mt/yr (scaled supply chain)" % (
                FACILITY["sorbent_beds"] * DIMS["sorbent_t_per_bed"] / 1e6 *
                10e9 / FACILITY["capture_t_year"]),
            "  Annual OPEX: ~$10B ($1/t x 10 Gt)",
            "",
            "PHASE 3 (2035+): MULTI-GT/YEAR",
            "  Giga-plants: 94-235 for net-zero (with 90%% emissions cuts)",
            "  Hybrid: DAC + EW + reforestation + biochar",
            "  Target: 4-10 Gt/year total CDR",
            "  Investment: $2-10T total (comparable to global energy infrastructure)",
            "  Annual OPEX: $4-10B/Gt ($1/t OPEX)",
            "  Land: ~600 ha per giga-plant (multi-story saves land)",
            "",
            "COST SUMMARY:",
            "  Per giga-plant CAPEX: ~$%.1fB ($1000/t-yr capacity)" % (
                FACILITY["capture_t_year"] / 1e3),
            "  Per giga-plant OPEX: ~$1/t CO2 (renewable energy, $0 fuel)",
            "  All-in cost (w/ CAPEX): ~$10-30/t at 50x scale",
            "  Current first-of-a-kind: $400-1000/t (declining with scale)",
            "  Target 2030: ~$200/t (published studies)",
            "  Target 2035: ~$100/t (learning curve projection)",
            "",
            "COMPARISON TO OTHER INFRASTRUCTURE:",
            "  Global renewable energy investment: ~$1T/year",
            "  Global fossil fuel subsidies: ~$7T/year (IMF, incl. externalities)",
            "  DAC for 1 Gt/yr: ~$%.0fB CAPEX = <1 year of renewables investment" % (
                1e9 / FACILITY["capture_t_year"] * FACILITY["capture_t_year"] / 1e3),
            "  DAC for net-zero (4 Gt): ~$%.0fB = redirecting fossil subsidies for <1 yr" % (
                4e9 / FACILITY["capture_t_year"] * FACILITY["capture_t_year"] / 1e3),
        ]),
        ("MAINTENANCE & LONGEVITY  (15-year near-zero-failure design)", [
            "DESIGN PHILOSOPHY: over-engineer for zero failures.",
            "Target: >99.9%% availability, 0-1 failures over 15 years.",
            "Method: contactless components + predictive maintenance + redundancy.",
            "Maintenance staff: %d of %d total, $%.1fM/year budget." % (
                MAINTENANCE["staff_maint"], FACILITY["staff"],
                MAINTENANCE["annual_maint_cost"] / 1e6),
            "",
            "COMPONENT RELIABILITY (MTBF / life / predictive prevention):",
            "  Fans:          %.0f h / %d yr / %.1f%% prevented (contactless mag bearings)" % (
                COMPONENTS["fans"]["mtbf_h"], COMPONENTS["fans"]["design_life_years"],
                COMPONENTS["fans"]["predictive_factor"] * 100),
            "  Sorbent:       degrades / %d yr / no random failures (1.5%%/yr)" % COMPONENTS["sorbent"]["design_life_years"],
            "  Regen units:   %.0f h / %d yr / %.0f%% prevented (SiC, 300%% derated)" % (
                COMPONENTS["regen_units"]["mtbf_h"], COMPONENTS["regen_units"]["design_life_years"],
                COMPONENTS["regen_units"]["predictive_factor"] * 100),
            "  Vacuum pumps:  %.0f h / %d yr / %.0f%% prevented (hermetic dry screw)" % (
                COMPONENTS["vacuum_pumps"]["mtbf_h"], COMPONENTS["vacuum_pumps"]["design_life_years"],
                COMPONENTS["vacuum_pumps"]["predictive_factor"] * 100),
            "  Compressors:   %.0f h / %d yr / %.0f%% prevented (hermetic diaphragm)" % (
                COMPONENTS["compressors"]["mtbf_h"], COMPONENTS["compressors"]["design_life_years"],
                COMPONENTS["compressors"]["predictive_factor"] * 100),
            "  Solar PV:      %.0f h / %d yr / %.0f%% prevented (solid-state)" % (
                COMPONENTS["solar_pv"]["mtbf_h"], COMPONENTS["solar_pv"]["design_life_years"],
                COMPONENTS["solar_pv"]["predictive_factor"] * 100),
            "  Wind turbines: %.0f h / %d yr / %.0f%% prevented (direct-drive + mag bearings)" % (
                COMPONENTS["wind_turbines"]["mtbf_h"], COMPONENTS["wind_turbines"]["design_life_years"],
                COMPONENTS["wind_turbines"]["predictive_factor"] * 100),
            "  Battery:       %.0f h / %d yr / %.1f%% prevented (LFP + per-cell BMS)" % (
                COMPONENTS["battery"]["mtbf_h"], COMPONENTS["battery"]["design_life_years"],
                COMPONENTS["battery"]["predictive_factor"] * 100),
            "  Pipeline:      %.0f h / %d yr / %.1f%% prevented (cathodic + fiber-optic)" % (
                COMPONENTS["pipeline"]["mtbf_h"], COMPONENTS["pipeline"]["design_life_years"],
                COMPONENTS["pipeline"]["predictive_factor"] * 100),
            "",
            "PREDICTIVE MAINTENANCE SYSTEMS:",
            "  -- Vibration sensors on all rotating equipment (fans, pumps)",
            "  -- Thermal imaging on heaters, compressors, battery racks",
            "  -- SCADA condition monitoring with ML anomaly detection",
            "  -- Automated sensor sweep: 30 min/day (no manual inspection)",
            "  -- Robotic cleaning: PV panels, mirrors, fan filters",
            "  -- Battery BMS: per-cell monitoring + auto-rebalance",
            "  -- Pipeline: fiber-optic strain + smart pigging annually",
            "  -- Geothermal: downhole pressure/temp + anti-scale injection",
            "",
            "SCHEDULED MAINTENANCE (automated, minimized):",
            "  Daily:    0.5h automated sensor sweep (no staff time)",
            "  Weekly:   2h robotic filter cleaning (640 fan filters)",
            "  Monthly:  4h sorbent capacity test (80 beds, sampled)",
            "  6-month:  24h major PM (vibration, thermal, ultrasonic)",
            "  Annual:   60h full overhaul + sensor recalibration + 8%% health restore",
            "",
            "SORBENT MANAGEMENT (health-threshold triggered):",
            "  Partial refresh at < 85%% health (replace 20%% of beds)",
            "  Full replacement at < 70%% health (replace all 80 beds)",
            "  Degradation: 1.5%%/year (antioxidant + thermal stabilized amine)",
            "  Target: maintain > 85%% sorbent health across 15-year campaign",
            "",
            "BATTERY: replace when health < 80%% (~20+ years, LFP chemistry)",
            "All other components: design life exceeds 15-year campaign",
            "",
            "OVER-ENGINEERING FOR ZERO FAILURES:",
            "  -- Fans: contactless magnetic bearings (no contact = no wear)",
            "  -- Vacuum pumps: hermetic dry screw, magnetic coupling (no oil)",
            "  -- Compressors: diaphragm type, hermetic (no gas contact)",
            "  -- Wind turbines: direct-drive PMG + magnetic bearings (no gearbox)",
            "  -- Heaters: SiC ceramic, 300%% derated (never stressed)",
            "  -- Battery: LFP chemistry (no thermal runaway), prismatic cells",
            "  -- Tanks: SS 316L clad (corrosion-proof for 50+ years)",
            "  -- Pipeline: cathodic protection + fiber-optic monitoring",
            "  -- Control: triple-redundant PLC + hot-spare servers",
            "",
            "REDUNDANCY (failures don't stop the plant):",
            "  -- 640 fans: lose 10%% and still operate at 90%% capacity",
            "  -- 16 regen units for 80 beds (5:1 redundancy)",
            "  -- 4 compressors (N+2 redundancy at 50%% load each)",
            "  -- 20 battery racks (hot-swap individual modules)",
            "  -- Sorbent beds individually isolable for replacement",
            "  -- PLC auto-reroutes to healthy units on any fault",
        ]),
        ("VERIFICATION CHECKLIST", [
            "[x] 80 air capture fans with 8 fans each (to scale)",
            "[x] CO2 filter material with honeycomb channels",
            "[x] staggered cycle (capture/release/cool taking turns)",
            "[x] heating chamber with heating elements + insulation",
            "[x] vacuum pump (pulls CO2 out of filter)",
            "[x] CO2 collection pipes",
            "[x] valve system (intake/exhaust/CO2 output)",
            "[x] solar panels (84 MW peak)",
            "[x] solar mirrors (238 MW peak heat)",
            "[x] heat battery -- molten salt (1500 MWh)",
            "[x] wind turbines (50 MW)",
            "[x] geothermal wells (60 MW heat, always on)",
            "[x] CO2 compressors (squeeze to 150 bar for pipeline)",
            "[x] CO2 storage tanks (8 x 500 t)",
            "[x] CO2 pipeline to underground storage",
            "[x] battery bank (800 MWh)",
            "[x] cooling towers (4)",
            "[x] control building with antenna",
            "[x] smart energy controller (renewables-first)",
            "[x] to-scale 3D parts, hover inspector, section/exploded/assembly",
            "[x] modular design (9 x 100 kt standard modules)",
            "[x] enhanced weathering complementary pathway info",
            "[x] MRV / LCA / third-party certification info",
            "[x] global deployment roadmap (2026-2035)",
            "[x] alternative technology references (MSCC-EC, etc.)",
            "[x] component wear & tear model (MTBF, health, failures)",
            "[x] 15-year longevity simulation with scheduled replacements",
            "[x] predictive maintenance (vibration, thermal, SCADA, ML)",
            "[x] near-zero failure engineering (magnetic bearings, dry screw, direct-drive)",
            "[x] per-component materials, dimensions, and design life",
            "[x] health-threshold triggered replacements (sorbent <70%%, battery <80%%)",
            "[x] redundancy design (N+2 compressors, 5:1 regen, 640 fans)",
            "[x] automated/robotic maintenance (sensor sweep, cleaning, BMS)",
        ]),
        ("CONTROLS", [
            "TAB  cycle FACILITY / CAPTURE / OPERATION / URBAN modes",
            "or click the mode tabs in the top bar",
            "mouse drag  orbit    wheel  zoom    right-drag  pan",
            "E  explode    X  cross-section    L  labels    R  reset view",
            "[ ]  step assembly    A  assemble all    C  clear",
            "click a part in the left PARTS list to pin it",
            "V  verification checklist    I  this info panel",
            "In OPERATION:  , / .  slow / speed up TIME-WARP",
            "H  controls    ESC  quit",
        ]),
        ("URBAN MINI-PLANT  (DAC inside skyscrapers)", [
            "Tab 4 shows a cutaway skyscraper with a DAC mini-plant",
            "installed on a vacant office floor. The concept: use existing",
            "empty commercial building floors in any city worldwide.",
            "",
            "WHAT IT IS:",
            "  A compact DAC unit that fits inside one floor (~500 m2)",
            "  4 small contactors (3m x 2.5m x 1.5m each) in a 2x2 grid",
            "  3 fans per contactor (0.8m dia, 4-blade CFRP, quiet)",
            "  2 compact regen chambers (VSA, 100 C)",
            "  1 CO2 compressor + 2 buffer tanks (5t each)",
            "  CO2 routed via building riser pipe to street collection",
            "",
            "PERFORMANCE:",
            "  Capture: %.0f t CO2/year per unit" % URBAN["capture_t_year"],
            "  Daily: %.1f t/day    Hourly: %.2f t/h" % (
                URBAN["capture_t_day"], URBAN["capture_t_hour"]),
            "  Power: %.0f kW avg (%.0f kW peak)" % (
                URBAN["power_kw"], URBAN["power_peak_kw"]),
            "  Energy: %.0f kWh/t CO2 (higher than giga-plant)" % URBAN["energy_kwh_t"],
            "  Noise: %d dB at 1m (office-compatible)" % URBAN["noise_db"],
            "  Water: %.1f m3/t (closed-loop)" % URBAN["water_m3_t"],
            "  Staff: 0 (remote monitoring, cloud-based)" % (),
            "",
            "ECONOMICS:",
            "  CAPEX: $%.1fM (factory-built, mass-produced)" % (URBAN["capex_usd"] / 1e6),
            "  OPEX: $%.0f/t CO2 (building power + maintenance)" % URBAN["opex_per_t"],
            "  All-in: $%.0f/t (with CAPEX over 15 years)" % URBAN["all_in_per_t"],
            "  Energy: $%.2f/kWh (commercial electricity rate)" % URBAN["energy_cost_kwh"],
            "  Sorbent: $%.0fK/bed, replace every %.0f years" % (
                URBAN["sorbent_repl_cost"] / 1e3, URBAN["sorbent_repl_years"]),
            "  Maintenance: $%.0fK/year (contracted service)" % (
                URBAN["maint_per_year"] / 1e3),
            "",
            "URBAN DEPLOYMENT:",
            "  Per building: %d units = %.0f kt/yr (10 vacant floors)" % (
                URBAN["units_per_building"], URBAN["building_capture_t_yr"] / 1000),
            "  Per city: %d buildings = %.1f Mt/yr" % (
                URBAN["buildings_per_city"],
                URBAN["buildings_per_city"] * URBAN["building_capture_t_yr"] / 1e6),
            "  100 cities: ~%.0f Mt/yr total (%dK units)" % (
                URBAN["global_capture_t_yr"] / 1e6, URBAN["units_global"] / 1000),
            "",
            "ADVANTAGES:",
            "  - Uses existing buildings (no land acquisition needed)",
            "  - Distributed: captures CO2 where people live",
            "  - Factory-built modules (ship + install in days)",
            "  - Building power supply (no separate energy infrastructure)",
            "  - Complementary with giga-plants (different use cases)",
            "  - Utilizes vacant commercial real estate productively",
            "",
            "CHALLENGES:",
            "  - Higher $/t than giga-plant (smaller = less efficient)",
            "  - Depends on building electrical grid (may add CO2 if fossil)",
            "  - CO2 collection requires street tanker logistics",
            "  - Building codes and permits for industrial equipment",
            "  - Best with green electricity contracts or rooftop solar",
        ]),
    ]


# =============================================================================
# SECTION 9 -- HUD / UI HELPERS
# =============================================================================

def vgradient(surf, top, bot):
    h = surf.get_height()
    w = surf.get_width()
    for y in range(h):
        t = y / max(1, h)
        col = (int(top[0] + (bot[0] - top[0]) * t),
               int(top[1] + (bot[1] - top[1]) * t),
               int(top[2] + (bot[2] - top[2]) * t))
        pygame.draw.line(surf, col, (0, y), (w, y))


def bar(surf, font, x, y, w, h, frac, color, label, valtext):
    pygame.draw.rect(surf, C_PANEL_HI, (x, y, w, h), border_radius=4)
    frac = clamp(frac)
    pygame.draw.rect(surf, color, (x, y, int(w * frac), h), border_radius=4)
    surf.blit(font.render(label, True, C_TEXT_DIM), (x, y - 16))
    img = font.render(valtext, True, C_TEXT)
    surf.blit(img, (x + w - img.get_width(), y - 16))


def panel(surf, x, y, w, h, alpha=210):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((C_PANEL[0], C_PANEL[1], C_PANEL[2], alpha))
    surf.blit(s, (x, y))
    pygame.draw.rect(surf, C_PANEL_HI, (x, y, w, h), 1, border_radius=6)


def wrap_text(font, text, maxpx):
    out, cur = [], ""
    for word in text.split(" "):
        trial = word if not cur else cur + " " + word
        if font.size(trial)[0] <= maxpx:
            cur = trial
        else:
            if cur:
                out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out or [""]


def _label(surf, font, text, pos, accent=False):
    col = (255, 210, 120) if accent else C_TEXT
    dot = (255, 210, 120) if accent else C_ACCENT
    img = font.render(text, True, col)
    x, y = int(pos[0]) + 6, int(pos[1]) - 6
    bg = pygame.Surface((img.get_width() + 8, img.get_height() + 4), pygame.SRCALPHA)
    bg.fill((10, 14, 20, 190))
    surf.blit(bg, (x - 4, y - 2))
    pygame.draw.circle(surf, dot, (int(pos[0]), int(pos[1])), 3)
    surf.blit(img, (x, y))


def draw_co2_glow(surf, x, y, R, intensity):
    """A CO2 capture glow -- blue-white, indicating active CO2 absorption."""
    R = int(max(2, R * (0.55 + 0.9 * intensity)))
    g = pygame.Surface((R * 2 + 4, R * 2 + 4), pygame.SRCALPHA)
    c = R + 2
    pygame.draw.circle(g, (90, 180, 220, int(60 * intensity)), (c, c), R)
    pygame.draw.circle(g, (120, 200, 240, int(100 * intensity)), (c, c), int(R * 0.62))
    pygame.draw.circle(g, (200, 230, 255, int(200 * intensity)), (c, c), int(R * 0.30))
    surf.blit(g, (int(x - c), int(y - c)))


def draw_heat_glow(surf, x, y, R, intensity):
    """A regeneration heat glow -- orange-red, indicating active heating."""
    R = int(max(2, R * (0.55 + 0.9 * intensity)))
    g = pygame.Surface((R * 2 + 4, R * 2 + 4), pygame.SRCALPHA)
    c = R + 2
    pygame.draw.circle(g, (255, 100, 40, int(70 * intensity)), (c, c), R)
    pygame.draw.circle(g, (255, 150, 60, int(120 * intensity)), (c, c), int(R * 0.62))
    pygame.draw.circle(g, (255, 220, 180, int(230 * intensity)), (c, c), int(R * 0.30))
    surf.blit(g, (int(x - c), int(y - c)))


# =============================================================================
# SECTION 10 -- 3D RENDERER (projects + paints the spec'd Parts)
# =============================================================================

class FacilityRenderer:
    """Projects + paints spec'd Parts with painter's algorithm. Supports full /
    exploded / assembly views, an optional half-section CUT, mouse hover-picking.
    Geometry-agnostic -- draws the whole facility and the capture unit."""

    def __init__(self, parts_builder, is_capture=False,
                 home_az=0.62, home_el=0.42, home_dist=1.75):
        self.parts_builder = parts_builder
        self.is_capture = is_capture
        self.parts = parts_builder()
        self._home = (home_az, home_el, home_dist)
        self.az, self.el, self.dist = home_az, home_el, home_dist
        self.pan = np.array([0.0, 0.0])
        self.light = np.array([0.4, 0.7, 1.0])
        self.light = self.light / np.linalg.norm(self.light)
        self.view = "full"
        self.section = False
        self.explode_amt = 0.0
        self.assembled = len(self.parts)
        self.hovered = None
        self.selected = None
        self.pop = np.zeros(len(self.parts))
        self.hover_spread = 0.0

    def reset_view(self):
        self.az, self.el, self.dist = self._home
        self.pan = np.array([0.0, 0.0])

    def zoom_at(self, factor, mouse_pos=None, rect=None):
        old = self.dist
        self.dist = max(0.35, min(11.0, self.dist * factor))
        if old <= 1e-6 or mouse_pos is None or rect is None:
            return
        if not rect.collidepoint(mouse_pos):
            return
        anchor = np.array([mouse_pos[0] - (rect.x + rect.w / 2.0),
                           mouse_pos[1] - (rect.y + rect.h / 2.0)], dtype=float)
        scale = old / self.dist
        self.pan = anchor - (anchor - self.pan) * scale

    def orbit(self, dx, dy, fine=False):
        sens = 0.004 if fine else 0.009
        self.az += dx * sens
        self.el += dy * sens
        self.el = max(-1.55, min(1.55, self.el))

    def pan_by(self, dx, dy, fine=False):
        sens = 0.45 if fine else 1.0
        self.pan += np.array([dx * sens, dy * sens])

    def set_view(self, mode):
        self.view = mode
        if mode == "assembly" and self.assembled >= len(self.parts):
            self.assembled = 0
        self.selected = None

    def toggle_section(self):
        self.section = not self.section

    def assembly_next(self):
        self.assembled = min(len(self.parts), self.assembled + 1)

    def assembly_prev(self):
        self.assembled = max(0, self.assembled - 1)

    def assembly_all(self):
        self.assembled = len(self.parts)

    def assembly_clear(self):
        self.assembled = 0

    def active_part(self):
        i = self.selected if self.selected is not None else self.hovered
        return self.parts[i] if i is not None else None

    def placing_part(self):
        for p in self.parts:
            if p.order == self.assembled:
                return p
        return None

    def tick(self, dt):
        if self.view != "assembly":
            target = 1.0 if self.view == "exploded" else 0.0
            self.explode_amt += (target - self.explode_amt) * min(1.0, dt * 4)
        hi = self.selected if self.selected is not None else self.hovered
        sp_target = 0.30 if (hi is not None and self.view == "full") else 0.0
        self.hover_spread += (sp_target - self.hover_spread) * min(1.0, dt * 5)
        for i in range(len(self.parts)):
            tp = 1.0 if i == hi else 0.0
            self.pop[i] += (tp - self.pop[i]) * min(1.0, dt * 8)

    def _layout(self, pi, vw, eamt):
        part = self.parts[pi]
        if vw == "assembly":
            if part.order < self.assembled:
                return part.explode * 0.0, 1.0, "normal"
            if part.order == self.assembled:
                return part.explode * 0.55, 1.0, "active"
            return part.explode * 1.0, 0.30, "pending"
        return part.explode * eamt, 1.0, "normal"

    def _section_cut(self, wv, face):
        c = wv[list(face)].mean(axis=0)
        return c[0] > 0.004

    def render(self, surf, rect, angles, co2_glow=None, heat_glow=None,
               mouse_pos=None, show_labels=True, label_font=None,
               interactive=False, sorbent_states=None):
        clip = surf.get_clip()
        surf.set_clip(rect)
        cx = rect.x + rect.w / 2.0 + self.pan[0]
        cy = rect.y + rect.h / 2.0 + self.pan[1]
        focal = min(rect.w, rect.h) * 1.12
        Rcam = rot_x(self.el) @ rot_y(self.az)
        default_ang = angles.get("default", 0.0)

        vw = self.view
        eamt = self.explode_amt
        if self.view == "full":
            eamt += self.hover_spread
        section = self.section and vw in ("full", "exploded")
        hi = self.selected if self.selected is not None else self.hovered

        polys, labels, leaders, screeninfo, glow_points = [], [], [], [], []
        lx, ly, lz = float(self.light[0]), float(self.light[1]), float(self.light[2])

        for pi, part in enumerate(self.parts):
            base_off, dim, tag = self._layout(pi, vw, eamt)
            pop = self.pop[pi]
            off = base_off + part.popdir * (pop * 0.16)
            highlight = (pi == hi)
            allcam = []
            for m in part.meshes:
                wv = m.world_verts(angles.get(m.group, default_ang)) + off
                cam = wv @ Rcam.T
                cam[:, 2] += self.dist
                allcam.append(cam)
                col = m.color
                # sorbent state color shifting
                if m.sorbent_state is not None and sorbent_states is not None:
                    state = sorbent_states.get(pi, "capture")
                    if state == "capture":
                        col = _mix(col, C_SORBENT_LOADED, 0.3)
                    elif state == "regen":
                        col = _mix(col, C_SORBENT_REGEN, 0.5)
                    elif state == "cool":
                        col = _mix(col, C_SORBENT_COOL, 0.3)
                if m.hot and heat_glow is not None and heat_glow > 0.01:
                    col = _mix(col, (255, 90, 40), min(0.7, heat_glow * 0.6))
                if dim < 0.99:
                    col = (int(col[0] * dim), int(col[1] * dim), int(col[2] * dim))
                if highlight:
                    col = _mix(col, (255, 255, 255), 0.28)
                cr, cg, cb = col
                if highlight:
                    outline, ow = C_ACCENT, 2
                elif tag == "active":
                    outline, ow = (255, 210, 120), 2
                else:
                    outline, ow = None, 0
                # Projection + face processing (hybrid: Python for small, numpy for large)
                fa = m.faces_np
                n_faces = len(fa)
                if n_faces == 0:
                    continue
                cam_list = cam.tolist()
                if (show_labels and label_font and m.name and tag != "pending"
                        and (vw != "full" or highlight)):
                    mcen = cam.mean(axis=0)
                    if mcen[2] > 0.05:
                        mlx = cx + focal * mcen[0] / mcen[2]
                        mly = cy - focal * mcen[1] / mcen[2]
                        labels.append((mcen[2], (mlx, mly), m.name, "detail"))

                if n_faces <= 16:
                    # Pure Python path — avoids numpy overhead for small meshes
                    n_verts = len(cam_list)
                    sxl = [0.0] * n_verts
                    syl = [0.0] * n_verts
                    for vi in range(n_verts):
                        vz = cam_list[vi][2]
                        if vz > 0.05:
                            sxl[vi] = cx + focal * cam_list[vi][0] / vz
                            syl[vi] = cy - focal * cam_list[vi][1] / vz
                    faces_list = m.faces
                    for fi in range(n_faces):
                        face = faces_list[fi]
                        f0, f1, f2 = face
                        vz0 = cam_list[f0][2]
                        vz1 = cam_list[f1][2]
                        vz2 = cam_list[f2][2]
                        if vz0 <= 0.05 or vz1 <= 0.05 or vz2 <= 0.05:
                            continue
                        if section:
                            fc_x = (wv[f0][0] + wv[f1][0] + wv[f2][0]) / 3.0
                            if fc_x > 0.004:
                                continue
                        ax, ay = cam_list[f0][0], cam_list[f0][1]
                        bx, by, bz = cam_list[f1][0], cam_list[f1][1], cam_list[f1][2]
                        fx, fy, fz = cam_list[f2][0], cam_list[f2][1], cam_list[f2][2]
                        az = vz0
                        ux, uy, uz = bx - ax, by - ay, bz - az
                        wx, wy, wz = fx - ax, fy - ay, fz - az
                        nx = uy * wz - uz * wy
                        ny = uz * wx - ux * wz
                        nz = ux * wy - uy * wx
                        nl = (nx * nx + ny * ny + nz * nz) ** 0.5
                        if nl < 1e-12:
                            nl = 1.0
                        inv = 1.0 / nl
                        nx *= inv; ny *= inv; nz *= inv
                        if nz > 0:
                            continue
                        d = nx * lx + ny * ly + nz * lz
                        s = 0.35 + 0.65 * (d if d > 0.0 else 0.0)
                        ds = (vz0 + vz1 + vz2) / 3.0
                        polys.append((ds, [(sxl[f0], syl[f0]),
                                           (sxl[f1], syl[f1]),
                                           (sxl[f2], syl[f2])],
                                      (int(cr * s), int(cg * s), int(cb * s)), outline, ow))
                else:
                    # Vectorized numpy path for large meshes
                    dz_arr = cam[:, 2]
                    safe_z = np.maximum(dz_arr, 0.06)
                    sx_arr = cx + focal * cam[:, 0] / safe_z
                    sy_arr = cy - focal * cam[:, 1] / safe_z
                    v0 = cam[fa[:, 0]]
                    v1 = cam[fa[:, 1]]
                    v2 = cam[fa[:, 2]]
                    face_dz = (v0[:, 2] + v1[:, 2] + v2[:, 2]) * (1.0 / 3.0)
                    visible = (v0[:, 2] > 0.05) & (v1[:, 2] > 0.05) & (v2[:, 2] > 0.05)
                    if section:
                        face_centers = (wv[fa[:, 0]] + wv[fa[:, 1]] + wv[fa[:, 2]]) * (1.0 / 3.0)
                        visible = visible & (face_centers[:, 0] <= 0.004)
                    ux = v1[:, 0] - v0[:, 0]
                    uy = v1[:, 1] - v0[:, 1]
                    uz = v1[:, 2] - v0[:, 2]
                    wx = v2[:, 0] - v0[:, 0]
                    wy = v2[:, 1] - v0[:, 1]
                    wz = v2[:, 2] - v0[:, 2]
                    nx = uy * wz - uz * wy
                    ny = uz * wx - ux * wz
                    nz = ux * wy - uy * wx
                    nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
                    nlen = np.maximum(nlen, 1e-12)
                    nx = nx / nlen
                    ny = ny / nlen
                    nz = nz / nlen
                    visible = visible & (nz <= 0)
                    d = nx * lx + ny * ly + nz * lz
                    shade = 0.35 + 0.65 * np.maximum(d, 0.0)
                    vis_idx = np.nonzero(visible)[0]
                    if len(vis_idx) == 0:
                        continue
                    # Batch-compute screen coords and colors for visible faces
                    fa_vis = fa[vis_idx]
                    sx_f = sx_arr[fa_vis[:, 0]]
                    sy_f = sy_arr[fa_vis[:, 1]]
                    sx_f2 = sx_arr[fa_vis[:, 2]]
                    sy_f2 = sy_arr[fa_vis[:, 2]]
                    sx_f0 = sx_arr[fa_vis[:, 0]]
                    sy_f0 = sy_arr[fa_vis[:, 0]]
                    sx_f1 = sx_arr[fa_vis[:, 1]]
                    sy_f1 = sy_arr[fa_vis[:, 1]]
                    shade_vis = shade[vis_idx]
                    fdz_vis = face_dz[vis_idx]
                    cr_i, cg_i, cb_i = int(cr), int(cg), int(cb)
                    s_list = shade_vis.tolist()
                    fd_list = fdz_vis.tolist()
                    sx0l = sx_f0.tolist()
                    sy0l = sy_f0.tolist()
                    sx1l = sx_f1.tolist()
                    sy1l = sy_f1.tolist()
                    sx2l = sx_f2.tolist()
                    sy2l = sy_f2.tolist()
                    _append = polys.append
                    for i in range(len(vis_idx)):
                        s = s_list[i]
                        _append((fd_list[i],
                                 [(sx0l[i], sy0l[i]), (sx1l[i], sy1l[i]), (sx2l[i], sy2l[i])],
                                 (int(cr_i * s), int(cg_i * s), int(cb_i * s)), outline, ow))

            if not allcam:
                continue
            cam_all = np.vstack(allcam)
            cen = cam_all.mean(axis=0)
            if cen[2] > 0.05:
                safez = np.where(cam_all[:, 2] <= 0.05, 1e9, cam_all[:, 2])
                scx = cx + focal * cam_all[:, 0] / safez
                scy = cy - focal * cam_all[:, 1] / safez
                pcx = cx + focal * cen[0] / cen[2]
                pcy = cy - focal * cen[1] / cen[2]
                rad = float(np.max(np.hypot(scx - pcx, scy - pcy))) * 0.55 + 6
                screeninfo.append((pi, pcx, pcy, rad, cen[2], tag))
                anchor = getattr(part, "fire_anchor", None)
                if anchor is not None and tag != "pending":
                    ca = (anchor + off) @ Rcam.T
                    caz = float(ca[2]) + self.dist
                    if caz > 0.05:
                        fpx = cx + focal * float(ca[0]) / caz
                        fpy = cy - focal * float(ca[1]) / caz
                        glow_points.append((caz, fpx, fpy, max(8.0, rad * 0.30)))
                if show_labels and label_font and tag != "pending":
                    labels.append((cen[2], (pcx, pcy), part.name, tag))
                if tag == "active":
                    hc = cen - (off @ Rcam.T)
                    if hc[2] > 0.05:
                        hx = cx + focal * hc[0] / hc[2]
                        hy = cy - focal * hc[1] / hc[2]
                        leaders.append(((pcx, pcy), (hx, hy)))

        if polys:
            depths = np.array([p[0] for p in polys])
            order = np.argsort(-depths)
            for idx in order:
                _, pts, fc, outline, ow = polys[idx]
                try:
                    pygame.draw.polygon(surf, fc, pts)
                    if outline is not None:
                        pygame.draw.polygon(surf, outline, pts, ow)
                except Exception:
                    pass

        glow_points.sort(key=lambda t: t[0], reverse=True)
        if co2_glow is not None and co2_glow > 0.05:
            for _, fpx, fpy, crad in glow_points:
                draw_co2_glow(surf, fpx, fpy, crad, co2_glow)
        if heat_glow is not None and heat_glow > 0.05:
            for _, fpx, fpy, crad in glow_points:
                draw_heat_glow(surf, fpx, fpy, crad, heat_glow)

        for a, b in leaders:
            pygame.draw.line(surf, (255, 210, 120), a, b, 1)
            pygame.draw.circle(surf, (255, 210, 120), (int(b[0]), int(b[1])), 5, 1)

        if show_labels and label_font:
            labels.sort(key=lambda t: t[0])
            used = []
            for _, (lxx, lyy), text, tag in labels:
                ly2 = lyy
                for uy in used:
                    if abs(ly2 - uy) < 16:
                        ly2 = uy + 16
                used.append(ly2)
                _label(surf, label_font, text, (lxx, ly2),
                       accent=(tag == "active" or tag == "detail"))

        if interactive and mouse_pos is not None:
            mxp, myp = mouse_pos
            best, bestd = None, 1e18
            for pi, pcx, pcy, rad, depth, tag in screeninfo:
                if tag == "pending":
                    continue
                if math.hypot(mxp - pcx, myp - pcy) <= rad and depth < bestd:
                    bestd, best = depth, pi
            self.hovered = best

        surf.set_clip(clip)


# =============================================================================
# SECTION 11 -- APPLICATION
# =============================================================================

VIEW_ROT_SCALE = 0.008


class App:
    MODES = ["facility", "capture", "operation", "urban"]
    MODE_NAME = {"facility": "FACILITY  (whole DAC plant)",
                 "capture": "CAPTURE  (single contactor unit)",
                 "operation": "OPERATION  (annual capture campaign)",
                 "urban": "URBAN  (skyscraper mini-plant)"}

    LEFT_PANEL_W = 220
    RIGHT_PANEL_W = 352
    TOP_BAR_H = 36
    BOTTOM_BAR_H = 86

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("CC.py -- Carbon Capture DAC Digital Twin")
        self.W, self.H = 1480, 900
        self.fullscreen = False
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 14)
        self.fs = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 12)
        self.fb = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 20, bold=True)
        self.fbig = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 30, bold=True)
        self.fmicro = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 11)

        self.fac_rend = FacilityRenderer(build_facility_parts, is_capture=False,
                                         home_az=0.78, home_el=0.55, home_dist=3.80)
        self._cap_rend = None
        self._urb_rend = None
        self.pt = CapturePowertrain()
        self.campaign = Campaign()

        self.mode = "facility"
        self.ang = {}
        self.co2_glow = 0.0
        self.heat_glow = 0.0
        self.show_labels = True
        self.show_help = False
        self.show_info = False
        self.show_checklist = False
        self.info_scroll = 0
        self.info_sections = build_info_sections()
        self.dragging = False
        self.panning = False
        self.running = True
        self.bg = None
        self._preview_hitboxes = {}
        self._mode_hitboxes = {}
        self._part_list_hitboxes = {}
        self._rebuild_bg()

    def _rebuild_bg(self):
        self.bg = pygame.Surface((self.W, self.H))
        vgradient(self.bg, BG_TOP, BG_BOT)

    def rend(self):
        if self.mode == "capture":
            if self._cap_rend is None:
                self._cap_rend = FacilityRenderer(build_capture_parts, is_capture=True,
                                                  home_az=0.62, home_el=0.42, home_dist=1.80)
            return self._cap_rend
        if self.mode == "urban":
            if self._urb_rend is None:
                self._urb_rend = FacilityRenderer(build_urban_parts, is_capture=False,
                                                   home_az=0.55, home_el=0.15, home_dist=3.50)
            return self._urb_rend
        return self.fac_rend

    def view_rect(self):
        if self.mode == "operation":
            return pygame.Rect(0, self.TOP_BAR_H, self.W, self.H - self.TOP_BAR_H)
        return pygame.Rect(self.LEFT_PANEL_W + 4, self.TOP_BAR_H + 4,
                           self.W - self.LEFT_PANEL_W - self.RIGHT_PANEL_W - 8,
                           self.H - self.TOP_BAR_H - self.BOTTOM_BAR_H - 8)

    def handle_events(self, dt):
        r = self.rend()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.VIDEORESIZE:
                self.W, self.H = e.w, e.h
                self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
                self._rebuild_bg()
            elif e.type == pygame.KEYDOWN:
                self._key(e)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    if self._handle_mode_tab_click(e.pos):
                        continue
                    if self.mode == "operation":
                        continue
                    if self._handle_part_list_click(e.pos):
                        continue
                    if self._handle_preview_click(e.pos):
                        continue
                    self.dragging = True
                    if r.hovered is not None:
                        r.selected = r.hovered
                    else:
                        r.selected = None
                elif e.button == 3:
                    self.panning = True
                elif e.button == 4:
                    r.zoom_at(0.9, pygame.mouse.get_pos(), self.view_rect())
                elif e.button == 5:
                    r.zoom_at(1.1, pygame.mouse.get_pos(), self.view_rect())
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    self.dragging = False
                elif e.button == 3:
                    self.panning = False
            elif e.type == pygame.MOUSEMOTION:
                if self.mode != "operation":
                    if self.dragging:
                        r.orbit(e.rel[0], e.rel[1])
                    elif self.panning:
                        r.pan_by(e.rel[0], e.rel[1])

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            info = pygame.display.Info()
            self.W, self.H = info.current_w, info.current_h
            self.screen = pygame.display.set_mode((self.W, self.H), pygame.FULLSCREEN)
        else:
            self.W, self.H = 1480, 900
            self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        self._rebuild_bg()

    def _key(self, e):
        k = e.key
        r = self.rend()
        if k == pygame.K_ESCAPE:
            if self.fullscreen:
                self._toggle_fullscreen()
            else:
                self.running = False
        elif k == pygame.K_q:
            self.running = False
        elif k == pygame.K_F11:
            self._toggle_fullscreen()
        elif k == pygame.K_TAB:
            i = self.MODES.index(self.mode)
            self.mode = self.MODES[(i + 1) % len(self.MODES)]
        elif k == pygame.K_h:
            self.show_help = not self.show_help
        elif k == pygame.K_i:
            self.show_info = not self.show_info
            self.info_scroll = 0
        elif self.show_info and k in (pygame.K_DOWN, pygame.K_j):
            self.info_scroll += 40
        elif self.show_info and k in (pygame.K_UP, pygame.K_k):
            self.info_scroll = max(0, self.info_scroll - 40)
        elif k == pygame.K_l:
            self.show_labels = not self.show_labels
        elif k == pygame.K_v:
            self.show_checklist = not self.show_checklist
        elif k == pygame.K_r:
            r.reset_view()
        elif k == pygame.K_1 and self.mode != "operation":
            r.set_view("full")
        elif k == pygame.K_2 and self.mode != "operation":
            r.set_view("exploded")
        elif k == pygame.K_3 and self.mode != "operation":
            r.set_view("assembly")
        elif k == pygame.K_4 and self.mode != "operation":
            r.toggle_section()
        elif k == pygame.K_e and self.mode != "operation":
            r.set_view("exploded" if r.view != "exploded" else "full")
        elif k == pygame.K_x and self.mode != "operation":
            r.toggle_section()
        elif k == pygame.K_LEFTBRACKET and self.mode != "operation":
            r.set_view("assembly"); r.assembly_prev()
        elif k == pygame.K_RIGHTBRACKET and self.mode != "operation":
            r.set_view("assembly"); r.assembly_next()
        elif k == pygame.K_a and self.mode != "operation":
            r.set_view("assembly"); r.assembly_all()
        elif k == pygame.K_c and self.mode != "operation":
            r.set_view("assembly"); r.assembly_clear()
        elif k == pygame.K_COMMA:
            self.campaign.cycle_warp(-1)
        elif k == pygame.K_PERIOD:
            self.campaign.cycle_warp(+1)
        elif self.mode == "operation":
            if k == pygame.K_LEFT:
                step = 4 if (e.mod & pygame.KMOD_SHIFT) else 1
                self.campaign.jump_relative(-step, self.pt)
            elif k == pygame.K_RIGHT:
                step = 4 if (e.mod & pygame.KMOD_SHIFT) else 1
                self.campaign.jump_relative(+step, self.pt)
            elif k == pygame.K_HOME:
                self.campaign.jump_to_week(-1, self.pt)

    def _advance_angles(self, dt):
        a = self.ang
        for key in ("fan", "contactorfan", "windrotor", "vacuum", "default"):
            a.setdefault(key, 0.0)
        if self.mode in ("capture", "operation"):
            w = VIEW_ROT_SCALE * 2 * math.pi / 60.0
            rpm = self.pt.fan_rpm if self.pt.fan_rpm > 0 else 120.0
            a["fan"] += rpm * w * dt
            a["contactorfan"] += rpm * w * 0.8 * dt
            a["vacuum"] += rpm * w * 1.5 * dt
        elif self.mode == "facility":
            # Gentle idle rotation for facility view fans (slow ambient spin)
            a["contactorfan"] += 0.15 * dt
        elif self.mode == "urban":
            # Gentle idle rotation for urban mini-plant fans
            a["contactorfan"] += 0.3 * dt
        # Wind rotor: slow, proportional to wind speed (realistic RPM feel)
        a["windrotor"] += (2.0 + self.campaign.wind_ms * 0.12) * dt
        a["default"] += 0.12 * dt

    def update(self, dt):
        self.rend().tick(dt)
        if self.mode == "capture":
            self.pt.update_demo(dt)
            self.co2_glow = 0.6
            self.heat_glow = clamp((self.pt.regen_temp_c - THERM["ambient_c"]) / 80.0)
        elif self.mode == "operation":
            if not self.campaign.finished and not self.campaign.time_jumping:
                dt_c = self.campaign.update(dt, self.pt)
                remaining = dt_c
                step_max = 120.0
                while remaining > 1e-6:
                    step = min(step_max, remaining)
                    self.pt.update(step, self.campaign.sun, self.campaign.wind_ms)
                    remaining -= step
            self.co2_glow = clamp(self.pt.capture_rate_kg_s / 100.0)
            self.heat_glow = clamp((self.pt.regen_temp_c - THERM["ambient_c"]) / 80.0)
        else:
            self.co2_glow = 0.0
            self.heat_glow = 0.0
        self._advance_angles(dt)

        r = self.rend()
        mp = pygame.mouse.get_pos()
        over_view = self.view_rect().collidepoint(mp) and not self._over_panel(mp)
        if self.mode != "operation" and over_view and not (self.dragging or self.panning):
            pass
        else:
            if not self.dragging:
                r.hovered = None

    def _over_panel(self, mp):
        if self.show_info or self.show_help or self.show_checklist:
            return True
        if mp[0] < self.LEFT_PANEL_W + 4 and self.mode != "operation":
            return True
        if mp[0] > self.W - self.RIGHT_PANEL_W - 4 and self.mode != "operation":
            return True
        if mp[1] > self.H - self.BOTTOM_BAR_H - 4 and self.mode != "operation":
            return True
        if mp[1] < self.TOP_BAR_H:
            return True
        return False

    def _handle_mode_tab_click(self, pos):
        for mode, rect in self._mode_hitboxes.items():
            if rect.collidepoint(pos):
                self.mode = mode
                return True
        return False

    def _handle_part_list_click(self, pos):
        r = self.rend()
        for pi, rect in self._part_list_hitboxes.items():
            if rect.collidepoint(pos):
                r.selected = pi
                return True
        return False

    def _handle_preview_click(self, pos):
        if self.mode == "operation":
            return False
        if self.show_checklist:
            self.show_checklist = False
            return True
        hit = self._preview_hitboxes
        if hit.get("labels") and hit["labels"].collidepoint(pos):
            self.show_labels = not self.show_labels
            return True
        if hit.get("reset") and hit["reset"].collidepoint(pos):
            self.rend().reset_view()
            return True
        if hit.get("section") and hit["section"].collidepoint(pos):
            self.rend().toggle_section()
            return True
        for mode, rect in hit.get("views", []):
            if rect.collidepoint(pos):
                if mode == "section":
                    self.rend().toggle_section()
                elif mode in ("full", "exploded", "assembly"):
                    self.rend().set_view(mode)
                else:
                    r = self.rend()
                    r.set_view("assembly")
                    if mode == "prev":
                        r.assembly_prev()
                    elif mode == "next":
                        r.assembly_next()
                    elif mode == "all":
                        r.assembly_all()
                    elif mode == "clear":
                        r.assembly_clear()
                return True
        return False

    def draw(self):
        self.screen.blit(self.bg, (0, 0))
        if self.mode == "operation":
            self.draw_operation()
        else:
            self.draw_preview()
        self.draw_topbar()
        if self.show_help:
            self.draw_help()
        if self.show_info:
            self.draw_info()
        if self.show_checklist:
            self.draw_checklist()
        pygame.display.flip()

    def draw_topbar(self):
        pygame.draw.rect(self.screen, C_PANEL, (0, 0, self.W, self.TOP_BAR_H))
        pygame.draw.line(self.screen, C_PANEL_HI, (0, self.TOP_BAR_H), (self.W, self.TOP_BAR_H), 1)
        self.screen.blit(self.fb.render("CARBON CAPTURE", True, C_CO2), (12, 6))
        self.screen.blit(self.font.render("DAC PLANT  |  " + self.MODE_NAME[self.mode],
                                          True, C_TEXT), (172, 10))
        self._mode_hitboxes = {}
        tab_x = self.W - 660
        tab_y = 4
        tab_h = 28
        for mode in self.MODES:
            label = mode.upper()
            active = (self.mode == mode)
            tw = self.fs.size(label)[0] + 24
            rect = pygame.Rect(tab_x, tab_y, tw, tab_h)
            panel(self.screen, rect.x, rect.y, rect.w, rect.h, alpha=240 if active else 170)
            col = C_ACCENT if active else C_TEXT_DIM
            self.screen.blit(self.fs.render(label, True, col), (rect.x + 12, rect.y + 8))
            self._mode_hitboxes[mode] = rect
            tab_x += tw + 6
        hint = "H help  I info  V checklist"
        img = self.fs.render(hint, True, C_TEXT_DIM)
        self.screen.blit(img, (self.W - img.get_width() - 12, self.TOP_BAR_H - 16))

    def draw_preview(self):
        r = self.rend()
        rect = self.view_rect()
        self._preview_hitboxes = {}
        mp = pygame.mouse.get_pos()
        interactive = rect.collidepoint(mp) and not self._over_panel(mp)
        sorbent_states = None
        if self.mode == "capture":
            state = self.pt.demo_cycle_state
            sorbent_states = {2: state}  # sorbent bed part index
            # Sync glow effects with cycle phase
            if state == "capture":
                self.co2_glow = clamp(self.pt.capture_rate_kg_s / 100.0)
                self.heat_glow = 0.0
            elif state == "regen":
                self.co2_glow = 0.3
                self.heat_glow = clamp((self.pt.regen_temp_c - THERM["ambient_c"]) / 80.0)
            else:  # cool
                self.co2_glow = 0.1
                self.heat_glow = clamp((self.pt.regen_temp_c - THERM["ambient_c"]) / 160.0)
        r.render(self.screen, rect, self.ang,
                 co2_glow=self.co2_glow if self.mode == "capture" else None,
                 heat_glow=self.heat_glow if self.mode == "capture" else None,
                 mouse_pos=mp, show_labels=self.show_labels, label_font=self.fs,
                 interactive=interactive, sorbent_states=sorbent_states)
        self.draw_view_tabs()
        self.draw_part_list()
        self.draw_scale_bar(rect)
        self.draw_spec_card()
        if self.mode == "facility":
            self.draw_facility_legend()
        elif self.mode == "urban":
            self.draw_urban_stats()
        else:
            self.draw_capture_stats()
        self.draw_preview_footer()

    def draw_view_tabs(self):
        r = self.rend()
        x, y = self.LEFT_PANEL_W + 8, self.TOP_BAR_H + 6
        items = [("full", "1 FULL"), ("exploded", "2 EXPLODED"),
                 ("assembly", "3 ASSEMBLY"), ("section", "4 SECTION")]
        views = []
        cursor = x
        for mode, label in items:
            active = r.section if mode == "section" else (r.view == mode)
            tw = self.fs.size(label)[0] + 18
            rect = pygame.Rect(cursor, y, tw, 24)
            panel(self.screen, rect.x, rect.y, rect.w, rect.h, alpha=235 if active else 175)
            col = C_ACCENT if active else C_TEXT_DIM
            self.screen.blit(self.fs.render(label, True, col), (rect.x + 9, rect.y + 6))
            views.append((mode, rect))
            cursor += tw + 8
        for action, label in (("prev", "<"), ("next", ">"), ("all", "ALL"), ("clear", "CLR")):
            tw = self.fs.size(label)[0] + 14
            rect = pygame.Rect(cursor, y, tw, 24)
            panel(self.screen, rect.x, rect.y, rect.w, rect.h, alpha=175)
            self.screen.blit(self.fs.render(label, True, C_TEXT_DIM), (rect.x + 7, rect.y + 6))
            views.append((action, rect))
            cursor += tw + 6
        self._preview_hitboxes["views"] = views

    def draw_preview_footer(self):
        r = self.rend()
        w = self.W - self.LEFT_PANEL_W - self.RIGHT_PANEL_W - 16
        h = self.BOTTOM_BAR_H - 8
        x = self.LEFT_PANEL_W + 8
        y = self.H - h - 4
        panel(self.screen, x, y, w, h, alpha=220)
        line1 = "drag orbit   right-drag pan   wheel zoom   click pin part   TAB mode"
        line2 = "L labels   R reset   E explode   X section   [ ] build   A all   C clear"
        self.screen.blit(self.fs.render(line1, True, C_TEXT), (x + 12, y + 10))
        self.screen.blit(self.fs.render(line2, True, C_TEXT_DIM), (x + 12, y + 30))
        line3 = "V checklist   I full info   H help   F11 fullscreen   , / . time-warp"
        self.screen.blit(self.fs.render(line3, True, C_TEXT_DIM), (x + 12, y + 50))

        labels_txt = "LABELS ON" if self.show_labels else "LABELS OFF"
        reset_txt = "RESET VIEW"
        section_txt = "CUT ON" if r.section else "CUT OFF"
        rx = x + w - 240
        chips = []
        for text, key, active in ((labels_txt, "labels", self.show_labels),
                                  (section_txt, "section", r.section),
                                  (reset_txt, "reset", False)):
            tw = self.fs.size(text)[0] + 16
            rect = pygame.Rect(rx, y + 18, tw, 24)
            panel(self.screen, rect.x, rect.y, rect.w, rect.h, alpha=235 if active else 180)
            self.screen.blit(self.fs.render(text, True, C_ACCENT if active else C_TEXT),
                             (rect.x + 8, rect.y + 6))
            chips.append((key, rect))
            rx += tw + 8
        for key, rect in chips:
            self._preview_hitboxes[key] = rect

    def draw_spec_card(self):
        r = self.rend()
        part = r.active_part()
        if part is None:
            if r.view == "assembly":
                part = r.placing_part()
            if part is None:
                self.draw_inspector_hint()
                return
        lines = part.specs
        w = self.LEFT_PANEL_W - 16
        body = []
        for ln in lines:
            body += wrap_text(self.fs, ln, w - 28)
        h = 72 + len(body) * 16
        x, y = 8, self.H - self.BOTTOM_BAR_H - h - 8
        panel(self.screen, x, y, w, h)
        self.screen.blit(self.fb.render(part.name, True, C_ACCENT), (x + 12, y + 8))
        state = "PINNED / HOVERED PART"
        if r.view == "assembly" and r.selected is None and r.hovered is None:
            state = "NEXT PART TO PLACE"
        self.screen.blit(self.fs.render(state, True, C_TEXT_DIM), (x + 14, y + 34))
        yy = y + 52
        for ln in body:
            self.screen.blit(self.fs.render("- " + ln, True, C_TEXT), (x + 14, yy))
            yy += 16

    def draw_inspector_hint(self):
        w, h = self.LEFT_PANEL_W - 16, 112
        x, y = 8, self.H - self.BOTTOM_BAR_H - h - 8
        panel(self.screen, x, y, w, h)
        self.screen.blit(self.fb.render("INSPECTOR", True, C_ACCENT), (x + 12, y + 8))
        lines = [
            "Hover a part to read its real dimensions and function.",
            "Click to pin the current part so it stays readable while orbiting.",
            "Use FULL / EXPLODED / ASSEMBLY / SECTION above to expose internals.",
        ]
        yy = y + 38
        for ln in lines:
            self.screen.blit(self.fs.render(ln, True, C_TEXT), (x + 14, yy))
            yy += 18

    def draw_facility_legend(self):
        w, x = self.RIGHT_PANEL_W - 16, self.W - self.RIGHT_PANEL_W + 8
        y = self.TOP_BAR_H + 4
        h = self.H - self.TOP_BAR_H - self.BOTTOM_BAR_H - 12
        panel(self.screen, x, y, w, h)
        self.screen.blit(self.fb.render("FACILITY SPEC", True, C_TEXT), (x + 12, y + 8))

        # --- Process flow diagram ---
        yy = y + 36
        self.screen.blit(self.fs.render("HOW IT WORKS (step by step)", True, C_ACCENT), (x + 12, yy))
        yy += 18
        flow_steps = [
            ("1. CAPTURE", "%d fans pull air" % FACILITY["air_contactors"], "CO2 sticks to filter", C_CONTACTOR),
            ("2. RELEASE", "%d heaters warm" % FACILITY["regen_units"], "filter -> CO2 lets go", C_REGEN),
            ("3. SQUEEZE", "%d compressors" % FACILITY["compressors"], "CO2 gas -> liquid", C_COMP),
            ("4. STORE", "%d tanks hold" % FACILITY["co2_storage_tanks"], "%.0f t liquid CO2" % (DIMS["storage_tanks"] * DIMS["storage_capacity_t"]), C_CO2TANK),
            ("5. BURY", "pipeline sends CO2", "deep underground", C_PIPELINE),
        ]
        for i, (title, desc1, desc2, col) in enumerate(flow_steps):
            # step box
            box_y = yy
            pygame.draw.rect(self.screen, col, (x + 14, box_y, 4, 28))
            self.screen.blit(self.fs.render(title, True, col), (x + 22, box_y))
            self.screen.blit(self.fmicro.render(desc1, True, C_TEXT_DIM), (x + 22, box_y + 13))
            self.screen.blit(self.fmicro.render(desc2, True, C_TEXT_DIM), (x + 22, box_y + 22))
            yy += 32
            if i < len(flow_steps) - 1:
                # arrow down
                self.screen.blit(self.fs.render("|", True, C_TEXT_DIM), (x + 15, yy - 4))
                self.screen.blit(self.fs.render("v", True, C_TEXT_DIM), (x + 14, yy + 2))
                yy += 8

        yy += 8
        self.screen.blit(self.fs.render("-" * 38, True, C_TEXT_DIM), (x + 12, yy))
        yy += 16

        # --- Key specs ---
        rows = [
            ("Class", "%.1f Mt CO2/year -- like %.1fM cars" % (
                FACILITY["capture_t_year"]/1e6, FACILITY["capture_t_year"]/4.6/1e6)),
            ("Site", "%.0f ha (%.1f km2), %d stories" % (
                FACILITY["land_area_m2"]/1e4, FACILITY["land_area_m2"]/1e6,
                DIMS["contactor_stories"])),
            ("Capture fans", "%d x %.0fm, %d fans each" % (
                FACILITY["air_contactors"], DIMS["contactor_w_m"], DIMS["contactor_fans"])),
            ("CO2 filter", "%.0f t/bed, grabs %.2f kg CO2/kg" % (
                DIMS["sorbent_t_per_bed"], DIMS["sorbent_cap_kg_kg"])),
            ("Cycle", "%.0f min (%.1f x/day per filter)" % (
                SORBENT["cycle_total_min"], SORBENT["cycles_per_day"])),
            ("", ""),
            ("ENERGY SOURCES", ""),
            ("Solar panels", "%.0f MW peak electricity" % (SOLAR_PV_PEAK_KW / 1000)),
            ("Solar mirrors", "%.0f MW peak heat" % (SOLAR_TH_PEAK_KW / 1000)),
            ("Wind", "%.0f MW (%d turbines)" % (
                WIND_RATED_KW / 1000, DIMS["wind_turbines"])),
            ("Geothermal", "%.0f MW heat (always on)" % (GEO_THERMAL_KW / 1000)),
            ("", ""),
            ("STORAGE", ""),
            ("Heat battery", "%.0f MWh molten salt" % DIMS["thermal_storage_mwh"]),
            ("Battery", "%.0f MWh lithium-ion" % DIMS["battery_mwh"]),
            ("", ""),
            ("CO2 OUTPUT", ""),
            ("Compressors", "%d units, squeeze to %.0f bar" % (
                FACILITY["compressors"], DIMS["storage_bar"])),
            ("CO2 tanks", "%d tanks, %.0f t total" % (
                DIMS["storage_tanks"],
                DIMS["storage_tanks"] * DIMS["storage_capacity_t"])),
            ("Pipeline", "%.0f t/h to underground storage" % CO2_STORE["pipeline_rate_t_h"]),
            ("", ""),
            ("PERFORMANCE", ""),
            ("Energy/t CO2", "%.0f kWh (thermal+elec)" % ENERGY["total_kwh_t"]),
            ("Target cost", "~$%.0f/t CO2" % CAMPAIGN["cost_per_t_target"]),
            ("Staff", "%d, 24/7" % FACILITY["staff"]),
            ("", ""),
            ("CO2 captured", "%.0f t (this run)" % self.pt.co2_captured_t),
            ("CO2 sequestered", "%.0f t" % self.pt.co2_sequestered_t),
            ("", ""),
            ("15-YEAR DEPLOYMENT PLAN", ""),
            ("This giga-plant", "%.1f Mt/yr = %.0f Mt over 15 yr" % (
                FACILITY["capture_t_year"]/1e6,
                FACILITY["capture_t_year"]*15/1e6)),
            ("Phase 1 (2026-30)", "6-24 plants -> 200 Mt-1 Gt/yr"),
            ("  Cost", "$20-100B total investment"),
            ("Phase 2 (2030-35)", "~235 plants -> ~10 Gt/yr"),
            ("  Cost", "$~10B cumulative"),
            ("Phase 3 (2035+)", "94-235 for net-zero, hybrid"),
            ("Per plant CAPEX", "~$%.1fB ($1000/t-yr capacity)" % (
                FACILITY["capture_t_year"]/1e3)),
            ("Per plant OPEX", "~$1/t (renewable energy)"),
            ("All-in cost", "~$10-30/t at 50x scale (w/ CAPEX)"),
            ("Plants for 1 Gt/yr", "~%.0f x %.1f Mt giga-plants" % (
                1e9/FACILITY["capture_t_year"], FACILITY["capture_t_year"]/1e6)),
            ("Plants for 4 Gt/yr", "~%.0f (net-zero target)" % (
                4e9/FACILITY["capture_t_year"])),
            ("  (with 90%% cuts)", "vs 47,000 old 850kt plants"),
        ]
        for lab, val in rows:
            if lab == "":
                yy += 6
                continue
            if val == "":
                self.screen.blit(self.fs.render(lab, True, C_ACCENT), (x + 12, yy))
            else:
                self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM), (x + 12, yy))
                col = C_CO2 if "CO2" in val else C_TEXT
                self.screen.blit(self.fs.render(val, True, col), (x + 130, yy))
            yy += 17

    def draw_capture_stats(self):
        w, x = self.RIGHT_PANEL_W - 16, self.W - self.RIGHT_PANEL_W + 8
        y = self.TOP_BAR_H + 4
        h = self.H - self.TOP_BAR_H - self.BOTTOM_BAR_H - 12
        panel(self.screen, x, y, w, h)
        self.screen.blit(self.fb.render("CAPTURE UNIT", True, C_CO2), (x + 12, y + 8))
        pt = self.pt

        # --- Cycle phase indicator ---
        yy = y + 36
        state = pt.demo_cycle_state
        phase_min = pt.demo_cycle_phase_min
        cycle_total = SORBENT["cycle_total_min"]
        if state == "capture":
            phase_label = "CAPTURE"
            phase_col = C_SORBENT_LOADED
            phase_elapsed = phase_min
            phase_dur = SORBENT["cycle_capture_min"]
        elif state == "regen":
            phase_label = "REGEN"
            phase_col = C_SORBENT_REGEN
            phase_elapsed = phase_min - SORBENT["cycle_capture_min"]
            phase_dur = SORBENT["cycle_regen_min"]
        else:
            phase_label = "COOL"
            phase_col = C_SORBENT_COOL
            phase_elapsed = phase_min - SORBENT["cycle_capture_min"] - SORBENT["cycle_regen_min"]
            phase_dur = SORBENT["cycle_cool_min"]

        self.screen.blit(self.fs.render("CURRENT PHASE", True, C_ACCENT), (x + 12, yy))
        yy += 16
        # phase bar
        bar_w = w - 24
        bar_x = x + 12
        pygame.draw.rect(self.screen, C_PANEL, (bar_x, yy, bar_w, 20))
        # full cycle bar segments
        cap_w = bar_w * SORBENT["cycle_capture_min"] / cycle_total
        reg_w = bar_w * SORBENT["cycle_regen_min"] / cycle_total
        cool_w = bar_w * SORBENT["cycle_cool_min"] / cycle_total
        pygame.draw.rect(self.screen, _mix(C_SORBENT_LOADED, C_PANEL, 0.5), (bar_x, yy, int(cap_w), 20))
        pygame.draw.rect(self.screen, _mix(C_SORBENT_REGEN, C_PANEL, 0.5), (bar_x + int(cap_w), yy, int(reg_w), 20))
        pygame.draw.rect(self.screen, _mix(C_SORBENT_COOL, C_PANEL, 0.5), (bar_x + int(cap_w) + int(reg_w), yy, int(cool_w), 20))
        # current position marker
        pos_x = bar_x + int(bar_w * phase_min / cycle_total)
        pygame.draw.rect(self.screen, phase_col, (pos_x - 2, yy - 2, 4, 24))
        pygame.draw.rect(self.screen, C_TEXT_DIM, (bar_x, yy, bar_w, 20), 1)
        yy += 24
        self.screen.blit(self.fs.render(phase_label, True, phase_col), (x + 12, yy))
        self.screen.blit(self.fs.render("%.0f / %.0f min" % (phase_elapsed, phase_dur), True, C_TEXT_DIM),
                         (x + 120, yy))
        yy += 20

        # --- Key specs ---
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 8, yy), (x + w - 8, yy), 1)
        yy += 6
        rows = [
            ("Fans", "%d x %.0f m, %d blades" % (
                DIMS["contactor_fans"], DIMS["fan_d_m"], DIMS["fan_blades"])),
            ("Fan speed", "%.0f rpm" % pt.fan_rpm),
            ("Sorbent", "%.0f t, %.2f kg CO2/kg" % (
                DIMS["sorbent_t_per_bed"], DIMS["sorbent_cap_kg_kg"])),
            ("Regen temp", "%.0f C (target %.0f)" % (
                pt.regen_temp_c, THERM["regen_target_c"])),
            ("Capture rate", "%.1f kg/s" % pt.capture_rate_kg_s),
            ("", ""),
            ("Cycle", "%.0f+%.0f+%.0f = %.0f min" % (
                SORBENT["cycle_capture_min"], SORBENT["cycle_regen_min"],
                SORBENT["cycle_cool_min"], cycle_total)),
            ("Cycles/day", "%.1f per bed" % SORBENT["cycles_per_day"]),
            ("CO2/cycle", "%.1f t" % (
                DIMS["sorbent_t_per_bed"] * DIMS["sorbent_cap_kg_kg"])),
            ("Capture eff", "%.0f%% of CO2 in air" % (SORBENT["capture_eff"] * 100)),
        ]
        for lab, val in rows:
            if lab == "":
                yy += 6
                continue
            self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM), (x + 12, yy))
            col = C_CO2 if "CO2" in val else C_TEXT
            self.screen.blit(self.fs.render(val, True, col), (x + 130, yy))
            yy += 17

        # --- Build dimensions (blueprint reference) ---
        yy += 6
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 8, yy), (x + w - 8, yy), 1)
        yy += 6
        self.screen.blit(self.fs.render("BUILD DIMENSIONS (blueprint)", True, C_ACCENT), (x + 12, yy))
        yy += 16
        build_dims = [
            ("Contactor", "%.0f x %.0f x %.0f m" % (
                DIMS["contactor_w_m"], DIMS["contactor_h_m"], DIMS["contactor_d_m"])),
            ("Filter bed", "%.0f x %.0f x %.1f m (%.0f m3)" % (
                DIMS["bed_w_m"], DIMS["bed_h_m"], DIMS["bed_d_m"],
                DIMS["bed_w_m"] * DIMS["bed_h_m"] * DIMS["bed_d_m"])),
            ("Regen chamber", "%.0f x %.0f x %.0f m (%.0f m3)" % (
                DIMS["regen_w_m"], DIMS["regen_h_m"], DIMS["regen_d_m"],
                DIMS["regen_w_m"] * DIMS["regen_h_m"] * DIMS["regen_d_m"])),
            ("Fans", "%d x %.0f m dia, %d-blade CFRP" % (
                DIMS["contactor_fans"], DIMS["fan_d_m"], DIMS["fan_blades"])),
            ("Vacuum pump", "%.1f m dia x %.1f m H" % (
                DIMS["regen_vacuum_d_m"], DIMS["regen_vacuum_h_m"])),
            ("Pipes", "%.0f mm dia, SS 316L" % (DIMS["manifold_d_m"] * 1000)),
            ("Scale", "1 unit = 8 m (capture view)"),
        ]
        for lab, val in build_dims:
            self.screen.blit(self.fmicro.render(lab, True, C_TEXT_DIM), (x + 14, yy))
            self.screen.blit(self.fmicro.render(val, True, C_TEXT), (x + 110, yy))
            yy += 14

        # --- Materials & fabrication (per-component) ---
        yy += 6
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 8, yy), (x + w - 8, yy), 1)
        yy += 6
        self.screen.blit(self.fs.render("MATERIALS & FABRICATION", True, C_ACCENT), (x + 12, yy))
        yy += 16
        mat_order = [
            ("Frame", "frame"), ("Foundation", "foundation"), ("Columns", "columns"),
            ("Fans", "fans"), ("Motor", "fan_motor"), ("Sorbent", "sorbent_bed"),
            ("Cassette", "bed_frame"), ("Regen vessel", "regen_chamber"),
            ("Insulation", "insulation"), ("Heaters", "heaters"),
            ("Vacuum pump", "vacuum_pump"), ("Manifold", "manifold"),
            ("Valves", "valves"), ("Plenum", "plenum"),
            ("Output pipe", "output_pipe"), ("Fasteners", "fasteners"),
        ]
        for lab, key in mat_order:
            m = MATERIALS[key]
            self.screen.blit(self.fmicro.render(lab, True, C_TEXT_DIM), (x + 14, yy))
            self.screen.blit(self.fmicro.render(m["material"], True, C_TEXT), (x + 80, yy))
            yy += 12
            self.screen.blit(self.fmicro.render("  %s" % m["grade"], True, C_TEXT_DIM), (x + 80, yy))
            yy += 12
            self.screen.blit(self.fmicro.render("  $%.0fK" % (m["est_cost_usd"] / 1000), True, C_CO2), (x + 80, yy))
            self.screen.blit(self.fmicro.render("SAVE: %s" % m["cost_reduction"], True, C_GOOD), (x + 130, yy))
            yy += 13
        yy += 4
        self.screen.blit(self.fs.render("TOTAL: $%.0fK per capture unit" % (CAPTURE_UNIT_COST / 1000), True, C_CO2), (x + 12, yy))
        yy += 16

        # --- Visual process flow ---
        yy += 6
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 8, yy), (x + w - 8, yy), 1)
        yy += 6
        self.screen.blit(self.fs.render("CAPTURE CYCLE FLOW", True, C_ACCENT), (x + 12, yy))
        yy += 18

        # Determine which steps are active based on current phase
        flow_steps = [
            ("1. Air intake", "fans draw ambient air", C_FAN, state == "capture"),
            ("2. CO2 capture", "sorbent absorbs CO2", C_SORBENT_LOADED, state == "capture"),
            ("3. Seal chamber", "valves close, vacuum on", C_WARN, state == "regen"),
            ("4. Heat to 100C", "sorbent releases CO2", C_SORBENT_REGEN, state == "regen"),
            ("5. Vacuum extract", "pump pulls CO2 out", C_COMP, state == "regen"),
            ("6. Cool down", "bed returns to capture", C_SORBENT_COOL, state == "cool"),
            ("7. CO2 to manifold", "collected for compress", C_CO2BAND, state == "regen"),
        ]
        for title, desc, col, active in flow_steps:
            if active:
                pygame.draw.rect(self.screen, _mix(col, C_PANEL, 0.7), (x + 10, yy - 1, w - 20, 22))
                pygame.draw.rect(self.screen, col, (x + 10, yy - 1, 3, 22))
            self.screen.blit(self.fmicro.render(title, True, col if active else C_TEXT_DIM), (x + 18, yy))
            self.screen.blit(self.fmicro.render(desc, True, C_TEXT if active else C_TEXT_DIM), (x + 18, yy + 11))
            yy += 23
            if active:
                yy += 1

    def draw_urban_stats(self):
        w, x = self.RIGHT_PANEL_W - 16, self.W - self.RIGHT_PANEL_W + 8
        y = self.TOP_BAR_H + 4
        h = self.H - self.TOP_BAR_H - self.BOTTOM_BAR_H - 12
        panel(self.screen, x, y, w, h)
        self.screen.blit(self.fb.render("URBAN MINI-PLANT", True, C_CO2), (x + 12, y + 8))
        self.screen.blit(self.fs.render("DAC on 1 floor -- building stays open", True, C_TEXT_DIM), (x + 12, y + 28))
        yy = y + 48

        # --- How it works ---
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 8, yy), (x + w - 8, yy), 1)
        yy += 6
        self.screen.blit(self.fs.render("HOW IT WORKS", True, C_ACCENT), (x + 12, yy))
        yy += 16
        steps = [
            "1 vacant floor -> DAC mini-plant (fits 1 floor)",
            "All other floors: active office with workers",
            "4 compact contactors (2x2 grid, 3m x 2.5m)",
            "Air drawn through MOF sorbent beds",
            "Regen: heat + vacuum releases CO2",
            "CO2 compressed -> building riser pipe",
            "Street-level tanker collection",
            "Building remains fully operational",
        ]
        for s in steps:
            self.screen.blit(self.fmicro.render(s, True, C_TEXT), (x + 14, yy))
            yy += 14
        yy += 6

        # --- Key specs ---
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 8, yy), (x + w - 8, yy), 1)
        yy += 6
        self.screen.blit(self.fs.render("KEY SPECS", True, C_ACCENT), (x + 12, yy))
        yy += 16
        rows = [
            ("Contactors", "%d units (%.1f x %.1f x %.1f m)" % (
                URBAN["contactors"], URBAN["contactor_w_m"],
                URBAN["contactor_h_m"], URBAN["contactor_d_m"])),
            ("Fans", "%d x %.1f m dia, %d-blade" % (
                URBAN["fans_per_contactor"], URBAN["fan_d_m"], URBAN["fan_blades"])),
            ("Sorbent", "%.0f t/bed, %.2f kg CO2/kg" % (
                URBAN["sorbent_t_per_bed"], URBAN["sorbent_cap_kg_kg"])),
            ("Regen units", "%d (%.0f x %.0f x %.0f m)" % (
                URBAN["regen_units"], URBAN["regen_w_m"],
                URBAN["regen_h_m"], URBAN["regen_d_m"])),
            ("CO2 tanks", "%d x %.0f t buffer" % (
                URBAN["co2_tanks"], URBAN["co2_tank_cap_t"])),
            ("", ""),
            ("Capture", "%.0f t CO2/year" % URBAN["capture_t_year"]),
            ("Daily rate", "%.1f t/day" % URBAN["capture_t_day"]),
            ("Power draw", "%.0f kW avg (%.0f kW peak)" % (
                URBAN["power_kw"], URBAN["power_peak_kw"])),
            ("Energy/t", "%.0f kWh/t CO2" % URBAN["energy_kwh_t"]),
            ("Noise", "%d dB at 1m (office-safe)" % URBAN["noise_db"]),
            ("Staff", "0 (remote monitoring)"),
            ("Floor area", "%.0f m2 (one floor)" % URBAN["floor_area_m2"]),
        ]
        for lab, val in rows:
            if lab == "":
                yy += 6
                continue
            self.screen.blit(self.fmicro.render(lab, True, C_TEXT_DIM), (x + 14, yy))
            col = C_CO2 if "CO2" in val else C_TEXT
            self.screen.blit(self.fmicro.render(val, True, col), (x + 110, yy))
            yy += 14

        # --- Economics ---
        yy += 6
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 8, yy), (x + w - 8, yy), 1)
        yy += 6
        self.screen.blit(self.fs.render("ECONOMICS", True, C_ACCENT), (x + 12, yy))
        yy += 16
        econ = [
            ("CAPEX", "$%.1fM (factory-built)" % (URBAN["capex_usd"] / 1e6)),
            ("OPEX", "$%.0f/t CO2" % URBAN["opex_per_t"]),
            ("All-in cost", "$%.0f/t (w/ CAPEX, 15 yr)" % URBAN["all_in_per_t"]),
            ("Energy cost", "$%.0f/kWh commercial" % URBAN["energy_cost_kwh"]),
            ("Sorbent repl", "$%.0fK/bed, every %.0f yr" % (
                URBAN["sorbent_repl_cost"] / 1e3, URBAN["sorbent_repl_years"])),
            ("Maintenance", "$%.0fK/year (contracted)" % (
                URBAN["maint_per_year"] / 1e3)),
        ]
        for lab, val in econ:
            self.screen.blit(self.fmicro.render(lab, True, C_TEXT_DIM), (x + 14, yy))
            col = C_GOOD if "$" in val and "/t" in val else C_TEXT
            self.screen.blit(self.fmicro.render(val, True, col), (x + 110, yy))
            yy += 14

        # --- Urban deployment plan ---
        yy += 6
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 8, yy), (x + w - 8, yy), 1)
        yy += 6
        self.screen.blit(self.fs.render("URBAN DEPLOYMENT PLAN", True, C_CO2), (x + 12, yy))
        yy += 16
        deploy = [
            ("Per building", "%.0f kt/yr (%d units)" % (
                URBAN["building_capture_t_yr"] / 1000, URBAN["units_per_building"])),
            ("Per city", "%.1f Mt/yr (%d buildings)" % (
                URBAN["buildings_per_city"] * URBAN["building_capture_t_yr"] / 1e6,
                URBAN["buildings_per_city"])),
            ("100 cities", "%.0f Mt/yr total" % (
                URBAN["global_capture_t_yr"] / 1e6)),
            ("Global units", "~%.0fK units" % (
                URBAN["units_global"] / 1000)),
            ("", ""),
            ("vs giga-plant", "94 giga-plants + urban = complementary"),
            ("Advantage", "Uses existing buildings, no land needed"),
            ("CO2 collection", "Street tanker via building riser"),
        ]
        for lab, val in deploy:
            if lab == "":
                yy += 6
                continue
            self.screen.blit(self.fmicro.render(lab, True, C_TEXT_DIM), (x + 14, yy))
            col = C_CO2 if "Mt" in val or "complementary" in val else C_TEXT
            self.screen.blit(self.fmicro.render(val, True, col), (x + 110, yy))
            yy += 14

    def draw_part_list(self):
        r = self.rend()
        w = self.LEFT_PANEL_W - 16
        x = 8
        y = self.TOP_BAR_H + 4
        h = self.H - self.TOP_BAR_H - self.BOTTOM_BAR_H - 220
        if h < 100:
            h = 100
        panel(self.screen, x, y, w, h)
        self.screen.blit(self.fb.render("PARTS", True, C_ACCENT), (x + 12, y + 8))
        self._part_list_hitboxes = {}
        yy = y + 36
        hi = r.selected if r.selected is not None else r.hovered
        for pi, part in enumerate(r.parts):
            active = (pi == hi)
            row_h = 20
            if yy + row_h > y + h - 4:
                break
            rect = pygame.Rect(x + 4, yy, w - 8, row_h)
            if active:
                panel(self.screen, rect.x, rect.y, rect.w, rect.h, alpha=235)
            elif pi % 2 == 0:
                panel(self.screen, rect.x, rect.y, rect.w, rect.h, alpha=120)
            col = C_ACCENT if active else C_TEXT
            num = "%d" % (pi + 1)
            self.screen.blit(self.fmicro.render(num, True, C_TEXT_DIM), (rect.x + 4, rect.y + 4))
            self.screen.blit(self.fs.render(part.name, True, col), (rect.x + 24, rect.y + 3))
            self._part_list_hitboxes[pi] = rect
            yy += row_h

    def draw_scale_bar(self, rect):
        r = self.rend()
        disp = CAP_DISP if self.mode == "capture" else FAC_DISP
        target_px = 120
        metres_per_px = 1.0 / (disp * min(rect.w, rect.h) * 1.12 / r.dist)
        target_m = target_px * metres_per_px
        magnitude = 10 ** math.floor(math.log10(max(1.0, target_m)))
        for mult in (1, 2, 5, 10, 20, 50, 100):
            nice = mult * magnitude
            if nice >= target_m:
                break
        bar_m = nice
        bar_px = int(bar_m * disp * min(rect.w, rect.h) * 1.12 / r.dist)
        bar_px = max(20, min(300, bar_px))
        bx = rect.x + 12
        by = rect.bottom - 22
        pygame.draw.rect(self.screen, C_TEXT_DIM, (bx, by, bar_px, 3))
        pygame.draw.rect(self.screen, C_TEXT_DIM, (bx, by - 4, 2, 11))
        pygame.draw.rect(self.screen, C_TEXT_DIM, (bx + bar_px - 2, by - 4, 2, 11))
        label = "%d m" % bar_m if bar_m >= 1 else "%.1f m" % bar_m
        self.screen.blit(self.fs.render(label, True, C_TEXT), (bx, by - 20))
        zoom_txt = "zoom %.2f" % r.dist
        img = self.fs.render(zoom_txt, True, C_TEXT_DIM)
        self.screen.blit(img, (rect.right - img.get_width() - 12, rect.y + 4))

    def draw_checklist(self):
        w, h = 560, 680
        x = (self.W - w) // 2
        y = (self.H - h) // 2
        panel(self.screen, x, y, w, h, alpha=245)
        pygame.draw.rect(self.screen, C_ACCENT, (x, y, w, h), 2, border_radius=6)
        self.screen.blit(self.fbig.render("VERIFICATION CHECKLIST", True, C_ACCENT), (x + 20, y + 12))
        self.screen.blit(self.fs.render("V or click outside to close", True, C_TEXT_DIM),
                         (x + 20, y + 48))
        items = [
            ("80 air capture fans (8 fans each)", "%.0f m x %.0f m, %d fans, %.0f m diam" % (
                DIMS["contactor_w_m"], DIMS["contactor_h_m"],
                DIMS["contactor_fans"], DIMS["fan_d_m"])),
            ("CO2 filter material", "%.0f t, grabs %.2f kg CO2/kg, %d layers" % (
                DIMS["sorbent_t_per_bed"], DIMS["sorbent_cap_kg_kg"],
                DIMS["sorbent_layers"])),
            ("Staggered cycle (take turns)", "capture %.0f / release %.0f / cool %.0f min" % (
                SORBENT["cycle_capture_min"], SORBENT["cycle_regen_min"],
                SORBENT["cycle_cool_min"])),
            ("Heating chamber", "%.0f C, %d heater rows, insulated" % (
                DIMS["regen_temp_c"], DIMS["regen_heater_rows"])),
            ("Vacuum pump (pulls CO2 out)", "%.0f kWh/t electrical" % ENERGY["vacuum_elec_kwh_t"]),
            ("CO2 collection pipes", "%d pipe rows" % DIMS["manifold_rows"]),
            ("Valve system (3 auto valves)", "intake / exhaust / CO2 output"),
            ("Solar panels", "%.0f m2, %.0f MW peak, %.0f%% eff" % (
                DIMS["solar_pv_m2"], SOLAR_PV_PEAK_KW / 1000, DIMS["solar_pv_eff"] * 100)),
            ("Solar mirrors (curved troughs)", "%.0f m2, %.0f MW peak heat" % (
                DIMS["trough_aperture_m2"], SOLAR_TH_PEAK_KW / 1000)),
            ("Heat battery (molten salt)", "%.0f MWh, %.0f C / %.0f C" % (
                DIMS["thermal_storage_mwh"],
                DIMS["salt_hot_temp_c"], DIMS["salt_cold_temp_c"])),
            ("Wind turbines", "%d x %.0f MW = %.0f MW" % (
                DIMS["wind_turbines"], DIMS["turbine_rated_mw"],
                DIMS["wind_turbines"] * DIMS["turbine_rated_mw"])),
            ("Geothermal wells (always on)", "%d wells, %.0f MW heat" % (
                DIMS["geo_wells"], DIMS["geo_mw_thermal"])),
            ("CO2 compressors", "%d units, squeeze to %.0f bar" % (
                FACILITY["compressors"], DIMS["storage_bar"])),
            ("CO2 storage tanks", "%d x %.0f t = %.0f t buffer" % (
                DIMS["storage_tanks"], DIMS["storage_capacity_t"],
                DIMS["storage_tanks"] * DIMS["storage_capacity_t"])),
            ("CO2 pipeline to underground storage", "%.0f t/h, %.0f m on-site" % (
                CO2_STORE["pipeline_rate_t_h"], DIMS["pipeline_len_m"])),
            ("Battery bank", "%.0f MWh Li-ion" % DIMS["battery_mwh"]),
            ("Cooling towers", "%d x %.0f m" % (DIMS["cooling_towers"], DIMS["cooling_h_m"])),
            ("Control building", "%d staff, 24/7" % FACILITY["staff"]),
            ("Renewables-first synergy controller", "thermal+PV+wind+geo, auto-bed-adjust"),
            ("To-scale 3D + inspector + 4 views", "real SI dims, hover/click, full/explode/assembly/section"),
        ]
        yy = y + 72
        for name, detail in items:
            pygame.draw.circle(self.screen, C_GOOD, (x + 28, yy + 6), 5)
            self.screen.blit(self.fs.render("[x] " + name, True, C_TEXT), (x + 42, yy))
            self.screen.blit(self.fmicro.render(detail, True, C_TEXT_DIM), (x + 42, yy + 16))
            yy += 30
        self._preview_hitboxes["checklist_close"] = pygame.Rect(0, 0, self.W, self.H)

    def draw_operation(self):
        rect = self.view_rect()
        c = self.campaign
        pt = self.pt
        top, bot = sky_colors(c.sun)
        horizon = rect.y + int(rect.h * 0.46)
        sky = pygame.Surface((rect.w, horizon - rect.y))
        vgradient(sky, top, _mix(top, bot, 0.6))
        self.screen.blit(sky, (rect.x, rect.y))
        self._draw_sun_moon(rect, horizon, c)
        # ground
        ground_col = _mix(C_GROUND, C_GROUND_NIGHT, 1.0 - c.sun)
        ground = pygame.Surface((rect.w, rect.bottom - horizon))
        vgradient(ground, _mix(ground_col, top, 0.15), ground_col)
        self.screen.blit(ground, (rect.x, horizon))
        # render facility broadside
        r = self.fac_rend
        stash = (r.az, r.el, r.dist, r.pan.copy(), r.view, r.section)
        r.az = 1.62 + 0.03 * math.sin(pygame.time.get_ticks() / 12000.0)
        r.el = 0.25
        r.dist = 4.50
        r.pan = np.array([0.0, -rect.h * 0.06])
        r.view, r.section = "full", False
        fac_rect = pygame.Rect(290, horizon - int(rect.h * 0.20),
                               self.W - 290 - 354, int(rect.h * 0.50))
        r.render(self.screen, fac_rect, self.ang,
                 co2_glow=None, heat_glow=None, mouse_pos=None,
                 show_labels=False, label_font=None, interactive=False)
        r.az, r.el, r.dist, r.pan, r.view, r.section = stash
        self._draw_operation_hud(rect)
        self._draw_longevity_panel(rect)
        self._draw_weekly_ops_panel(rect)
        self._draw_progress_strip(rect)
        if c.finished:
            self._draw_finished(rect)

    def _draw_sun_moon(self, rect, horizon, c):
        h = c.hour_of_day
        if 5.5 <= h <= 18.5:
            t = (h - 5.5) / 13.0
            sx = rect.x + int(t * rect.w)
            sy = horizon - int(math.sin(t * math.pi) * (horizon - rect.y) * 0.82) - 8
            glow = pygame.Surface((120, 120), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 230, 150, 60), (60, 60), 54)
            pygame.draw.circle(glow, (255, 240, 190, 120), (60, 60), 30)
            self.screen.blit(glow, (sx - 60, sy - 60))
            pygame.draw.circle(self.screen, (255, 240, 180), (sx, sy), 18)
        else:
            hm = (h + 12) % 24
            t = clamp((hm - 5.5) / 13.0)
            sx = rect.x + int(t * rect.w)
            sy = horizon - int(math.sin(clamp(t) * math.pi) * (horizon - rect.y) * 0.7) - 8
            pygame.draw.circle(self.screen, (220, 224, 235), (sx, sy), 13)
            pygame.draw.circle(self.screen, _mix(C_SKY_NIGHT1, (0, 0, 0), 0.3),
                               (sx + 6, sy - 4), 12)
            rng = np.random.RandomState(7)
            for _ in range(70):
                stx = rect.x + int(rng.rand() * rect.w)
                sty = rect.y + int(rng.rand() * (horizon - rect.y) * 0.9)
                if rng.rand() < 0.5:
                    self.screen.set_at((stx, sty), (200, 210, 230))

    def _draw_operation_hud(self, rect):
        pt = self.pt
        c = self.campaign
        w, x = 340, self.W - 354
        y = 44
        h = 810
        panel(self.screen, x, y, w, h, alpha=255)
        mode_col = {"PEAK SOLAR -- FULL CAPTURE": C_GOOD,
                    "NIGHT OPERATION": (150, 160, 210),
                    "LOW POWER -- REDUCED CAPTURE": C_BAD,
                    "GEOTHERMAL BASELOAD": C_WARN}.get(pt.mode, C_CO2)
        self.screen.blit(self.fb.render(pt.mode, True, mode_col), (x + 12, y + 8))
        day_in_year = c.day - (c.year_num - 1) * 365
        clock = "Year %d  Day %d  %02d:%02d" % (c.year_num, day_in_year,
                                               int(c.hour_of_day),
                                               int((c.hour_of_day % 1) * 60))
        cimg = self.fs.render(clock, True, C_TEXT_DIM)
        self.screen.blit(cimg, (x + w - cimg.get_width() - 12, y + 12))

        yy = y + 42
        self.screen.blit(self.fbig.render("%.0f" % pt.capture_rate_t_h, True, C_TEXT), (x + 12, yy))
        self.screen.blit(self.font.render("t CO2/h", True, C_TEXT_DIM), (x + 130, yy + 14))
        # Cost per tonne (live estimate)
        total_cost = pt.cumulative_opex + pt.cumulative_sorbent_cost + pt.cumulative_battery_cost
        cost_per_t = total_cost / max(1.0, pt.co2_captured_t)
        self.screen.blit(self.fbig.render("$%.0f" % cost_per_t, True, C_GOOD if cost_per_t < 50 else C_WARN), (x + 200, yy))
        self.screen.blit(self.font.render("cost/t OPEX", True, C_TEXT_DIM), (x + 256, yy + 14))
        yy += 52

        def gbar(lab, frac, color, val):
            nonlocal yy
            bar(self.screen, self.fs, x + 14, yy + 14, w - 30, 12, frac, color, lab, val)
            yy += 36

        gbar("BATTERY", pt.soc, C_GOOD if pt.soc > 0.3 else C_WARN,
             "%.0f%%" % (pt.soc * 100))
        gbar("HEAT BATTERY", pt.thermal_frac, C_WARN,
             "%.0f%%" % (pt.thermal_frac * 100))
        gbar("CO2 TANKS", pt.co2_storage_frac, C_CO2,
             "%.0f t" % pt.co2_storage_t)
        gbar("CAMPAIGN", c.progress, C_ACCENT,
             "Yr %d / %.0f" % (c.year_num, c.duration_years))

        yy += 2
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 12, yy), (x + w - 12, yy), 1)
        yy += 8
        self.screen.blit(self.fs.render("LIVE ENERGY (kW in/out)", True, C_ACCENT), (x + 14, yy))
        yy += 18
        flows = [
            ("Solar panels", pt.flow["solar_pv"], (86, 150, 230)),
            ("Solar mirrors", pt.flow["solar_th"], (255, 200, 60)),
            ("Wind", pt.flow["wind"], (220, 225, 230)),
            ("Geothermal", pt.flow["geo"], (160, 100, 60)),
            ("Heat from store", pt.flow["thermal_store"], C_WARN),
            ("Heat for CO2 release", -pt.flow["regen_thermal"], C_REGEN_HOT),
            ("Fan electricity", -pt.flow["fan_elec"], C_ACCENT),
            ("Vacuum pump", -pt.flow["vacuum_elec"], C_COMP),
            ("Compressor", -pt.flow["compress_elec"], C_COMP_HOT),
            ("Battery net", pt.flow["batt_net"], C_GOOD if pt.flow["batt_net"] >= 0 else C_BAD),
            ("Thermal net", pt.flow["thermal_net"], C_WARN if pt.flow["thermal_net"] >= 0 else C_BAD),
        ]
        for lab, val, col in flows:
            self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM), (x + 14, yy))
            s = "%+.0f kW" % val
            img = self.fs.render(s, True, col)
            self.screen.blit(img, (x + w - 16 - img.get_width(), yy))
            yy += 18

        yy += 6
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 12, yy), (x + w - 12, yy), 1)
        yy += 8
        self.screen.blit(self.fs.render("15-YEAR TOTALS (so far)", True, C_CO2), (x + 14, yy))
        yy += 18
        ref_t = ref_emitter_t(c.progress)
        # Capacity factor: actual capture vs theoretical max
        cap_factor = pt.capture_hours / max(1.0, pt.elapsed_years * 8760.0) if pt.elapsed_years > 0 else 0.0
        # Net CO2 benefit (captured - embodied amortized)
        embodied_amort = CAMPAIGN["embodied_co2_t"] * c.progress
        net_co2 = pt.co2_captured_t - embodied_amort
        stats = [
            ("CO2 captured", "%.0f t" % pt.co2_captured_t),
            ("CO2 underground", "%.0f t" % pt.co2_sequestered_t),
            ("Net CO2 benefit", "%.0f t" % net_co2),
            ("= cars off road", "%.0f k" % (pt.co2_captured_t / 4.6 / 1000)),
            ("CO2 purity", "%.2f%%" % (pt.co2_purity * 100)),
            ("Uptime", "%.1f%%" % (cap_factor * 100)),
            ("Peak capture", "%.0f t/h" % pt.peak_capture_t_h),
            ("Filters active", "%d / %d" % (pt.beds_active, FACILITY["sorbent_beds"])),
            ("Fan speed", "%.0f rpm" % pt.fan_rpm),
            ("Release temp", "%.0f C" % pt.regen_temp_c),
            ("Water used", "%.0f m3" % pt.water_used_m3),
            ("Solar electricity", "%.1f GWh" % (pt.solar_pv_kwh / 1e6)),
            ("Solar heat", "%.1f GWh" % (pt.solar_th_kwh / 1e6)),
            ("Wind energy", "%.1f GWh" % (pt.wind_kwh / 1e6)),
            ("Geothermal", "%.1f GWh" % (pt.geo_kwh / 1e6)),
            ("Total cost", "$%.1fM" % (total_cost / 1e6)),
            ("Filters replaced", "%d ($%.1fM)" % (pt.sorbent_replaced_count, pt.cumulative_sorbent_cost / 1e6)),
            ("Batteries replaced", "%d ($%.1fM)" % (pt.battery_replaced_count, pt.cumulative_battery_cost / 1e6)),
            ("Availability", "%.1f%%" % (pt.availability * 100)),
            ("Downtime", "%.0f h" % pt.downtime_h),
            ("Wind now", "%.1f m/s" % c.wind_ms),
            ("Weather", "%.0f%%" % (c.weather * 100)),
            ("Time warp", "x%.0f" % c.warp),
        ]
        for lab, val in stats:
            self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM), (x + 14, yy))
            img = self.fs.render(val, True, C_TEXT)
            self.screen.blit(img, (x + w - 16 - img.get_width(), yy))
            yy += 17

        # --- Global deployment plan ---
        yy += 6
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 12, yy), (x + w - 12, yy), 1)
        yy += 8
        self.screen.blit(self.fs.render("GLOBAL DEPLOYMENT PLAN", True, C_CO2), (x + 14, yy))
        yy += 17
        # This plant's contribution
        this_plant_15yr = FACILITY["capture_t_year"] * CAMPAIGN["years"] / 1e6
        plan_stats = [
            ("This giga-plant (15 yr)", "%.0f Mt CO2" % this_plant_15yr),
            ("Phase 1: 2026-30", "6-24 giga-plants"),
            ("  -> 200 Mt-1 Gt/yr", "$20-100B invested"),
            ("Phase 2: 2030-35", "~235 giga-plants"),
            ("  -> ~10 Gt/yr", "$~10B cumulative"),
            ("For 1 Gt/yr target", "~%.0f giga-plants needed" % (
                1e9 / FACILITY["capture_t_year"])),
            ("For net-zero (~4 Gt)", "~%.0f giga-plants" % (
                4e9 / FACILITY["capture_t_year"])),
            ("  (with 90%% cuts)", "vs 47,000 old 850kt plants"),
            ("Per plant cost", "$%.1fB CAPEX + $1/t OPEX" % (
                FACILITY["capture_t_year"] / 1e3)),
            ("All-in cost/t", "$10-30/t at 50x scale"),
        ]
        for lab, val in plan_stats:
            self.screen.blit(self.fmicro.render(lab, True, C_TEXT_DIM), (x + 14, yy))
            img = self.fmicro.render(val, True, C_CO2 if "plant" in lab.lower() else C_TEXT)
            self.screen.blit(img, (x + w - 16 - img.get_width(), yy))
            yy += 14

    def _draw_longevity_panel(self, rect):
        pt = self.pt
        c = self.campaign
        w, x = 280, 8
        y = 44
        h = 810
        panel(self.screen, x, y, w, h, alpha=255)
        self.screen.blit(self.fb.render("PLANT HEALTH", True, C_ACCENT), (x + 12, y + 8))
        yy = y + 40

        # Component health bars
        comp_labels = {
            "fans": "Fans", "sorbent": "CO2 filters", "regen_units": "CO2 release units",
            "vacuum_pumps": "Vacuum pumps", "compressors": "Compressors",
            "solar_pv": "Solar panels", "solar_thermal": "Solar mirrors",
            "wind_turbines": "Wind turbines", "geothermal": "Geothermal",
            "battery": "Battery", "thermal_store": "Heat battery",
            "co2_tanks": "CO2 tanks", "pipeline": "Pipeline",
            "control_system": "Control system",
        }
        for cname in comp_labels:
            health = pt.component_health.get(cname, 1.0)
            col = C_GOOD if health > 0.8 else (C_WARN if health > 0.6 else C_BAD)
            bar(self.screen, self.fs, x + 14, yy + 14, w - 30, 10, health, col,
                comp_labels[cname], "%.0f%%" % (health * 100))
            yy += 30

        yy += 4
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 12, yy), (x + w - 12, yy), 1)
        yy += 8
        self.screen.blit(self.fs.render("MAINTENANCE SUMMARY", True, C_ACCENT), (x + 14, yy))
        yy += 18
        ms = pt.maintenance_summary
        mstats = [
            ("Availability", "%.1f%%" % (ms["availability"] * 100)),
            ("Breakdowns", "%d" % ms["total_failures"]),
            ("Prevented", "%d (%.0f%%)" % (ms["total_prevented"], ms["prevention_rate"] * 100)),
            ("Downtime", "%.0f h" % ms["downtime_h"]),
            ("Maint hours", "%.0f h" % ms["maintenance_h"]),
            ("Filters replaced", "%d" % ms["sorbent_replaced"]),
            ("Batteries replaced", "%d" % ms["battery_replaced"]),
        ]
        for lab, val in mstats:
            self.screen.blit(self.fs.render(lab, True, C_TEXT_DIM), (x + 14, yy))
            img = self.fs.render(val, True, C_TEXT)
            self.screen.blit(img, (x + w - 16 - img.get_width(), yy))
            yy += 17

        yy += 6
        pygame.draw.line(self.screen, C_PANEL_HI, (x + 12, yy), (x + w - 12, yy), 1)
        yy += 8
        self.screen.blit(self.fs.render("MAINTENANCE LOG", True, C_ACCENT), (x + 14, yy))
        yy += 18
        # Show last 8 maintenance log entries
        log = pt.maintenance_log[-8:] if pt.maintenance_log else []
        if not log:
            self.screen.blit(self.fs.render("(no events yet)", True, C_TEXT_DIM), (x + 14, yy))
            yy += 16
        for entry in log:
            yr, comp, desc = entry
            comp_display = comp_labels.get(comp, comp)
            # Truncate long descriptions
            if len(desc) > 38:
                desc = desc[:36] + ".."
            col = C_BAD if "FAILURE" in desc else (C_WARN if "REPLAC" in desc else C_TEXT_DIM)
            self.screen.blit(self.fs.render("Y%d %s" % (yr, comp_display), True, C_TEXT_DIM), (x + 14, yy))
            self.screen.blit(self.fs.render(desc, True, col), (x + 70, yy))
            yy += 16

        # Year-by-year snapshot summary (if available)
        if c.year_history:
            yy += 6
            pygame.draw.line(self.screen, C_PANEL_HI, (x + 12, yy), (x + w - 12, yy), 1)
            yy += 8
            self.screen.blit(self.fs.render("YEAR-BY-YEAR", True, C_ACCENT), (x + 14, yy))
            yy += 17
            for snap in c.year_history[-5:]:
                yr = snap["year"]
                co2 = snap["co2_captured_t"]
                cf = snap["capacity_factor"] * 100
                cost = snap["cost_per_t"]
                avail = snap["availability"] * 100
                col = C_GOOD if cost < 200 else C_WARN
                line = "Y%d: %.0f kt, %.0f%% CF, $%.0f/t, %.1f%% up" % (yr, co2/1000, cf, cost, avail)
                self.screen.blit(self.fs.render(line, True, col), (x + 14, yy))
                yy += 16

    def _draw_weekly_ops_panel(self, rect):
        c = self.campaign
        pt = self.pt
        # Bottom-center panel for weekly ops data
        pw = self.W - 290 - 354 - 16
        px = 298
        py = rect.bottom - 165
        ph = 120
        panel(self.screen, px, py, pw, ph, alpha=255)

        # Header
        if c.time_jumping and 0 <= c.jump_index < len(c.week_history):
            snap = c.week_history[c.jump_index]
            header = "WEEKLY OPS  --  VIEWING WEEK %d (Year %d)  [<- -> browse, HOME=live]" % (
                snap["week"], snap["year"])
            hdr_col = C_ACCENT
        elif c.week_history:
            snap = c.week_history[-1]
            header = "WEEKLY OPS  --  LATEST WEEK %d (Year %d)  [<- -> jump in time]" % (
                snap["week"], snap["year"])
            hdr_col = C_CO2
        else:
            header = "WEEKLY OPS  --  (collecting data...)"
            hdr_col = C_TEXT_DIM
            self.screen.blit(self.fs.render(header, True, hdr_col), (px + 12, py + 8))
            return

        self.screen.blit(self.fs.render(header, True, hdr_col), (px + 12, py + 8))

        if not c.week_history:
            return

        # Use the selected snapshot or the latest
        if c.time_jumping and 0 <= c.jump_index < len(c.week_history):
            snap = c.week_history[c.jump_index]
        else:
            snap = c.week_history[-1]

        yy = py + 28
        col_w = (pw - 24) // 4

        # Column 1: Production
        cx1 = px + 12
        self.screen.blit(self.fs.render("PRODUCTION", True, C_ACCENT), (cx1, yy))
        prod_lines = [
            ("CO2 captured", "%.0f t" % snap["week_co2_t"]),
            ("CO2 underground", "%.0f t" % snap["co2_sequestered_t"]),
            ("Uptime", "%.1f%%" % (snap["capacity_factor"] * 100)),
            ("CO2 purity", "%.2f%%" % (snap["co2_purity"] * 100)),
            ("Filters active", "%d / %d" % (snap["beds_active"], FACILITY["sorbent_beds"])),
        ]
        for i, (lab, val) in enumerate(prod_lines):
            self.screen.blit(self.fmicro.render(lab, True, C_TEXT_DIM), (cx1, yy + 16 + i * 14))
            img = self.fmicro.render(val, True, C_TEXT)
            self.screen.blit(img, (cx1 + col_w - 10 - img.get_width(), yy + 16 + i * 14))

        # Column 2: Costs
        cx2 = px + 12 + col_w
        self.screen.blit(self.fs.render("COSTS", True, C_GOOD), (cx2, yy))
        cost_lines = [
            ("Week cost", "$%.1fk" % (snap["week_cost"] / 1000)),
            ("Cost per tonne", "$%.0f/t" % snap["week_cost_per_t"]),
            ("Total so far", "$%.2fM" % (snap["cumulative_cost"] / 1e6)),
            ("Filters replaced", "%d" % snap["sorbent_replaced"]),
            ("Batteries replaced", "%d" % snap["battery_replaced"]),
        ]
        for i, (lab, val) in enumerate(cost_lines):
            col = C_GOOD if i == 1 and snap["week_cost_per_t"] < 200 else C_TEXT
            self.screen.blit(self.fmicro.render(lab, True, C_TEXT_DIM), (cx2, yy + 16 + i * 14))
            img = self.fmicro.render(val, True, col)
            self.screen.blit(img, (cx2 + col_w - 10 - img.get_width(), yy + 16 + i * 14))

        # Column 3: Energy
        cx3 = px + 12 + col_w * 2
        self.screen.blit(self.fs.render("ENERGY (this week)", True, C_WARN), (cx3, yy))
        energy_lines = [
            ("Solar panels", "%.1f MWh" % (snap["week_solar_kwh"] / 1000)),
            ("Solar mirrors", "%.1f MWh" % (snap["week_thermal_kwh"] / 1000)),
            ("Wind", "%.1f MWh" % (snap["week_wind_kwh"] / 1000)),
            ("Geothermal", "%.1f MWh" % (snap["week_geo_kwh"] / 1000)),
            ("Availability", "%.1f%%" % (snap["availability"] * 100)),
        ]
        for i, (lab, val) in enumerate(energy_lines):
            self.screen.blit(self.fmicro.render(lab, True, C_TEXT_DIM), (cx3, yy + 16 + i * 14))
            img = self.fmicro.render(val, True, C_TEXT)
            self.screen.blit(img, (cx3 + col_w - 10 - img.get_width(), yy + 16 + i * 14))

        # Column 4: Water & Summary
        cx4 = px + 12 + col_w * 3
        self.screen.blit(self.fs.render("WATER & STATUS", True, (100, 180, 220)), (cx4, yy))
        water_lines = [
            ("Water used", "%.0f m3" % snap["week_water_m3"]),
            ("Water per t CO2", "%.1f m3/t" % (snap["week_water_m3"] / max(1.0, snap["week_co2_t"]))),
            ("CO2 in tanks", "%.0f t (%.0f%%)" % (snap["co2_storage_t"], snap["co2_storage_frac"] * 100)),
            ("Fan speed", "%.0f rpm" % snap["fan_rpm"]),
            ("Time warp", "x%.0f" % c.warp),
        ]
        for i, (lab, val) in enumerate(water_lines):
            self.screen.blit(self.fmicro.render(lab, True, C_TEXT_DIM), (cx4, yy + 16 + i * 14))
            img = self.fmicro.render(val, True, C_TEXT)
            self.screen.blit(img, (cx4 + col_w - 10 - img.get_width(), yy + 16 + i * 14))

        # Mini sparkline of weekly CO2 production (last 12 weeks)
        if len(c.week_history) >= 2:
            spark_x = px + 12
            spark_y = py + ph - 12
            spark_w = pw - 24
            spark_h = 10
            recent = c.week_history[-12:]
            vals = [s["week_co2_t"] for s in recent]
            vmax = max(vals) if max(vals) > 0 else 1.0
            for i in range(len(recent)):
                frac = vals[i] / vmax
                bx = spark_x + int(i * spark_w / max(1, len(recent)))
                bw = max(2, int(spark_w / max(1, len(recent))) - 1)
                bh = max(1, int(frac * spark_h))
                col = C_CO2 if not (c.time_jumping and c.jump_index == len(c.week_history) - len(recent) + i) else C_ACCENT
                pygame.draw.rect(self.screen, col, (bx, spark_y - bh, bw, bh))

    def _draw_progress_strip(self, rect):
        c = self.campaign
        y = rect.bottom - 12
        x0, x1 = 300, self.W - 380
        pygame.draw.line(self.screen, C_TEXT_DIM, (x0, y), (x1, y), 2)

        # Year tick marks
        for yr in range(1, int(c.duration_years) + 1):
            tx = int(x0 + (yr - 1) / max(1, c.duration_years - 1) * (x1 - x0))
            col = C_ACCENT if yr == c.year_num else C_TEXT_DIM
            pygame.draw.circle(self.screen, col, (tx, y), 3)
            if yr == 1 or yr == int(c.duration_years) or yr % 5 == 0:
                lab = "Y%d" % yr
                img = self.fmicro.render(lab, True, col)
                self.screen.blit(img, (tx - img.get_width() // 2, y - 14))

        # Live position marker
        sx = int(x0 + c.progress * (x1 - x0))
        pygame.draw.circle(self.screen, C_CO2, (sx, y), 5)
        pygame.draw.circle(self.screen, C_TEXT, (sx, y), 5, 1)

        # Jump position marker (when browsing history)
        if c.time_jumping and 0 <= c.jump_index < len(c.week_history):
            snap = c.week_history[c.jump_index]
            jfrac = snap["elapsed_h"] / c.duration_h
            jx = int(x0 + jfrac * (x1 - x0))
            pygame.draw.circle(self.screen, C_ACCENT, (jx, y), 4)
            pygame.draw.circle(self.screen, C_TEXT, (jx, y), 4, 1)
            jlab = self.fmicro.render("W%d" % snap["week"], True, C_ACCENT)
            self.screen.blit(jlab, (jx - jlab.get_width() // 2, y - 14))
            pygame.draw.line(self.screen, C_PANEL_HI, (jx, y), (sx, y), 1)

    def _draw_finished(self, rect):
        pt = self.pt
        c = self.campaign
        w, h = 560, 340
        x = rect.x + (rect.w - w) // 2
        y = rect.y + (rect.h - h) // 2 - 40
        panel(self.screen, x, y, w, h, alpha=235)
        self.screen.blit(self.fbig.render("15-YEAR CAMPAIGN COMPLETE", True, C_CO2),
                         (x + 24, y + 18))
        ref_t = ref_emitter_t(1.0) * c.duration_years
        total_cost = pt.cumulative_opex + pt.cumulative_sorbent_cost + pt.cumulative_battery_cost
        cost_per_t = total_cost / max(1.0, pt.co2_captured_t)
        embodied_amort = CAMPAIGN["embodied_co2_t"]
        net_co2 = pt.co2_captured_t - embodied_amort
        lines = [
            "CO2 captured:   %.0f t over %.0f years" % (pt.co2_captured_t, c.duration_years),
            "CO2 sequestered: %.0f t (offsetting %.0f t/year emitter)" % (
                pt.co2_sequestered_t, ref_t / c.duration_years),
            "Net CO2 benefit: %.0f t (after %.0f t embodied carbon)" % (net_co2, embodied_amort),
            "Cost per tonne:  $%.0f/t (target: $%.0f/t)" % (cost_per_t, CAMPAIGN["cost_per_t_target"]),
            "Water consumed:  %.0f m3 (%.1f m3/t CO2)" % (
                pt.water_used_m3, pt.water_used_m3 / max(1.0, pt.co2_captured_t)),
            "Availability:    %.1f%%  (downtime: %.0f h)" % (pt.availability * 100, pt.downtime_h),
            "Sorbent replaced: %d beds  Battery replaced: %d racks" % (
                pt.sorbent_replaced_count, pt.battery_replaced_count),
            "Energy: all renewable (solar PV + thermal + wind + geothermal)",
        ]
        for i, ln in enumerate(lines):
            self.screen.blit(self.font.render(ln, True, C_TEXT), (x + 24, y + 62 + i * 24))

    def draw_help(self):
        w, h = 560, 460
        x = (self.W - w) // 2
        y = (self.H - h) // 2
        panel(self.screen, x, y, w, h, alpha=240)
        self.screen.blit(self.fb.render("CONTROLS", True, C_ACCENT), (x + 20, y + 16))
        lines = [
            "TAB            cycle FACILITY / CAPTURE / OPERATION / URBAN",
            "or click       mode tabs in the top bar",
            "mouse drag     orbit the model",
            "right-drag     pan     mouse wheel  zoom",
            "click part     pin it in the inspector (left sidebar)",
            "click PARTS    left sidebar list to select any part",
            "1 2 3 4        full / exploded / assembly / section-cut",
            "E              quick exploded toggle",
            "X              cross-section half-cut toggle",
            "L              toggle part labels",
            "R              reset the camera",
            "[  ]           step the assembly build",
            "A / C          assemble all / clear",
            ",  .           slow / speed up TIME-WARP (OPERATION)",
            "",
            "OPERATION -- TIME JUMP CONTROLS:",
            "LEFT / RIGHT   jump 1 week back / forward in time",
            "SHIFT+LEFT/RT  jump 4 weeks back / forward",
            "HOME           return to LIVE (current time)",
            "Weekly ops panel shows production, costs, energy,",
            "water, and a sparkline of recent CO2 output.",
            "V              verification checklist overlay",
            "I              full informational specification",
            "F11            toggle fullscreen (window is resizable)",
            "H              this help     ESC exits fullscreen / Q quit",
        ]
        for i, ln in enumerate(lines):
            self.screen.blit(self.font.render(ln, True, C_TEXT), (x + 24, y + 52 + i * 22))

    def draw_info(self):
        w, h = 760, self.H - 90
        x = (self.W - w) // 2
        y = 50
        panel(self.screen, x, y, w, h, alpha=244)
        self.screen.blit(self.fb.render("CARBON CAPTURE  --  FULL SPECIFICATION", True, C_CO2),
                         (x + 20, y + 14))
        self.screen.blit(self.fs.render("scroll: up/down arrows   close: I", True,
                                        C_TEXT_DIM), (x + w - 250, y + 20))
        clip = pygame.Rect(x + 16, y + 46, w - 32, h - 60)
        self.screen.set_clip(clip)
        yy = y + 50 - self.info_scroll
        maxpx = w - 60
        total = 0
        for head, lines in self.info_sections:
            if yy > y + 30 and yy < y + h:
                self.screen.blit(self.fb.render(head, True, C_CO2), (x + 24, yy))
            yy += 26
            total += 26
            for ln in lines:
                for wl in wrap_text(self.font, ln, maxpx):
                    if y + 30 < yy < y + h:
                        self.screen.blit(self.font.render(wl, True, C_TEXT), (x + 30, yy))
                    yy += 20
                    total += 20
            yy += 12
            total += 12
        self.screen.set_clip(None)
        self._info_total = total
        self.info_scroll = max(0, min(self.info_scroll, max(0, total - (h - 70))))

    def run(self):
        _print_banner()
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            dt = min(dt, 0.05)
            self.handle_events(dt)
            self.update(dt)
            self.draw()
        pygame.quit()


def _print_banner():
    print("=" * 70)
    print(" CC.py  --  CARBON CAPTURE GIGA-PLANT (Direct Air Capture)")
    print("=" * 70)
    print(" A 3D model of a multi-story giga-facility that removes CO2")
    print(" from the air. 50-story design = 50x fewer plants needed.")
    print(" Powered entirely by clean energy (solar, wind, geothermal).")
    print()
    print(" Modes (TAB):  FACILITY (whole plant)  |  CAPTURE (close-up)")
    print("               OPERATION (live sim)    |  URBAN (skyscraper mini-plant)")
    print(" Target:       %.1f Mt CO2/year -- like taking %.1fM cars off the road" % (
        FACILITY["capture_t_year"] / 1e6, FACILITY["capture_t_year"] / 4.6 / 1e6))
    print(" Urban:        %.0f t CO2/year per unit -- DAC inside vacant skyscraper floors" % (
        URBAN["capture_t_year"]))
    print(" Energy:       solar panels + solar mirrors + wind + geothermal")
    print("               ($0 energy cost, no fossil fuels)")
    print(" Controls:     H = help,  I = plain-English info,  V = checklist")
    print("               F11 = fullscreen,  ESC = quit")
    print(" OPERATION:    <- -> = jump weeks,  SHIFT+<- -> = jump 4 weeks,  HOME = live")
    print("=" * 70)


def main():
    App().run()


if __name__ == "__main__":
    main()
