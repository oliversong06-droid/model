# Bio-Chemical Parameters for Ecological Modeling

This document summarizes key scientific parameters used in the biological and chemical sub-models of the oil spill simulation. These values are derived from scientific literature and experimental data.

## 1. Dissolved Oxygen (DO) Consumption ( $k_{consume}$ )

The consumption of dissolved oxygen is primarily driven by microbial biodegradation of oil hydrocarbons. This process is aerobic and temperature-dependent.

- **Parameter:** Biodegradation Rate Constant ($k_{bio}$)
- **Value:** ~0.05 - 0.1 day $^{-1}$ (First-order decay rate)
- **Derived from:** Half-life of dispersed oil in marine waters, typically **7 to 14 days** [1].
  - $k = \ln(2) / t_{1/2}$
  - For $t_{1/2} = 7$ days, $k \approx 0.1$ day $^{-1}$.
- **Oxygen Stoichiometry:** ~3.5 g O $_2$ consumed per g of Hydrocarbon oxidized (theoretical max).
  - _Note:_ In practice, nutrient limitations (N, P) often control the rate [2].
- **Threshold:** Biodegradation rates decrease sharply below DO concentrations of **2.0 - 3.0 mg/L** [3].

**References:**

1.  _Prince, R. C., et al. (2013). "The primary biodegradation of dispersed crude oil in the sea."_
2.  _Atlas, R. M., & Hazen, T. C. (2011). "Oil biodegradation and bioremediation: a tale of the two worst spills in US history."_
3.  _NOAA. "Oxygen Depletion in Oil Spills."_

## 2. Reaeration Rate ( $k_{reaer}$ )

The reaeration rate represents the transfer of oxygen from the atmosphere to the water column. Oil slicks act as a physical barrier and dampen surface turbulence, significantly reducing this rate.

- **Baseline Marine Reaeration ($k_{reaer, base}$):** ~0.1 - 0.5 day $^{-1}$ (highly dependent on wind/waves).
  - _Mangrove Estuary Gas Transfer Velocity:_ ~3.3 cm/h (~0.8 m/day) [4].
- **Oil Impact:** Oil slicks can reduce reaeration by **40% to 90%** depending on slick thickness and type [5].
- **Modeling Approach:**
  - $k_{reaer, oil} = k_{reaer, base} \times (1 - \text{ReductionFactor})$
  - Reduction Factor $\approx$ 0.8 for thick slicks.

**References:** 4. _Ho, D. T., et al. (2016). "Gas transfer velocities in a mangrove estuary."_ 5. _Fingas, M. (2011). "Oil Spill Science and Technology."_

## 3. Plankton Mortality & Toxicity Thresholds

Plankton sensitivity varies by species and oil composition (especially PAHs). Dispersants often increase toxicity.

### Zooplankton

- **Lethal Concentration (LC50):**
  - **Crude Oil:** ~32.4 $\mu$ L/L (ppm) (16h exposure) [6].
  - **Dispersed Oil:** ~5 - 12 ppm (48-96h exposure) [7].
  - _Note:_ Dispersed oil is approx. 2-3x more toxic than crude oil alone.
- **Mortality Rate:** High immediate mortality in affected zones.
- **Recovery Time:** **2 to 4 weeks** for population recovery in open waters (due to recruitment from unaffected areas) [8].

### Phytoplankton

- **Growth Inhibition:**
  - **Stimulation:** < 1 mg/L (low concentrations may stimulate growth).
  - **Inhibition:** > 100 mg/L (severe inhibition) [9].
  - **EC50:** 1 - 100 mg/L.
- **Recovery Time:** **3 to 10 months** for full community structure recovery (blooms may occur earlier) [10].

**References:** 6. _Almeda, R., et al. (2013). "Effects of crude oil exposure on bioaccumulation and survival of mesozooplankton."_ 7. _Detailed search results on Corexit 9500A toxicity._ 8. _Daly, K. L., et al. (2016). "Herbivory and grazing dynamics in the Gulf of Mexico after the Deepwater Horizon oil spill."_ 9. _Ozhan, K., et al. (2014). "Phytoplankton responses to crude oil: a review."_ 10. _Abbriano, R. M., et al. (2011). "Deepwater Horizon oil spill: A review of the planktonic response."_

## 4. Summary Table for Simulation

| Parameter                | Variable Name   | Typical Value | Unit             | Notes                               |
| :----------------------- | :-------------- | :------------ | :--------------- | :---------------------------------- |
| Biodegradation Rate      | `k_bio`         | 0.07          | day $^{-1}$      | Assumes $t_{1/2} \approx 10$ days   |
| Oxygen Consumption       | `O2_demand`     | 3.0           | g O $_2$ / g Oil | Stoichiometric approx.              |
| Base Reaeration          | `k_reaer_base`  | 0.4           | day $^{-1}$      | Wind-dependent                      |
| Oil Reaeration Factor    | `oil_dampening` | 0.2           | dimensionless    | Multiplier for $k_{reaer}$ in slick |
| Zooplankton LC50         | `zoo_LC50`      | 30.0          | mg/L (ppm)       | 16-24h exposure                     |
| Phytoplankton Inhibition | `phyto_inhib`   | 100.0         | mg/L             | Concentration for growth stop       |
| Zooplankton Recovery     | `zoo_recov`     | 21            | days             | Time to return to baseline          |
