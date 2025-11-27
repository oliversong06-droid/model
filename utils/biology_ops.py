# biology_ops.py
"""
Biology mini-models for oil spill:
- DO reduction and recovery (Dissolved Oxygen)
- Microbial/Plankton response (Mortality & Recovery)
- Ecological Recovery Index

Scientific parameters derived from literature (see doc/bio_chem_params.md).
"""

import numpy as np

# --- Scientific Constants (converted to SI units: 1/s) ---
# Source: Prince et al. (2013), NOAA
DAYS_TO_SECONDS = 86400.0

# Biodegradation Rate (k_bio): ~0.07 day^-1
K_BIO_DEFAULT = 0.07 / DAYS_TO_SECONDS  # ~8.1e-7 s^-1

# Oxygen Consumption Stoichiometry: ~3.0 g O2 / g Hydrocarbon
O2_DEMAND_DEFAULT = 3.0

# Base Reaeration Rate (k_reaer): ~0.4 day^-1 (wind dependent)
K_REAER_BASE_DEFAULT = 0.4 / DAYS_TO_SECONDS  # ~4.6e-6 s^-1

# Oil Dampening Factor on Reaeration: ~20-80% reduction
OIL_DAMPENING_DEFAULT = 0.2  # 20% reduction by default, can go up to 0.8 for thick slicks

# Zooplankton LC50 (Lethal Concentration 50%): ~30.0 mg/L (16h)
ZOO_LC50_DEFAULT = 30.0

# Recovery Times (for simple linear recovery model)
RECOVERY_RATE_PLANKTON = 1.0 / (21.0 * DAYS_TO_SECONDS)  # ~3 weeks to recover


def update_DO(
    DO,
    DO_sat,
    oil_conc,
    k_consume=None,
    k_reaer=None,
    dt=3600.0,
    oil_dampening=OIL_DAMPENING_DEFAULT
):
    """
    Update Dissolved Oxygen (DO) concentration.
    
    Model: d(DO)/dt = Reaeration - Consumption
    
    Reaeration = k_reaer * (1 - dampening) * (DO_sat - DO)
    Consumption = k_consume * oil_conc
    *Note: k_consume here represents the effective O2 removal rate per unit oil.*

    Parameters
    ----------
    DO : np.ndarray
        Current DO [mg/L]
    DO_sat : np.ndarray or float
        Saturation DO [mg/L] (func of T, S)
    oil_conc : np.ndarray
        Oil concentration [mg/L]
    k_consume : float, optional
        O2 consumption rate constant. 
        If None, calculated as K_BIO_DEFAULT * O2_DEMAND_DEFAULT.
    k_reaer : float, optional
        Reaeration rate constant [1/s]. 
        If None, uses K_REAER_BASE_DEFAULT.
    dt : float
        Time step [s]
    oil_dampening : float
        Factor (0~1) reducing reaeration due to oil slick.

    Returns
    -------
    DO_new : np.ndarray
    """
    DO = np.asarray(DO, dtype=float)
    DO_sat = np.asarray(DO_sat, dtype=float)
    oil_conc = np.asarray(oil_conc, dtype=float)

    # Set defaults if not provided
    if k_consume is None:
        k_consume = K_BIO_DEFAULT * O2_DEMAND_DEFAULT
    if k_reaer is None:
        k_reaer = K_REAER_BASE_DEFAULT

    # 1. Reaeration with oil dampening effect
    # Oil slick acts as a barrier, reducing oxygen transfer.
    # Dampening can be function of oil thickness, here simplified as constant factor where oil > 0
    dampening_mask = (oil_conc > 1.0)  # Only apply significant dampening if oil is present (>1mg/L)
    effective_k_reaer = np.full_like(DO, k_reaer)
    effective_k_reaer[dampening_mask] *= (1.0 - oil_dampening)

    reaer_term = effective_k_reaer * (DO_sat - DO)

    # 2. Consumption (Biodegradation)
    # Limited by available DO (Monod kinetics could be used, here linear approx clipped)
    consume_term = k_consume * oil_conc
    
    # Prevent consumption from driving DO below 0 immediately (simple limiter)
    # In reality, rate slows down as DO -> 0.
    
    DO_new = DO + dt * (reaer_term - consume_term)
    
    # Clip to valid range [0, DO_sat]
    DO_new = np.clip(DO_new, 0.0, DO_sat)
    
    return DO_new


def plankton_response(
    plankton, 
    oil_conc, 
    sens_coeff=None, 
    dt=3600.0, 
    lc50=ZOO_LC50_DEFAULT,
    recovery_rate=RECOVERY_RATE_PLANKTON
):
    """
    Calculate Plankton Biomass response (Mortality & Recovery).

    Model:
    If Oil > Threshold: Mortality dominates (Sigmoidal or Linear based on LC50)
    If Oil ~ 0: Recovery (Logistic growth towards carrying capacity)

    Parameters
    ----------
    plankton : np.ndarray
        Current biomass (relative units, e.g., 0~1 or mg/L)
    oil_conc : np.ndarray
        Oil concentration [mg/L]
    sens_coeff : float, optional
        Legacy parameter for linear sensitivity. 
        If None, derived from LC50 logic.
    dt : float
        Time step [s]
    lc50 : float
        Lethal Concentration 50% [mg/L]
    recovery_rate : float
        Growth rate [1/s] when no oil is present.

    Returns
    -------
    plankton_new : np.ndarray
    """
    plankton = np.asarray(plankton, dtype=float)
    oil_conc = np.asarray(oil_conc, dtype=float)
    
    # --- Mortality Term ---
    # Use a Hill equation or simple ratio for toxicity:
    # Mortality Rate ~ Max_Rate * (C / (C + LC50))
    # Let's assume max mortality rate is high (e.g., 50% per day at LC50)
    # 50% loss in 16h (approx 57600s) -> k_mort ~ 1.2e-5 s^-1
    k_mort_max = 0.693 / (16.0 * 3600.0) # based on 16h half-life at LC50
    
    # Mortality factor (0 to k_mort_max)
    # Hill coefficient n=2 for steeper transition
    mortality_rate = k_mort_max * (oil_conc**2) / (oil_conc**2 + lc50**2 + 1e-12)
    
    # Legacy support: if sens_coeff is explicitly given, add it (linear model)
    if sens_coeff is not None:
        mortality_rate += sens_coeff * oil_conc

    dP_mortality = -mortality_rate * plankton * dt

    # --- Recovery Term ---
    # Logistic growth towards 1.0 (assuming 1.0 is carrying capacity/reference)
    # Only recovers if oil is low (< 1% of LC50)
    can_recover = (oil_conc < (0.01 * lc50))
    dP_recovery = np.zeros_like(plankton)
    
    # Simple logistic: r * P * (1 - P)
    # We assume plankton is normalized (0~1). If not, this needs reference P.
    # Here we assume input 'plankton' is relative to healthy state (approx 1.0).
    dP_recovery[can_recover] = recovery_rate * plankton[can_recover] * (1.0 - plankton[can_recover]) * dt

    plankton_new = plankton + dP_mortality + dP_recovery
    
    return np.clip(plankton_new, 0.0, None)


def ecological_recovery_index(
    DO,
    DO_ref,
    plankton,
    plankton_ref,
    benthos,
    benthos_ref,
    w_DO=0.4,
    w_plankton=0.3,
    w_benthos=0.3,
):
    """
    Calculate Ecological Recovery Index (0~1).

    Weighted average of normalized ecosystem health indicators.
    
    Weights (based on ecosystem service importance):
    - DO (0.4): Critical for all life.
    - Plankton (0.3): Base of food web.
    - Benthos (0.3): Long-term sediment health indicator.

    Parameters
    ----------
    DO, DO_ref : np.ndarray or float
    plankton, plankton_ref : np.ndarray or float
    benthos, benthos_ref : np.ndarray or float
    w_DO, w_plankton, w_benthos : float

    Returns
    -------
    idx : np.ndarray or float
        0.0 (Dead) -> 1.0 (Fully Recovered)
    """
    # Normalize (clip at 1.0 to prevent 'super-recovery' masking other deficits)
    # Note: Original code clipped at 2.0, but for an index 0-1, clipping at 1.0 (or slightly above) 
    # for components is usually safer to avoid offsetting. We'll keep 1.2 to allow slight bloom compensation.
    
    DO_rel = np.clip(np.asarray(DO) / (DO_ref + 1e-12), 0.0, 1.2)
    plank_rel = np.clip(np.asarray(plankton) / (plankton_ref + 1e-12), 0.0, 1.2)
    benth_rel = np.clip(np.asarray(benthos) / (benthos_ref + 1e-12), 0.0, 1.2)

    # Normalize weights to sum to 1
    total_w = w_DO + w_plankton + w_benthos
    w_DO /= total_w
    w_plankton /= total_w
    w_benthos /= total_w

    idx = w_DO * DO_rel + w_plankton * plank_rel + w_benthos * benth_rel
    
    return np.clip(idx, 0.0, 1.0)

