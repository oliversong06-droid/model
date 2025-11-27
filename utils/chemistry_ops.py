import numpy as np

"""
Chemistry-related mini models for oil spill:
- UV/fluorescence -> concentration conversion
- TOC based contamination level
- Chemical decay (weathering, photolysis)
- Dispersant effectiveness
- Toxicity Threshold Checks

Scientific parameters derived from literature (see doc/bio_chem_params.md).
"""

# --- Scientific Constants ---
# Phytoplankton Inhibition Threshold: > 100 mg/L (severe)
PHYTO_INHIB_THRESHOLD = 100.0 

# General Chemical Decay (Weathering) Rate: ~0.01 - 0.05 day^-1 (slow without bio)
# Photolysis can be faster but depth-limited.
DAYS_TO_SECONDS = 86400.0
K_CHEM_DEFAULT = 0.02 / DAYS_TO_SECONDS


# --------- 1. UV / fluorescence -> Concentration Conversion --------- #

def uv_to_concentration_linear(intensity, a, b):
    """
    Linearly convert UV / fluorescence intensity to oil concentration.

    intensity : np.ndarray or float
        Measured UV / fluorescence intensity.
    a, b : float
        Calibration coefficients from experiments / papers.
        (e.g., y = a x + b)

    Returns
    -------
    conc : np.ndarray or float
        Estimated oil concentration [e.g. mg/L].
    """
    return a * intensity + b


def uv_to_concentration_poly(intensity, coeffs):
    """
    Polynomial calibration curve (for non-linear response).

    intensity : np.ndarray or float
    coeffs : list or np.ndarray
        Polynomial coefficients [c0, c1, c2, ...] so that
        conc = c0 + c1*I + c2*I^2 + ...

    Returns
    -------
    conc : np.ndarray or float
    """
    # np.polyval expects highest order first, so reverse
    return np.polyval(coeffs[::-1], intensity)


# --------- 2. TOC Based Contamination Level --------- #

def classify_toc(toc_value, threshold_moderate, threshold_high):
    """
    Classify contamination level based on TOC (Total Organic Carbon).

    Parameters
    ----------
    toc_value : float or np.ndarray
        TOC value [mg/L].
    threshold_moderate : float
        Lower threshold (e.g. 'significant contamination' start).
    threshold_high : float
        High contamination threshold.

    Returns
    -------
    level : int or np.ndarray
        0 = low / background
        1 = moderate contamination
        2 = high / severe contamination
    """
    toc = np.asarray(toc_value)
    level = np.zeros_like(toc, dtype=int)
    level[(toc >= threshold_moderate) & (toc < threshold_high)] = 1
    level[toc >= threshold_high] = 2
    return level


# --------- 3. Chemical Decay / Weathering --------- #

def apply_chemical_decay(conc, k_chem=None, dt=3600.0):
    """
    First-order chemical degradation (photo-oxidation, dissolution, etc.).

    dC/dt = -k_chem * C  ->  C(t+dt) = C(t) * exp(-k_chem * dt)

    Parameters
    ----------
    conc : np.ndarray
        Current oil concentration (water column or surface film).
    k_chem : float, optional
        First-order decay constant [1/s].
        If None, uses K_CHEM_DEFAULT.
    dt : float
        Time step [s].

    Returns
    -------
    conc_new : np.ndarray
        Updated concentration after chemical decay.
    """
    if k_chem is None:
        k_chem = K_CHEM_DEFAULT
        
    return conc * np.exp(-k_chem * dt)


def apply_multiphase_decay(conc_dissolved, conc_droplet, k_dissolved, k_droplet, dt):
    """
    Different decay rate for dissolved vs droplet phase.

    Parameters
    ----------
    conc_dissolved : np.ndarray
    conc_droplet   : np.ndarray
    k_dissolved    : float  (Dissolved phase decay rate)
    k_droplet      : float  (Droplet/Emulsion decay rate)
    dt             : float

    Returns
    -------
    conc_dissolved_new, conc_droplet_new
    """
    cd_new = conc_dissolved * np.exp(-k_dissolved * dt)
    cp_new = conc_droplet * np.exp(-k_droplet * dt)
    return cd_new, cp_new


# --------- 4. Dispersant / Chemical Treatment Effect --------- #

def apply_dispersant_effect(surface_conc, dispersant_dose, efficiency):
    """
    Simple model: Fraction of surface oil dispersed into water column.

    Parameters
    ----------
    surface_conc : np.ndarray
        Surface oil concentration (e.g. g/m^2).
    dispersant_dose : float
        Normalized dose (0~1 recommended scaling).
    efficiency : float
        Fraction of surface oil removed per unit dose (0~1).

    Returns
    -------
    surface_new : np.ndarray
        Reduced surface concentration.
    dispersed_amount : np.ndarray
        Amount moved to subsurface/droplet phase.
    """
    # Linear approximation clipped at 100% removal
    removal_frac = np.clip(dispersant_dose * efficiency, 0.0, 1.0)
    dispersed_amount = surface_conc * removal_frac
    surface_new = surface_conc - dispersed_amount
    return surface_new, dispersed_amount


# --------- 5. Toxicity Assessment (New) --------- #

def check_toxicity_thresholds(oil_conc, thresholds=None):
    """
    Check if oil concentration exceeds specific toxicity thresholds.
    
    Parameters
    ----------
    oil_conc : np.ndarray
        Oil concentration [mg/L]
    thresholds : dict, optional
        Dictionary of thresholds. Default:
        {'phyto_inhibit': 100.0, 'zoo_lc50': 30.0}
        
    Returns
    -------
    flags : dict of np.ndarray (bool)
        Keys matching thresholds, values are boolean masks where limit is exceeded.
    """
    if thresholds is None:
        thresholds = {
            'phyto_inhibit': PHYTO_INHIB_THRESHOLD,
            'zoo_lc50': 30.0  # Consistent with biology_ops default
        }
        
    flags = {}
    oil_conc = np.asarray(oil_conc)
    
    for key, val in thresholds.items():
        flags[key] = (oil_conc >= val)
        
    return flags
