import numpy as np

"""
Chemistry-related mini models for oil spill:
- UV/fluorescence → concentration 변환
- TOC 기반 오염 등급 판단
- 화학적 분해(산화/가수분해 등) 속도식
- 분산제(또는 화학 처리제) 효과 반영
"""


# --------- 1. UV / fluorescence → 농도 변환 --------- #

def uv_to_concentration_linear(intensity, a, b):
    """
    Linearly convert UV / fluorescence intensity to oil concentration.

    intensity : np.ndarray or float
        Measured UV / fluorescence intensity.
    a, b : float
        Calibration coefficients from experiments / papers.
        (예: 논문에서 y = a x + b 형태로 제시된 값)

    Returns
    -------
    conc : np.ndarray or float
        Estimated oil concentration [e.g. mg/L].
    """
    return a * intensity + b


def uv_to_concentration_poly(intensity, coeffs):
    """
    Polynomial calibration curve (비선형 보정이 필요할 때).

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


# --------- 2. TOC 기반 오염 등급 --------- #

def classify_toc(toc_value, threshold_moderate, threshold_high):
    """
    Classify contamination level based on TOC (Total Organic Carbon).

    Parameters
    ----------
    toc_value : float or np.ndarray
        TOC value [mg/L].
    threshold_moderate : float
        Lower threshold (e.g. 'significant contamination' 시작점).
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


# --------- 3. 화학적 분해 / 소실 속도 --------- #

def apply_chemical_decay(conc, k_chem, dt):
    """
    First-order chemical degradation (photo-oxidation, dissolution 등).

    dC/dt = -k_chem * C  →  C(t+dt) = C(t) * exp(-k_chem * dt)

    Parameters
    ----------
    conc : np.ndarray
        Current oil concentration (water column 또는 surface film).
    k_chem : float
        First-order decay constant [1/time].
        (논문에서 half-life, rate constant 등으로 추정)
    dt : float
        Time step.

    Returns
    -------
    conc_new : np.ndarray
        Updated concentration after chemical decay.
    """
    return conc * np.exp(-k_chem * dt)


def apply_multiphase_decay(conc_dissolved, conc_droplet, k_dissolved, k_droplet, dt):
    """
    Different decay rate for dissolved vs droplet phase.

    Parameters
    ----------
    conc_dissolved : np.ndarray
    conc_droplet   : np.ndarray
    k_dissolved    : float  (용존 상태 분해율)
    k_droplet      : float  (방울/유제 상태 분해율)
    dt             : float

    Returns
    -------
    conc_dissolved_new, conc_droplet_new
    """
    cd_new = conc_dissolved * np.exp(-k_dissolved * dt)
    cp_new = conc_droplet * np.exp(-k_droplet * dt)
    return cd_new, cp_new


# --------- 4. 분산제 / 화학 처리 효과 --------- #

def apply_dispersant_effect(surface_conc, dispersant_dose, efficiency):
    """
    Simple model: 일부 표면유가 분산되어 물속으로 이동.

    Parameters
    ----------
    surface_conc : np.ndarray
        Surface oil concentration (e.g. g/m^2).
    dispersant_dose : float
        Normalized dose (0~1 범위로 scaling해서 사용 권장).
    efficiency : float
        Fraction of surface oil removed per unit dose (0~1).

    Returns
    -------
    surface_new : np.ndarray
        Reduced surface concentration.
    dispersed_amount : np.ndarray
        Amount moved to subsurface/droplet phase (필요시 biology 모델과 연결).
    """
    # 실제로는 non-linear response일 수 있지만 1차 근사로 둠
    removal_frac = np.clip(dispersant_dose * efficiency, 0.0, 1.0)
    dispersed_amount = surface_conc * removal_frac
    surface_new = surface_conc - dispersed_amount
    return surface_new, dispersed_amount
