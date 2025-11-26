# biology_ops.py
"""
Biology mini-models for oil spill:
- DO 감소 및 회복
- 미생물/플랑크톤 반응
- 생태 회복 지수(Ecological Recovery Index)
모든 계수는 논문(실험/현장 연구)에서 얻은 값을 사용.
"""

import numpy as np


def update_DO(
    DO,
    DO_sat,
    oil_conc,
    k_consume,
    k_reaer,
    dt,
):
    """
    DO(t+dt) = DO + dt * [ reaeration - consumption ]

    reaeration ≈ k_reaer * (DO_sat - DO)
    consumption ≈ k_consume * oil_conc

    Parameters
    ----------
    DO : np.ndarray
        현재 용존산소 [mg/L]
    DO_sat : np.ndarray or float
        포화 DO (온도, 염분, 대기압 기반. 외부에서 계산해서 넣기)
    oil_conc : np.ndarray
        기름 또는 유기오염 농도 [mg/L]
    k_consume : float
        오염물 1단위당 DO 소비 속도 [ (mg/L)^{-1} * (mg/L)/s ]
        (실제로는 미생물 활성 포함된 유효계수, 논문 값 사용)
    k_reaer : float
        재포기 속도상수 [1/s]
    dt : float
        시간 스텝 [s]

    Returns
    -------
    DO_new : np.ndarray
    """
    DO = np.asarray(DO, dtype=float)
    DO_sat = np.asarray(DO_sat, dtype=float)
    oil_conc = np.asarray(oil_conc, dtype=float)

    reaer = k_reaer * (DO_sat - DO)
    consume = k_consume * oil_conc

    DO_new = DO + dt * (reaer - consume)
    # 비음수 및 최대 포화도 클리핑
    DO_new = np.clip(DO_new, 0.0, DO_sat)
    return DO_new


def plankton_response(plankton, oil_conc, sens_coeff, dt):
    """
    단순한 플랑크톤/미생물 biomss 반응 모델.

    dP/dt = - sens_coeff * oil_conc * P

    Parameters
    ----------
    plankton : np.ndarray
        현재 biomass (상대 단위 가능)
    oil_conc : np.ndarray
        기름 농도 [mg/L]
    sens_coeff : float
        감수성 계수 (논문에서 보고된 'x% 감소'를 환산해서 세팅)
    dt : float
        시간 스텝 [s]

    Returns
    -------
    plankton_new : np.ndarray
    """
    plankton = np.asarray(plankton, dtype=float)
    oil_conc = np.asarray(oil_conc, dtype=float)
    dP = -sens_coeff * oil_conc * plankton * dt
    return np.clip(plankton + dP, 0.0, None)


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
    0~1 사이의 생태 회복 지수 계산.

    각 항목은 (현재 / 기준) 으로 normalization 후 가중 평균.

    Parameters
    ----------
    DO, DO_ref : np.ndarray or float
        현재 및 참고 DO (사고 이전 평균 등)
    plankton, plankton_ref : np.ndarray or float
    benthos, benthos_ref : np.ndarray or float
    w_DO, w_plankton, w_benthos : float
        가중치 (논문/전문가 judgment로 설정)

    Returns
    -------
    idx : np.ndarray or float
        0~1 회복 지수 (1에 가까울수록 회복)
    """
    DO_rel = np.clip(np.asarray(DO) / (DO_ref + 1e-12), 0.0, 2.0)
    plank_rel = np.clip(np.asarray(plankton) / (plankton_ref + 1e-12), 0.0, 2.0)
    benth_rel = np.clip(np.asarray(benthos) / (benthos_ref + 1e-12), 0.0, 2.0)

    idx = w_DO * DO_rel + w_plankton * plank_rel + w_benthos * benth_rel
    # 0~1 로 스케일링 (2 이상이면 1로 클리핑)
    return np.clip(idx, 0.0, 1.0)

