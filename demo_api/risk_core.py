"""Bankruptcy-risk scoring used by the notebook (US_bankrupt.ipynb), as plain functions.

Feature definitions and coefficients are taken verbatim from the notebook cells:
Altman Z (cells 14, 24, 26), Zmijewski (cells 16, 28, 30), Ohlson (cells 18, 32, 34)
and the rule-based ``overall_risk`` combiner (cell 36).
"""
from __future__ import annotations

import math
from typing import Any

REQUIRED = {
    "Current_Assets": "Current assets",
    "Total_Current_Liabilities": "Total current liabilities",
    "Total_Assets": "Total assets",
    "Retained_Earnings": "Retained earnings",
    "EBIT": "EBIT",
    "Market_Value": "Market value of equity",
    "Net_Sales": "Net sales",
    "Net_Income": "Net income",
    "Total_Liabilities": "Total liabilities",
}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def parse_financials(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, label in REQUIRED.items():
        if key not in raw or raw[key] in (None, ""):
            raise ValueError(f"Missing value: {label}")
        try:
            val = float(raw[key])
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a number") from None
        if not math.isfinite(val):
            raise ValueError(f"{label} must be a finite number")
        out[key] = val
    for key in ("Total_Assets", "Total_Liabilities", "Current_Assets", "Total_Current_Liabilities"):
        if out[key] <= 0:
            raise ValueError(f"{REQUIRED[key]} must be greater than zero")
    return out


def features(f: dict[str, float]) -> dict[str, float]:
    ta = f["Total_Assets"]
    return {
        "WC_to_TA": (f["Current_Assets"] - f["Total_Current_Liabilities"]) / ta,
        "RE_to_TA": f["Retained_Earnings"] / ta,
        "EBIT_to_TA": f["EBIT"] / ta,
        "MVE_to_TL": f["Market_Value"] / f["Total_Liabilities"],
        "Sales_to_TA": f["Net_Sales"] / ta,
        "ROA": f["Net_Income"] / ta,
        "Leverage": f["Total_Liabilities"] / ta,
        "Current_Ratio_Calc": f["Current_Assets"] / f["Total_Current_Liabilities"],
        "ln_TA": math.log(ta),
        "CL_to_CA": f["Total_Current_Liabilities"] / f["Current_Assets"],
        "Neg_Net_Income": 1.0 if f["Net_Income"] < 0 else 0.0,
    }


def altman(x: dict[str, float]) -> tuple[float, str]:
    z = 1.2 * x["WC_to_TA"] + 1.4 * x["RE_to_TA"] + 3.3 * x["EBIT_to_TA"] + 0.6 * x["MVE_to_TL"] + 1.0 * x["Sales_to_TA"]
    zone = "Safe" if z > 2.99 else "Gray" if z > 1.81 else "Distress"
    return z, zone


def zmijewski(x: dict[str, float]) -> tuple[float, float, str]:
    score = -4.3 - 4.5 * x["ROA"] + 5.7 * x["Leverage"] - 0.004 * x["Current_Ratio_Calc"]
    prob = _norm_cdf(score)
    return score, prob, "High Risk" if prob > 0.5 else "Low Risk"


def ohlson(f: dict[str, float], x: dict[str, float]) -> tuple[float, float, str]:
    ta = f["Total_Assets"]
    score = (
        -1.32
        - 0.407 * x["ln_TA"]
        + 6.03 * (f["Total_Liabilities"] / ta)
        - 1.43 * x["WC_to_TA"]
        + 0.0757 * x["CL_to_CA"]
        - 2.37 * (f["Net_Income"] / ta)
        + 0.285 * x["Neg_Net_Income"]
    )
    prob = _sigmoid(score)
    return score, prob, "High Risk" if prob > 0.5 else "Low Risk"


def overall(altman_zone: str, zmijewski_risk: str, ohlson_risk: str) -> str:
    if altman_zone == "Distress" or zmijewski_risk == "High Risk" or ohlson_risk == "High Risk":
        return "Risk"
    if altman_zone == "Gray":
        return "Risk"
    return "Low Risk"


def score(raw: dict[str, Any]) -> dict[str, Any]:
    f = parse_financials(raw)
    x = features(f)
    z, zone = altman(x)
    zs, zp, zr = zmijewski(x)
    os_, op, orisk = ohlson(f, x)
    return {
        "inputs": f,
        "features": {k: round(v, 4) for k, v in x.items()},
        "altman": {"z": round(z, 3), "zone": zone},
        "zmijewski": {"score": round(zs, 3), "probability": round(zp, 4), "risk": zr},
        "ohlson": {"score": round(os_, 3), "probability": round(op, 4), "risk": orisk},
        "overall": overall(zone, zr, orisk),
    }
