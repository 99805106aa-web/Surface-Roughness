import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_dist

from logic.study2_core import run_study2_analysis

EPS = 1e-12


def _build_regression_tables(df: pd.DataFrame):
    x = df["Reference"].astype(float)
    y = df["Value"].astype(float)
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    grouped = df.groupby("Reference", observed=True)["Value"]
    n_total = int(len(df))
    n_levels = int(grouped.ngroups)

    pure_error_ss = float(grouped.apply(lambda g: ((g - g.mean()) ** 2).sum()).sum())
    pure_error_df = max(n_total - n_levels, 0)
    pure_error_ms = pure_error_ss / pure_error_df if pure_error_df > 0 else np.nan

    error_ss = float(model.ssr)
    error_df = int(round(model.df_resid))
    error_ms = error_ss / error_df if error_df > 0 else np.nan

    lack_of_fit_ss = max(error_ss - pure_error_ss, 0.0)
    lack_of_fit_df = max(n_levels - 2, 0)
    lack_of_fit_ms = lack_of_fit_ss / lack_of_fit_df if lack_of_fit_df > 0 else np.nan

    if lack_of_fit_df > 0 and pure_error_df > 0 and pure_error_ms > EPS:
        lack_of_fit_f = lack_of_fit_ms / pure_error_ms
        lack_of_fit_p = float(f_dist.sf(lack_of_fit_f, lack_of_fit_df, pure_error_df))
    else:
        lack_of_fit_f = np.nan
        lack_of_fit_p = np.nan

    regression_df = pd.DataFrame(
        [
            {
                "항목": "Intercept",
                "Estimate": float(model.params["const"]),
                "S.E.": float(model.bse["const"]),
                "T": float(model.tvalues["const"]),
                "p-value": float(model.pvalues["const"]),
            },
            {
                "항목": "Slope",
                "Estimate": float(model.params["Reference"]),
                "S.E.": float(model.bse["Reference"]),
                "T": float(model.tvalues["Reference"]),
                "p-value": float(model.pvalues["Reference"]),
            },
        ]
    )

    total_ss = float(((y - y.mean()) ** 2).sum())
    total_df = max(n_total - 1, 0)
    regression_ss = float(model.ess)
    regression_df_value = 1
    regression_ms = regression_ss / regression_df_value if regression_df_value > 0 else np.nan

    anova_df = pd.DataFrame(
        [
            {
                "Source": "Regression",
                "DF": regression_df_value,
                "SS": regression_ss,
                "MS": regression_ms,
                "F": float(model.fvalue) if model.fvalue is not None else np.nan,
                "p-value": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
            },
            {
                "Source": "Error",
                "DF": error_df,
                "SS": error_ss,
                "MS": error_ms,
                "F": np.nan,
                "p-value": np.nan,
            },
            {
                "Source": "Lack-of-Fit",
                "DF": lack_of_fit_df,
                "SS": lack_of_fit_ss,
                "MS": lack_of_fit_ms,
                "F": lack_of_fit_f,
                "p-value": lack_of_fit_p,
            },
            {
                "Source": "Pure Error",
                "DF": pure_error_df,
                "SS": pure_error_ss,
                "MS": pure_error_ms,
                "F": np.nan,
                "p-value": np.nan,
            },
            {
                "Source": "Total",
                "DF": total_df,
                "SS": total_ss,
                "MS": np.nan,
                "F": np.nan,
                "p-value": np.nan,
            },
        ]
    )

    stats = {
        "intercept": float(model.params["const"]),
        "slope_value": float(model.params["Reference"]),
        "slope_bias": float(model.params["Reference"] - 1.0),
        "intercept_p": float(model.pvalues["const"]),
        "slope_p": float(model.pvalues["Reference"]),
        "s_value": float(np.sqrt(error_ms)) if error_ms > EPS else 0.0,
        "r_squared": float(model.rsquared),
        "pure_error_ms": float(pure_error_ms) if not pd.isna(pure_error_ms) else np.nan,
        "lack_of_fit_ms": float(lack_of_fit_ms) if not pd.isna(lack_of_fit_ms) else np.nan,
        "lack_of_fit_p": lack_of_fit_p,
    }
    return model, regression_df, anova_df, stats


def run_study4_analysis(df, USL, LSL, u_CAL_val, resolution, grr_df=None):
    df = df.copy()
    df["Reference"] = pd.to_numeric(df["Reference"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Reference", "Value"]).reset_index(drop=True)
    df["Bias"] = df["Value"] - df["Reference"]

    tol = float(USL - LSL)
    _, regression_table, anova_table, stats = _build_regression_tables(df)

    u_bi = 0.0
    u_evr = float(np.sqrt(stats["pure_error_ms"])) if stats["pure_error_ms"] > EPS else 0.0
    u_lin = float(np.sqrt(stats["lack_of_fit_ms"])) if stats["lack_of_fit_ms"] > EPS else 0.0
    u_re = float(resolution / np.sqrt(12))
    u_cal = float(u_CAL_val)
    u_rest = 0.0

    u_ms = float(np.sqrt(u_bi ** 2 + u_evr ** 2 + u_lin ** 2 + u_re ** 2 + u_cal ** 2 + u_rest ** 2))
    u_ms_expanded = float(2 * u_ms)
    q_ms = float((2 * u_ms_expanded / tol) * 100) if tol > EPS else np.nan
    c_ms = float((0.3 * tol) / (6 * u_ms)) if u_ms > EPS else np.nan

    summary = pd.DataFrame(
        [
            {"항목": "Intercept", "값": stats["intercept"], "기준": "p > 0.05", "판정": "NG" if stats["intercept_p"] <= 0.05 else "OK"},
            {"항목": "Slope", "값": stats["slope_value"], "기준": "1에 근접", "판정": "OK" if abs(stats["slope_value"] - 1.0) <= 0.05 else "NG"},
            {"항목": "S", "값": stats["s_value"], "기준": "-", "판정": "-"},
            {"항목": "R-squared", "값": stats["r_squared"], "기준": "-", "판정": "-"},
            {"항목": "Q_MS", "값": q_ms, "기준": "<= 15%", "판정": "OK" if not pd.isna(q_ms) and q_ms <= 15 else "NG"},
            {"항목": "C_MS", "값": c_ms, "기준": ">= 1.33", "판정": "OK" if not pd.isna(c_ms) and c_ms >= 1.33 else "NG"},
        ]
    )

    uncertainty_components = pd.DataFrame(
        [
            {"구분": "(1) Uncertainty arising from bias (Type A)", "코드": "u_BI", "값": u_bi},
            {"구분": "(2) Repeatability on reference standard (Type A)", "코드": "u_EVR", "값": u_evr},
            {"구분": "(3) Uncertainty arising from linearity (Type A)", "코드": "u_LIN", "값": u_lin},
            {"구분": "(4) Resolution of the measuring system (Type B)", "코드": "u_RE", "값": u_re},
            {"구분": "(5) Calibration uncertainty (Type B)", "코드": "u_CAL", "값": u_cal},
            {"구분": "(6) Other uncertainty components (Type B)", "코드": "u_REST", "값": u_rest},
            {"구분": "* Combined standard uncertainty", "코드": "u_MS", "값": u_ms},
            {"구분": "* Expanded measurement uncertainty", "코드": "U_MS", "값": u_ms_expanded},
        ]
    )

    resolution_summary = pd.DataFrame(
        [
            {"항목": "USL", "값": float(USL)},
            {"항목": "LSL", "값": float(LSL)},
            {"항목": "Tol", "값": tol},
            {"항목": "RE", "값": float(resolution)},
            {"항목": "%RE", "값": float((resolution / tol) * 100) if tol > EPS else np.nan},
        ]
    )

    linearity_capability = pd.DataFrame(
        [
            {"항목": "Q_MS", "값": q_ms, "기준": "<= 15%", "판정": "OK" if not pd.isna(q_ms) and q_ms <= 15 else "NG"},
            {"항목": "C_MS", "값": c_ms, "기준": ">= 1.33", "판정": "OK" if not pd.isna(c_ms) and c_ms >= 1.33 else "NG"},
        ]
    )

    output = {
        "summary": summary,
        "regression_table": regression_table,
        "anova_table": anova_table,
        "uncertainty_components": uncertainty_components,
        "resolution_summary": resolution_summary,
        "linearity_capability": linearity_capability,
        "u_ms": u_ms,
        "u_ms_expanded": u_ms_expanded,
        "q_ms": q_ms,
        "c_ms": c_ms,
        "r_squared": stats["r_squared"],
        "raw_with_bias": df,
        "linearity_components": {
            "u_BI": u_bi,
            "u_EVR": u_evr,
            "u_LIN": u_lin,
            "u_RE": u_re,
            "u_CAL": u_cal,
            "u_REST": u_rest,
        },
        "linearity_pass": bool(linearity_capability["판정"].eq("OK").all()),
    }

    if grr_df is None or grr_df.empty:
        return output

    grr_result = run_study2_analysis(
        grr_df,
        tol,
        {
            "u_CAL": u_cal,
            "u_BI": u_bi,
            "u_EVR": u_evr,
            "u_RE": u_re,
            "u_LIN": u_lin,
            "u_MSREST": u_rest,
            "u_T": 0.0,
            "u_STAB": 0.0,
            "u_OBJ": 0.0,
            "u_GV": 0.0,
            "u_REST": 0.0,
        },
    )

    capability_summary = pd.DataFrame(
        [
            {
                "항목": "Q_LMP",
                "값": float(grr_result["q_mp"]),
                "기준": "<= 30%",
                "판정": "OK" if not pd.isna(grr_result["q_mp"]) and float(grr_result["q_mp"]) <= 30 else "NG",
            },
            {
                "항목": "C_MP",
                "값": float(grr_result["c_mp"]),
                "기준": ">= 1.33",
                "판정": "OK" if not pd.isna(grr_result["c_mp"]) and float(grr_result["c_mp"]) >= 1.33 else "NG",
            },
        ]
    )

    component_order = [
        "u_BI", "u_EVR", "u_LIN", "u_RE", "u_CAL", "u_REST",
        "u_EVO", "u_AV", "u_IA", "u_T", "u_STAB", "u_OBJ", "u_GV",
    ]
    process_components = []
    for key in component_order:
        if key in grr_result["u_elements"]:
            process_components.append({"항목": key, "값": float(grr_result["u_elements"][key])})
    process_components.extend(
        [
            {"항목": "u_MP", "값": float(grr_result["u_mp"])},
            {"항목": "U_MP", "값": float(grr_result["u_mp_expanded"])},
        ]
    )

    output.update(
        {
            "gage_result": grr_result,
            "capability_summary": capability_summary,
            "process_components": pd.DataFrame(process_components),
            "q_lmp": float(grr_result["q_mp"]),
            "c_mp": float(grr_result["c_mp"]),
            "u_mp": float(grr_result["u_mp"]),
            "u_mp_expanded": float(grr_result["u_mp_expanded"]),
            "process_pass": bool(capability_summary["판정"].eq("OK").all()),
        }
    )
    return output
