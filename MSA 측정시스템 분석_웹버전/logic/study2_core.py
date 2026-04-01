import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import f

from logic.common import normalize_study2_df

EPS = 1e-12


def _safe_pct(part, whole):
    if abs(whole) <= EPS:
        return 0.0
    return 100.0 * part / whole


def _safe_pct_tol(part, tol):
    if tol <= EPS:
        return np.nan
    return 100.0 * part / tol


def _format_gage_row(source, sd_value, total_sv, tol):
    six_sd = 6 * sd_value
    return {
        "출처": source,
        "SD": sd_value,
        "6*SD": six_sd,
        "%SV": _safe_pct(six_sd, total_sv),
        "%공차": _safe_pct_tol(six_sd, tol),
    }


def _build_named_anova_table(raw_table, rename_index_map, ordered_index):
    table = raw_table.copy()
    table["mean_sq"] = table["sum_sq"] / table["df"]
    table = table.rename(index=rename_index_map).rename(
        columns={
            "df": "자유도(DF)",
            "sum_sq": "제곱합(SS)",
            "mean_sq": "평균제곱(MS)",
            "F": "F값",
            "PR(>F)": "P값",
        }
    )
    total_ss = table["제곱합(SS)"].sum()
    total_df = table["자유도(DF)"].sum()
    table.loc["총변동", "자유도(DF)"] = total_df
    table.loc["총변동", "제곱합(SS)"] = total_ss
    table.loc["총변동", ["평균제곱(MS)", "F값", "P값"]] = np.nan
    return table.reindex(ordered_index)[["자유도(DF)", "제곱합(SS)", "평균제곱(MS)", "F값", "P값"]]


def _build_full_interaction_anova(df):
    model_full = smf.ols("측정값 ~ C(부품)*C(측정자)", data=df).fit()
    anova_full_raw = sm.stats.anova_lm(model_full, typ=2)
    anova_full = _build_named_anova_table(
        anova_full_raw,
        {
            "C(부품)": "부품",
            "C(측정자)": "측정자",
            "C(부품):C(측정자)": "부품:측정자",
            "Residual": "반복성",
        },
        ["부품", "측정자", "부품:측정자", "반복성", "총변동"],
    )

    ms_interaction = anova_full.loc["부품:측정자", "평균제곱(MS)"]
    df_interaction = anova_full.loc["부품:측정자", "자유도(DF)"]
    ms_repeat = anova_full.loc["반복성", "평균제곱(MS)"]
    df_repeat = anova_full.loc["반복성", "자유도(DF)"]

    for factor_name in ["부품", "측정자"]:
        ms_factor = anova_full.loc[factor_name, "평균제곱(MS)"]
        df_factor = anova_full.loc[factor_name, "자유도(DF)"]
        if ms_interaction > EPS:
            f_value = ms_factor / ms_interaction
            p_value = 1 - f.cdf(f_value, df_factor, df_interaction)
        else:
            f_value = np.nan
            p_value = np.nan
        anova_full.loc[factor_name, "F값"] = f_value
        anova_full.loc[factor_name, "P값"] = p_value

    if ms_repeat > EPS:
        f_interaction = ms_interaction / ms_repeat
        p_interaction = 1 - f.cdf(f_interaction, df_interaction, df_repeat)
    else:
        f_interaction = np.nan
        p_interaction = np.nan

    anova_full.loc["부품:측정자", "F값"] = f_interaction
    anova_full.loc["부품:측정자", "P값"] = p_interaction
    anova_full.loc["반복성", ["F값", "P값"]] = np.nan
    return anova_full, p_interaction


def _build_reduced_anova(df):
    model_reduced = smf.ols("측정값 ~ C(부품)+C(측정자)", data=df).fit()
    anova_reduced_raw = sm.stats.anova_lm(model_reduced, typ=2)
    return _build_named_anova_table(
        anova_reduced_raw,
        {
            "C(부품)": "부품",
            "C(측정자)": "측정자",
            "Residual": "반복성",
        },
        ["부품", "측정자", "반복성", "총변동"],
    )


def _build_study3_anova(df):
    model = smf.ols("측정값 ~ C(부품)", data=df).fit()
    anova_raw = sm.stats.anova_lm(model, typ=2)
    return _build_named_anova_table(
        anova_raw,
        {
            "C(부품)": "부품",
            "Residual": "반복성",
        },
        ["부품", "반복성", "총변동"],
    )


def _build_judgement_table(ndc, q_mp, pct_sv_gagerr, pct_tol_gagerr):
    rows = [
        {
            "항목": "NDC",
            "값": ndc,
            "기준": ">= 5",
            "판정": "OK" if not np.isnan(ndc) and ndc >= 5 else "NG",
        },
        {
            "항목": "Q_MP",
            "값": q_mp,
            "기준": "<= 30%",
            "판정": "OK" if not np.isnan(q_mp) and q_mp <= 30 else "NG",
        },
        {
            "항목": "%SV",
            "값": pct_sv_gagerr,
            "기준": "<= 30%",
            "판정": "OK" if not np.isnan(pct_sv_gagerr) and pct_sv_gagerr <= 30 else "NG",
        },
        {
            "항목": "%공차",
            "값": pct_tol_gagerr,
            "기준": "<= 30%",
            "판정": "OK" if not np.isnan(pct_tol_gagerr) and pct_tol_gagerr <= 30 else "NG",
        },
    ]
    return pd.DataFrame(rows)


def run_study2_analysis(df, TOL, u_B_dict):
    """
    ISO 22514-7 Study 2/3 Gage R&R 분석 로직.

    참조 기준:
    - ISO 22514-7 Study 2_REV06.py
    - 교호작용 유의성 판단은 부품:측정자 항을 반복성으로 검정
    - Q_MP, %SV, %공차 판정 기준은 30% 이하
    """
    df = normalize_study2_df(df)
    df["부품"] = df["부품"].astype("category").cat.remove_unused_categories()
    df["측정자"] = df["측정자"].astype("category").cat.remove_unused_categories()

    counts = df.groupby(["부품", "측정자"], observed=True).size()
    if counts.empty:
        raise ValueError("Study-2/3 데이터 그룹화 결과가 비어 있습니다.")

    n_o = df["측정자"].nunique()
    n_p = df["부품"].nunique()
    n_r = counts.min()

    if n_p < 2:
        raise ValueError("Study-2/3 데이터는 서로 다른 부품이 2개 이상 필요합니다.")
    if n_r < 2:
        raise ValueError("Study-2/3 데이터는 각 부품×측정자 조합당 반복 측정이 2회 이상 필요합니다.")

    warnings = []
    if counts.min() != counts.max():
        warnings.append(f"불균형 데이터 감지: 반복수 min={int(counts.min())}, max={int(counts.max())}")

    u_CAL = float(u_B_dict.get("u_CAL", 0.0))
    u_BI = float(u_B_dict.get("u_BI", 0.0))
    u_EVR = float(u_B_dict.get("u_EVR", 0.0))
    u_RE = float(u_B_dict.get("u_RE", 0.0))
    u_LIN = float(u_B_dict.get("u_LIN", 0.0))
    u_MSREST = float(u_B_dict.get("u_MSREST", 0.0))
    u_T = float(u_B_dict.get("u_T", 0.0))
    u_STAB = float(u_B_dict.get("u_STAB", 0.0))
    u_OBJ = float(u_B_dict.get("u_OBJ", 0.0))
    u_GV = float(u_B_dict.get("u_GV", 0.0))
    u_REST = float(u_B_dict.get("u_REST", 0.0))

    if n_o == 1:
        anova_table = _build_study3_anova(df)
        ms_repeat = anova_table.loc["반복성", "평균제곱(MS)"]
        ms_part = anova_table.loc["부품", "평균제곱(MS)"]

        sd_repeat = np.sqrt(ms_repeat)
        sd_operator = 0.0
        sd_interaction = 0.0
        sd_reprod = 0.0
        sd_gagerr = sd_repeat
        sd_part = np.sqrt(max((ms_part - ms_repeat) / n_r, 0.0))
        interaction_significant = False

        u_EVO = sd_repeat
        u_AV = 0.0
        u_IA = 0.0
        u_EV = max(u_EVR, u_RE, u_EVO)
        u_mp_components = [u_CAL, u_BI, u_EV, u_AV, u_IA, u_LIN, u_MSREST, u_T, u_STAB, u_OBJ, u_GV, u_REST]
    else:
        anova_full, p_interaction = _build_full_interaction_anova(df)
        interaction_significant = not pd.isna(p_interaction) and float(p_interaction) <= 0.05

        if interaction_significant:
            anova_table = anova_full.copy()
        else:
            anova_table = _build_reduced_anova(df)

        ms_repeat = anova_table.loc["반복성", "평균제곱(MS)"]
        ms_part = anova_table.loc["부품", "평균제곱(MS)"]
        ms_operator = anova_table.loc["측정자", "평균제곱(MS)"] if "측정자" in anova_table.index else 0.0

        if interaction_significant and "부품:측정자" in anova_table.index:
            ms_interaction = anova_table.loc["부품:측정자", "평균제곱(MS)"]
            var_repeat = ms_repeat
            var_operator = max((ms_operator - ms_interaction) / (n_p * n_r), 0.0)
            var_interaction = max((ms_interaction - ms_repeat) / n_r, 0.0)
            var_part = max((ms_part - ms_interaction) / (n_o * n_r), 0.0)
            u_IA = np.sqrt(max(ms_interaction - ms_repeat, 0.0) / (n_p * n_r))
        else:
            ms_interaction = 0.0
            var_repeat = ms_repeat
            var_operator = max((ms_operator - ms_repeat) / (n_p * n_r), 0.0)
            var_interaction = 0.0
            var_part = max((ms_part - ms_repeat) / (n_o * n_r), 0.0)
            u_IA = 0.0

        sd_repeat = np.sqrt(var_repeat)
        sd_operator = np.sqrt(var_operator)
        sd_interaction = np.sqrt(var_interaction)
        sd_reprod = np.sqrt(sd_operator**2 + sd_interaction**2)
        sd_gagerr = np.sqrt(sd_repeat**2 + sd_reprod**2)
        sd_part = np.sqrt(var_part)

        u_EVO = np.sqrt(ms_repeat)
        u_AV = np.sqrt(max(ms_operator - ms_repeat, 0.0) / (n_p * n_r))
        u_EV = max(u_EVR, u_RE, u_EVO)
        u_mp_components = [u_CAL, u_BI, u_EVO, u_AV, u_IA, u_LIN, u_MSREST, u_T, u_STAB, u_OBJ, u_GV, u_REST]

    sd_total = np.sqrt(sd_gagerr**2 + sd_part**2)
    six_sd_total = 6 * sd_total

    gage_rows = [
        _format_gage_row("총 Gage R&R", sd_gagerr, six_sd_total, TOL),
        _format_gage_row("  반복성", sd_repeat, six_sd_total, TOL),
    ]

    if n_o > 1:
        gage_rows.append(_format_gage_row("  재현성", sd_reprod, six_sd_total, TOL))
        if interaction_significant:
            gage_rows.append(_format_gage_row("    측정자", sd_operator, six_sd_total, TOL))
            gage_rows.append(_format_gage_row("    교호작용", sd_interaction, six_sd_total, TOL))

    gage_rows.extend(
        [
            _format_gage_row("부품-대-부품", sd_part, six_sd_total, TOL),
            _format_gage_row("총 변동", sd_total, six_sd_total, TOL),
        ]
    )
    df_gage = pd.DataFrame(gage_rows, columns=["출처", "SD", "6*SD", "%SV", "%공차"])
    if abs(six_sd_total) > EPS:
        df_gage.loc[df_gage["출처"] == "총 변동", "%SV"] = 100.0

    u_MP = np.sqrt(sum(v**2 for v in u_mp_components if abs(v) > EPS))
    U_MP = 2 * u_MP
    Q_MP = _safe_pct_tol(2 * U_MP, TOL)
    C_MP = (0.4 * TOL) / (2 * U_MP) if U_MP > EPS else np.nan
    ndc = (1.41 * sd_part / sd_gagerr) if sd_gagerr > EPS else np.nan

    pct_sv_gagerr = float(df_gage.loc[df_gage["출처"] == "총 Gage R&R", "%SV"].iloc[0])
    pct_tol_gagerr = float(df_gage.loc[df_gage["출처"] == "총 Gage R&R", "%공차"].iloc[0])
    judgement = _build_judgement_table(ndc, Q_MP, pct_sv_gagerr, pct_tol_gagerr)

    u_elements = {
        "u_CAL": u_CAL,
        "u_BI": u_BI,
        "u_EVR": u_EVR,
        "u_RE": u_RE,
        "u_EVO": u_EVO,
        "u_AV": u_AV,
        "u_IA": u_IA,
        "u_EV": u_EV,
        "u_LIN": u_LIN,
        "u_MSREST": u_MSREST,
        "u_T": u_T,
        "u_STAB": u_STAB,
        "u_OBJ": u_OBJ,
        "u_GV": u_GV,
        "u_REST": u_REST,
    }

    return {
        "anova": anova_table,
        "gage": df_gage,
        "ndc": ndc,
        "u_mp": u_MP,
        "u_mp_expanded": U_MP,
        "q_mp": Q_MP,
        "c_mp": C_MP,
        "u_elements": u_elements,
        "judgement": judgement,
        "overall_pass": bool((judgement["판정"] == "OK").all()),
        "interaction_significant": interaction_significant,
        "warnings": warnings,
    }
