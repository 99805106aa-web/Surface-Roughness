import numpy as np
import pandas as pd


def _rate_effectiveness(value: float) -> str:
    if value >= 0.9:
        return "적합"
    if value >= 0.8:
        return "조건부 채택"
    return "부적합"


def _rate_false_alarm(value: float) -> str:
    if value < 0.02:
        return "적합"
    if value < 0.05:
        return "조건부 채택"
    return "부적합"


def _rate_miss(value: float) -> str:
    if value < 0.05:
        return "적합"
    if value < 0.1:
        return "조건부 채택"
    return "부적합"


def _rate_kappa_aiag(value: float) -> str:
    if value >= 0.75:
        return "우수"
    if value > 0.4:
        return "보통"
    return "미흡"


def _rate_kappa_minitab(value: float) -> str:
    if value >= 0.9:
        return "우수"
    if value > 0.7:
        return "보통"
    return "미흡"


def _overall_acceptance(grades) -> str:
    if all(grade == "적합" for grade in grades):
        return "적합"
    if any(grade == "부적합" for grade in grades):
        return "부적합"
    return "조건부 채택"


def calc_fleiss_kappa(df: pd.DataFrame) -> dict:
    appraisers = [col for col in df.columns if col not in ["Sample", "Standard"]]
    n_raters = len(appraisers)
    n_subjects = len(df)

    if n_raters < 2:
        return {
            "fleiss_kappa": np.nan,
            "aiag_grade": "?대떦 ?놁쓬 (?됯???2紐??댁긽 ?꾩슂)",
            "minitab_grade": "?대떦 ?놁쓬",
            "n_raters": n_raters,
            "n_subjects": n_subjects,
        }

    ratings = df[appraisers].astype(float).values
    observed_agreement = np.full(n_subjects, np.nan)
    total_zero = 0.0
    total_one = 0.0
    total_valid = 0

    for index in range(n_subjects):
        row = ratings[index]
        valid = row[~np.isnan(row)]
        n_valid = len(valid)
        if n_valid < 2:
            continue
        zeros = float((valid == 0).sum())
        ones = float((valid == 1).sum())
        total_zero += zeros
        total_one += ones
        total_valid += n_valid
        observed_agreement[index] = (zeros * (zeros - 1) + ones * (ones - 1)) / (n_valid * (n_valid - 1))

    if total_valid == 0:
        return {
            "fleiss_kappa": np.nan,
            "aiag_grade": "怨꾩궛 遺덇?",
            "minitab_grade": "怨꾩궛 遺덇?",
            "n_raters": n_raters,
            "n_subjects": n_subjects,
        }

    p_bar = float(np.nanmean(observed_agreement))
    p_zero = total_zero / total_valid
    p_one = total_one / total_valid
    p_e_bar = p_zero ** 2 + p_one ** 2
    kappa = (p_bar - p_e_bar) / (1 - p_e_bar) if abs(1 - p_e_bar) > 1e-12 else 1.0

    return {
        "fleiss_kappa": float(kappa),
        "aiag_grade": _rate_kappa_aiag(kappa),
        "minitab_grade": _rate_kappa_minitab(kappa),
        "n_raters": n_raters,
        "n_subjects": n_subjects,
        "p_bar": p_bar,
        "p_e_bar": p_e_bar,
        "p0": p_zero,
        "p1": p_one,
    }


def run_study6_analysis(df):
    appraisers = [col for col in df.columns if col not in ["Sample", "Standard"]]
    standard = df["Standard"].astype(float)

    ng_total = int((standard == 1).sum())
    ok_total = int((standard == 0).sum())
    total = int(len(df))

    results = []
    for appraiser in appraisers:
        current = df[appraiser].astype(float)

        correct = int((current == standard).sum())
        effectiveness = correct / total if total else np.nan

        false_alarm_count = int(((standard == 0) & (current == 1)).sum())
        miss_count = int(((standard == 1) & (current == 0)).sum())

        p_fa = false_alarm_count / ok_total if ok_total else 0.0
        p_miss = miss_count / ng_total if ng_total else 0.0

        p_o = effectiveness
        p_e = ((current == 1).mean() * (standard == 1).mean()) + ((current == 0).mean() * (standard == 0).mean())
        kappa = (p_o - p_e) / (1 - p_e) if abs(1 - p_e) > 1e-12 else 1.0

        grade_e = _rate_effectiveness(effectiveness)
        grade_fa = _rate_false_alarm(p_fa)
        grade_miss = _rate_miss(p_miss)
        aiag_grade = _rate_kappa_aiag(kappa)
        minitab_grade = _rate_kappa_minitab(kappa)

        results.append(
            {
                "평가자": appraiser,
                "유효성(E)": effectiveness,
                "유효성 계산": f"{correct} / {total} = {effectiveness:.3f}",
                "유효성 판정": grade_e,
                "P(FA)": p_fa,
                "P(FA) 계산": f"{false_alarm_count} / {ok_total} = {p_fa:.3f}" if ok_total else "0 / 0 = 0.000",
                "P(FA) 판정": grade_fa,
                "P(miss)": p_miss,
                "P(miss) 계산": f"{miss_count} / {ng_total} = {p_miss:.3f}" if ng_total else "0 / 0 = 0.000",
                "P(miss) 판정": grade_miss,
                "Kappa": kappa,
                "AIAG 판정": aiag_grade,
                "미니탭 판정": minitab_grade,
                "종합 판정": _overall_acceptance([grade_e, grade_fa, grade_miss]),
            }
        )

    return pd.DataFrame(results)
