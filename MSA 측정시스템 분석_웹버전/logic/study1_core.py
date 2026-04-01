import numpy as np
import pandas as pd
import scipy.stats as stats
from .common import calc_u_distribution

def run_study1_analysis(RE_val, RefV_val, USL_val, LSL_val, u_CAL_val, u_LIN, u_MS_REST, 
                         X_array, bias_dist="균등분포", ure_dist="균등분포"):
    """
    ISO 22514-7 Study 1 핵심 분석 로직
    """
    N = len(X_array)
    if N < 2:
        raise ValueError("측정 데이터는 최소 2개 이상이어야 합니다.")
    
    TOL = round(USL_val - LSL_val, 6)
    if TOL <= 0:
        raise ValueError(f"공차(TOL)가 {TOL}입니다. USL이 LSL보다 커야 합니다.")

    # 통계 계산
    mean_x = np.mean(X_array)
    std_x = np.std(X_array, ddof=1)

    # 불확도 계산 (ISO 22514-7 기준)
    PntRE = RE_val / TOL * 100
    Abias = abs(mean_x - RefV_val) * 2
    
    # Study 2_Rev06에서 사용된 방식 적용 (분포 선택 반영)
    u_BI = calc_u_distribution(Abias, bias_dist)
    u_EVR = std_x
    u_RE = calc_u_distribution(RE_val, ure_dist)
    u_EV = max(u_EVR, u_RE)
    
    # 합성 표준 불확도
    u_MS = np.sqrt(u_CAL_val**2 + u_LIN**2 + u_BI**2 + u_EV**2 + u_MS_REST**2)
    U_MS = 2 * u_MS
    Q_MS = 2 * U_MS / TOL * 100

    # 지표 계산
    C_MS = 0.3 * TOL / (6 * u_MS) if u_MS > 0 else np.nan
    Cg = (0.2 * TOL) / (6 * std_x) if std_x > 0 else np.inf
    
    if std_x > 0:
        Cgk = (0.1 * TOL - abs(mean_x - RefV_val)) / (3 * std_x)
    else:
        Cgk = np.inf if abs(mean_x - RefV_val) == 0 else np.nan

    # t-test (치우침 검정)
    t_stat, p_val = stats.ttest_1samp(X_array, RefV_val) if std_x > 0 else (np.nan, np.nan)
    se = std_x / np.sqrt(N) if std_x > 0 else np.nan
    if std_x > 0 and N > 1:
        tcrit = stats.t.ppf(0.975, N - 1)
        diff = mean_x - RefV_val
        ci_low, ci_high = diff - tcrit * se, diff + tcrit * se
    else:
        ci_low, ci_high = np.nan, np.nan
        
    percent_EV = (6 * std_x) / TOL * 100 if TOL > 0 else np.nan

    # 결과 요약 DataFrame 생성
    results_dict = {
        "항목": [
            "참조값 (RefV)", "표준편차", "평균", "USL", "LSL", "TOL",
            "Cg (반복성 1.33 이상)", "Cgk (1.33 이상)",
            "C_MS (1.33 이상)", "Q_MS (15% 이내)", "%EV (30% 이내)", "RE (%) (5% 이내)",
            "u_CAL", "u_LIN", "u_MS_REST", "Abias", "u_BI", "u_EVR", "u_RE", "u_EV",
            "u_MS", "U_MS", "치우침", "표준오차", "t 통계량", "p-value", "95% CI 하한", "95% CI 상한"
        ],
        "값": [
            RefV_val, std_x, mean_x, USL_val, LSL_val, TOL,
            Cg, Cgk, C_MS, Q_MS, percent_EV, PntRE,
            u_CAL_val, u_LIN, u_MS_REST, Abias, u_BI, u_EVR, u_RE, u_EV,
            u_MS, U_MS, mean_x - RefV_val, se if not np.isnan(se) else 0,
            t_stat, p_val, ci_low, ci_high
        ]
    }
    
    df_summary = pd.DataFrame(results_dict)
    
    # 판정 로직 및 기준값 설정
    def get_criterion_and_judgment(row):
        name = row['항목']
        val = row['값']
        if "Cg (반복성 1.33 이상)" in name:
            return "≥ 1.33", ("OK" if val >= 1.33 else "NG")
        if "Cgk (1.33 이상)" in name:
            return "≥ 1.33", ("OK" if val >= 1.33 else "NG")
        if "C_MS (1.33 이상)" in name:
            return "≥ 1.33", ("OK" if val >= 1.33 else "NG")
        if "Q_MS (15% 이내)" in name:
            return "≤ 15.00", ("OK" if val <= 15.0 else "NG")
        if "%EV (30% 이내)" in name:
            return "≤ 30.00%", ("OK" if val <= 30.0 else "NG")
        if "RE (%) (5% 이내)" in name:
            return "≤ 5.00%", ("OK" if val <= 5.0 else "NG")
        return "-", "-"

    # apply 함수를 사용하여 '기준'과 '판정' 컬럼 생성
    res = df_summary.apply(lambda r: pd.Series(get_criterion_and_judgment(r)), axis=1)
    df_summary['기준'] = res[0]
    df_summary['판정'] = res[1]
    
    # 컬럼 순서 재조정 (이미지 UI: 항목 -> 값 -> 기준 -> 판정)
    df_summary = df_summary[['항목', '값', '기준', '판정']]
    
    return df_summary, X_array, (RefV_val, TOL)
