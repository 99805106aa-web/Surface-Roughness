import numpy as np
import pandas as pd

def run_study5_analysis(df):
    """
    ISO 22514-7 & AIAG Study-5 (안정성) 분석
    df: 각 행이 하나의 부분군(Subgroup)인 데이터프레임
    """
    # 데이터 처리 (숫자형만 추출)
    data = df.select_dtypes(include=[np.number])
    subgroup_size = data.shape[1]
    
    # 관리도 계수 (Subgroup size 기준)
    factors = {
        2: {"A2": 1.880, "D3": 0, "D4": 3.267},
        3: {"A2": 1.023, "D3": 0, "D4": 2.574},
        4: {"A2": 0.729, "D3": 0, "D4": 2.282},
        5: {"A2": 0.577, "D3": 0, "D4": 2.114}
    }
    
    f = factors.get(subgroup_size, {"A2": 3/np.sqrt(subgroup_size), "D3": 0, "D4": 2.0}) # 기본값
    
    # X-bar, R 계산
    x_bars = data.mean(axis=1)
    ranges = data.max(axis=1) - data.min(axis=1)
    
    grand_mean = x_bars.mean()
    mean_range = ranges.mean()
    
    # 관리 한계선
    ucl_x = grand_mean + f["A2"] * mean_range
    lcl_x = grand_mean - f["A2"] * mean_range
    ucl_r = f["D4"] * mean_range
    lcl_r = f["D3"] * mean_range
    
    # 이상점 탐지
    out_x = (x_bars > ucl_x) | (x_bars < lcl_x)
    out_r = (ranges > ucl_r) | (ranges < lcl_r)
    
    summary = pd.DataFrame({
        "항목": ["전체 평균 (X-double bar)", "평균 범위 (R-bar)", "X-bar UCL", "X-bar LCL", "R UCL", "R LCL"],
        "값": [grand_mean, mean_range, ucl_x, lcl_x, ucl_r, lcl_r]
    })
    
    return {
        "summary": summary,
        "x_bars": x_bars,
        "ranges": ranges,
        "out_x": out_x,
        "out_r": out_r,
        "is_stable": not (out_x.any() or out_r.any())
    }
