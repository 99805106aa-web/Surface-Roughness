import pandas as pd
import numpy as np
import os
import re
import csv
from scipy.stats import norm
import io

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
STUDY2_REQUIRED_COLUMNS = ["부품", "측정자", "측정값"]
_STUDY2_COLUMN_ALIASES = {
    "부품": "부품",
    "부품명": "부품",
    "part": "부품",
    "partname": "부품",
    "partno": "부품",
    "partnumber": "부품",
    "부품part": "부품",
    "측정자": "측정자",
    "작업자": "측정자",
    "검사자": "측정자",
    "평가자": "측정자",
    "operator": "측정자",
    "appraiser": "측정자",
    "inspector": "측정자",
    "measurer": "측정자",
    "측정자operator": "측정자",
    "측정값": "측정값",
    "값": "측정값",
    "value": "측정값",
    "measurement": "측정값",
    "measuredvalue": "측정값",
    "result": "측정값",
    "reading": "측정값",
    "측정값value": "측정값",
}
STUDY4_REQUIRED_COLUMNS = ["Reference", "Value"]
_STUDY4_COLUMN_ALIASES = {
    "reference": "Reference",
    "ref": "Reference",
    "refvalue": "Reference",
    "nominal": "Reference",
    "master": "Reference",
    "standard": "Reference",
    "기준값": "Reference",
    "참조값": "Reference",
    "표준값": "Reference",
    "측정값": "Value",
    "값": "Value",
    "value": "Value",
    "measurement": "Value",
    "result": "Value",
    "reading": "Value",
}
_STUDY6_SAMPLE_ALIASES = {
    "sample", "sampleid", "item", "itemid", "part", "partid", "id",
    "번호", "샘플", "샘플번호", "시료", "시료번호", "부품", "부품번호"
}
_STUDY6_STANDARD_ALIASES = {
    "standard", "master", "reference", "true", "actual", "기준", "기준값", "표준", "표준값", "정답", "판정기준"
}
_STUDY6_POSITIVE_TOKENS = {
    "1", "true", "t", "yes", "y", "ng", "fail", "reject", "rejected", "defect", "bad", "불량", "부적합", "불합격"
}
_STUDY6_NEGATIVE_TOKENS = {
    "0", "false", "f", "no", "n", "ok", "pass", "accept", "accepted", "good", "양품", "적합", "합격"
}


def _normalize_column_token(column_name) -> str:
    return re.sub(r"[\s_/()\-]+", "", str(column_name).replace("\ufeff", "").strip()).lower()


def _read_file_bytes(file) -> bytes:
    if hasattr(file, "getvalue"):
        raw = file.getvalue()
    else:
        if hasattr(file, "seek"):
            file.seek(0)
        raw = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
    if not raw:
        raise ValueError("빈 파일입니다.")
    return raw


def _read_csv_bytes(raw: bytes, header):
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            text = raw.decode(encoding)
            sample = "\n".join(text.splitlines()[:10])
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
            return pd.read_csv(io.StringIO(text), sep=delimiter, header=header)
        except Exception as exc:
            last_error = exc
    raise last_error or ValueError("CSV 파일을 읽지 못했습니다.")


def _normalize_loaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).replace("\ufeff", "").strip() for col in normalized.columns]
    return normalized


def _pick_best_candidate(candidates):
    return sorted(candidates, key=lambda item: (item[0], item[1] == "infer"), reverse=True)[0][2]


def _get_data_size(data) -> int:
    if isinstance(data, pd.DataFrame):
        return len(data.index)
    if isinstance(data, pd.Series):
        return len(data.index)
    if isinstance(data, np.ndarray):
        return int(data.size)
    return 0


def _rename_columns_by_alias(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    rename_map = {}
    used_targets = set()
    for col in df.columns:
        normalized = _normalize_column_token(col)
        target = aliases.get(normalized)
        if target and target not in used_targets:
            rename_map[col] = target
            used_targets.add(target)
    return df.rename(columns=rename_map)


def load_and_normalize_data(file, normalizer):
    """헤더 포함/미포함 파일을 모두 시도해 가장 타당한 데이터 구조를 선택"""
    candidates = []
    errors = []
    for header in ("infer", None):
        try:
            normalized = normalizer(safe_load_df(file, header=header))
            candidates.append((_get_data_size(normalized), header, normalized))
        except Exception as exc:
            errors.append(str(exc))
    if not candidates:
        unique_errors = list(dict.fromkeys(errors))
        raise ValueError(unique_errors[0] if len(unique_errors) == 1 else " / ".join(unique_errors))
    return _pick_best_candidate(candidates)


def detect_outliers(values: np.ndarray, method: str = "iqr") -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 3:
        return {"has_outliers": False, "outlier_indices": [], "outlier_values": [], "method": method, "bounds": (None, None)}

    if method == "zscore":
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std <= 1e-12:
            return {
                "has_outliers": False,
                "outlier_indices": [],
                "outlier_values": [],
                "method": method,
                "bounds": (mean - 3 * std, mean + 3 * std),
            }
        z_scores = np.abs((arr - mean) / std)
        mask = z_scores > 3.0
        lower, upper = mean - 3 * std, mean + 3 * std
    else:
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (arr < lower) | (arr > upper)

    indices = list(np.where(mask)[0])
    return {
        "has_outliers": bool(mask.any()),
        "outlier_indices": indices,
        "outlier_values": [float(arr[index]) for index in indices],
        "method": method,
        "bounds": (lower, upper),
    }


def check_duplicate_measurements(df: pd.DataFrame, value_col: str = "측정값", group_cols: list = None) -> dict:
    if df is None or df.empty:
        return {"has_duplicates": False, "duplicate_count": 0, "duplicate_rows": pd.DataFrame()}

    columns = group_cols if group_cols else list(df.columns)
    duplicate_mask = df.duplicated(subset=columns, keep=False)
    duplicate_rows = df[duplicate_mask]
    return {
        "has_duplicates": bool(duplicate_mask.any()),
        "duplicate_count": int(duplicate_mask.sum()),
        "duplicate_rows": duplicate_rows,
    }


def normalize_study1_values(df: pd.DataFrame) -> np.ndarray:
    """Study-1 입력 표에서 숫자 측정값만 추출"""
    if df is None or df.empty:
        raise ValueError("Study-1 데이터가 비어 있습니다.")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float).ravel()
    values = values[~np.isnan(values)]
    if values.size < 2:
        raise ValueError("Study-1 데이터는 숫자 측정값 2개 이상이 필요합니다.")
    return values


def normalize_study2_df(df: pd.DataFrame) -> pd.DataFrame:
    """Study 2/3 입력 파일을 표준 컬럼 구조로 정리"""
    if df is None or df.empty:
        raise ValueError("업로드된 데이터가 비어 있습니다.")
    if df.shape[1] < 3:
        raise ValueError("Study-2/3 데이터는 최소 3개 열(부품, 측정자, 측정값)이 필요합니다.")

    normalized = _rename_columns_by_alias(_normalize_loaded_columns(df), _STUDY2_COLUMN_ALIASES)

    if all(col in normalized.columns for col in STUDY2_REQUIRED_COLUMNS):
        normalized = normalized[STUDY2_REQUIRED_COLUMNS].copy()
    else:
        normalized = normalized.iloc[:, :3].copy()
        normalized.columns = STUDY2_REQUIRED_COLUMNS

    normalized["부품"] = normalized["부품"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    normalized["측정자"] = normalized["측정자"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    normalized["측정값"] = pd.to_numeric(normalized["측정값"], errors="coerce")
    normalized = normalized.dropna(subset=["측정값"])
    normalized = normalized[(normalized["부품"] != "") & (normalized["측정자"] != "")]

    if normalized.empty:
        raise ValueError("Study-2/3 데이터에 사용할 유효한 행이 없습니다.")

    return normalized.reset_index(drop=True)


def normalize_study4_df(df: pd.DataFrame) -> pd.DataFrame:
    """Study-4 입력 파일을 [Reference, Value] 구조로 정리"""
    if df is None or df.empty:
        raise ValueError("Study-4 데이터가 비어 있습니다.")
    if df.shape[1] < 2:
        raise ValueError("Study-4 데이터는 최소 2개 열(참조값, 측정값)이 필요합니다.")

    normalized = _rename_columns_by_alias(_normalize_loaded_columns(df), _STUDY4_COLUMN_ALIASES)
    if all(col in normalized.columns for col in STUDY4_REQUIRED_COLUMNS):
        normalized = normalized[STUDY4_REQUIRED_COLUMNS].copy()
    else:
        normalized = normalized.iloc[:, :2].copy()
        normalized.columns = STUDY4_REQUIRED_COLUMNS

    normalized["Reference"] = pd.to_numeric(normalized["Reference"], errors="coerce")
    normalized["Value"] = pd.to_numeric(normalized["Value"], errors="coerce")
    normalized = normalized.dropna(subset=STUDY4_REQUIRED_COLUMNS).reset_index(drop=True)

    if len(normalized) < 3:
        raise ValueError("Study-4 데이터는 유효한 측정값 3개 이상이 필요합니다.")
    if normalized["Reference"].nunique() < 2:
        raise ValueError("Study-4 데이터는 서로 다른 참조값이 2개 이상 필요합니다.")
    return normalized


def normalize_study5_df(df: pd.DataFrame) -> pd.DataFrame:
    """Study-5 입력 파일에서 숫자 부분군 데이터만 추출"""
    if df is None or df.empty:
        raise ValueError("Study-5 데이터가 비어 있습니다.")

    normalized = _normalize_loaded_columns(df).copy()
    for col in normalized.columns:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized = normalized.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    if normalized.shape[1] < 2:
        raise ValueError("Study-5 데이터는 부분군당 숫자 측정값 2열 이상이 필요합니다.")
    if len(normalized) < 2:
        raise ValueError("Study-5 데이터는 부분군 2개 이상이 필요합니다.")

    if all(str(col).isdigit() for col in normalized.columns):
        normalized.columns = [f"Trial {idx}" for idx in range(1, normalized.shape[1] + 1)]
    return normalized


def _normalize_binary_series(series: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isna().all():
        valid_numeric = numeric.dropna()
        if set(valid_numeric.unique()).issubset({0, 1}):
            return numeric

    raw = series.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    non_blank = raw != ""
    lowered = raw.str.lower().map(_normalize_column_token)
    mapped = pd.Series(np.nan, index=series.index, dtype="float")
    mapped[lowered.isin(_STUDY6_POSITIVE_TOKENS)] = 1.0
    mapped[lowered.isin(_STUDY6_NEGATIVE_TOKENS)] = 0.0

    unresolved = lowered[non_blank & mapped.isna()].dropna().unique().tolist()
    if unresolved:
        unique_tokens = lowered[non_blank].dropna().unique().tolist()
        if len(unique_tokens) == 2 and all(
            token not in _STUDY6_POSITIVE_TOKENS and token not in _STUDY6_NEGATIVE_TOKENS for token in unique_tokens
        ):
            fallback_map = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
            mapped[non_blank] = lowered[non_blank].map(fallback_map).astype(float)
        else:
            raise ValueError(
                f"Study-6 열 '{column_name}'은(는) 0/1 또는 OK/NG, PASS/FAIL, 양품/불량 형태여야 합니다."
            )
    return mapped


def normalize_study6_df(df: pd.DataFrame) -> pd.DataFrame:
    """Study-6 입력 파일을 [Sample, Standard, Appraiser...] 구조로 정리"""
    if df is None or df.empty:
        raise ValueError("Study-6 데이터가 비어 있습니다.")
    if df.shape[1] < 2:
        raise ValueError("Study-6 데이터는 기준값과 평가자 결과가 필요합니다.")

    normalized = _normalize_loaded_columns(df).copy()
    column_tokens = {col: _normalize_column_token(col) for col in normalized.columns}

    sample_col = next((col for col, token in column_tokens.items() if token in _STUDY6_SAMPLE_ALIASES), None)
    standard_col = next((col for col, token in column_tokens.items() if token in _STUDY6_STANDARD_ALIASES), None)

    if standard_col is None:
        if len(normalized.columns) < 3:
            raise ValueError("Study-6 데이터는 최소 3개 열(Sample, Standard, Appraiser1...)이 필요합니다.")
        sample_col = normalized.columns[0]
        standard_col = normalized.columns[1]

    appraiser_cols = [col for col in normalized.columns if col not in {sample_col, standard_col}]
    if not appraiser_cols:
        raise ValueError("Study-6 데이터에는 평가자 결과 열이 1개 이상 있어야 합니다.")

    if sample_col is None:
        sample_values = pd.Series(range(1, len(normalized) + 1), name="Sample")
    else:
        sample_values = normalized[sample_col].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
        sample_values = sample_values.where(sample_values != "", pd.Series(range(1, len(normalized) + 1), index=normalized.index))

    result = pd.DataFrame({"Sample": sample_values})
    result["Standard"] = _normalize_binary_series(normalized[standard_col], standard_col)
    for idx, col in enumerate(appraiser_cols, start=1):
        if isinstance(col, str) and col.strip() and not _normalize_column_token(col).isdigit():
            appraiser_name = col
        else:
            appraiser_name = f"Appraiser{idx}"
        result[appraiser_name] = _normalize_binary_series(normalized[col], appraiser_name)

    required_cols = [col for col in result.columns if col != "Sample"]
    result = result.dropna(subset=required_cols).reset_index(drop=True)
    if len(result) < 2:
        raise ValueError("Study-6 데이터에 사용할 유효한 행이 부족합니다.")
    return result


def safe_load_df(file, header="infer") -> pd.DataFrame:
    """Streamlit UploadedFile 객체를 읽어 데이터프레임으로 반환"""
    try:
        ext = os.path.splitext(file.name)[1].lower()
        raw = _read_file_bytes(file)
        if ext == ".csv":
            df = _read_csv_bytes(raw, header=header)
        elif ext in [".xlsx", ".xls"]:
            excel_header = 0 if header == "infer" else None
            df = pd.read_excel(io.BytesIO(raw), header=excel_header)
        else:
            raise ValueError("CSV 또는 XLSX 파일만 지원합니다.")
        return _normalize_loaded_columns(df)
    except Exception as e:
        raise ValueError(f"파일 로드 실패: {e}")

def calc_u_distribution(val, method, CP=0.9545):
    """불확도 분포에 따른 표준 불확도 계산"""
    if method == "균등분포":
        return val / np.sqrt(12)
    elif method == "정규분포":
        return val / (2 * norm.ppf((1 + CP) / 2))
    elif method == "삼각분포":
        return val / np.sqrt(24)
    elif method == "U분포":
        return val / np.sqrt(8)
    return 0.0

def generate_study2_template():
    """Study 2/3 데이터 입력을 위한 Excel 템플릿 생성"""
    output = io.BytesIO()
    # 10개 부품, 3명 측정자, 3회 반복 예시 데이터 생성
    parts = np.repeat(np.arange(1, 11), 9)
    operators = np.tile(np.repeat(['A', 'B', 'C'], 3), 10)
    
    # 임의의 측정값 생성 (예: 평균 50, 표준편차 0.5)
    np.random.seed(42)
    values = np.round(np.random.normal(50, 0.5, 90), 3)
    
    df = pd.DataFrame({
        "부품 (Part)": parts,
        "측정자 (Operator)": operators,
        "측정값 (Value)": values
    })
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='GageRR_Data')
    return output.getvalue()

