import datetime
import io
from pathlib import Path

import numpy as np
import pandas as pd
from fpdf import FPDF


FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
]


def _find_font_path() -> str | None:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return str(path)
    return None


FONT_PATH = _find_font_path()


class ReportPDF(FPDF):
    def __init__(self, title: str, standard: str):
        super().__init__()
        self.title = title
        self.standard = standard
        self.set_auto_page_break(auto=True, margin=15)
        self.font_name = "Helvetica"
        if FONT_PATH:
            self.add_font("AppFont", "", FONT_PATH)
            self.font_name = "AppFont"

    @property
    def content_width(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def set_app_font(self, size: int = 10, bold: bool = False):
        style = "B" if bold and self.font_name == "Helvetica" else ""
        self.set_font(self.font_name, style=style, size=size)

    def header(self):
        self.set_fill_color(57, 93, 170)
        self.set_text_color(255, 255, 255)
        self.set_app_font(12, bold=True)
        self.cell(0, 10, self.title, ln=True, align="C", fill=True)
        self.set_text_color(60, 60, 60)
        self.set_app_font(8)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 6, f"Standard: {self.standard} | Generated: {timestamp}", ln=True, align="R")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_text_color(120, 120, 120)
        self.set_app_font(8)
        self.cell(0, 8, f"MSA AI Studio | Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def section(self, title: str):
        self.ln(2)
        self.set_fill_color(91, 155, 213)
        self.set_text_color(255, 255, 255)
        self.set_app_font(10, bold=True)
        self.cell(0, 8, title, ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def key_value(self, label: str, value: str):
        self.set_app_font(9, bold=True)
        self.cell(55, 7, label, border=0)
        self.set_app_font(9)
        self.multi_cell(0, 7, value, border=0)

    def banner(self, text: str, ok: bool):
        color = (46, 125, 50) if ok else (183, 28, 28)
        self.set_text_color(*color)
        self.set_app_font(12, bold=True)
        self.cell(0, 10, text, ln=True, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def table(self, df: pd.DataFrame, max_rows: int = 30):
        if df is None or df.empty:
            self.set_app_font(9)
            self.cell(0, 7, "No data available.", ln=True)
            return

        display_df = df.head(max_rows).copy()
        if len(df) > max_rows:
            display_df.loc[len(display_df)] = ["..."] * len(display_df.columns)

        columns = [str(column) for column in display_df.columns]
        widths = [self.content_width / len(columns)] * len(columns)

        self.set_fill_color(57, 93, 170)
        self.set_text_color(255, 255, 255)
        self.set_app_font(8, bold=True)
        for index, column in enumerate(columns):
            self.cell(widths[index], 8, column[:28], border=1, align="C", fill=True)
        self.ln()

        self.set_text_color(0, 0, 0)
        self.set_app_font(8)
        alt_fill = False
        for _, row in display_df.iterrows():
            fill_color = (245, 245, 245) if alt_fill else (255, 255, 255)
            self.set_fill_color(*fill_color)
            for index, value in enumerate(row):
                if isinstance(value, float) and not np.isnan(value):
                    text = f"{value:.4f}"
                elif pd.isna(value):
                    text = "-"
                else:
                    text = str(value)
                self.cell(widths[index], 7, text[:28], border=1, align="C", fill=True)
            self.ln()
            alt_fill = not alt_fill
        self.ln(2)


def _metadata_rows(metadata: dict | None) -> list[tuple[str, str]]:
    if not metadata:
        return []
    rows = []
    for key, value in metadata.items():
        if value is None or str(value).strip() == "":
            continue
        rows.append((str(key), str(value)))
    return rows


def _write_metadata(pdf: ReportPDF, metadata: dict | None):
    rows = _metadata_rows(metadata)
    if not rows:
        return
    pdf.section("Metadata")
    for key, value in rows:
        pdf.key_value(key, value)


def _write_params(pdf: ReportPDF, params: dict | None):
    if not params:
        return
    pdf.section("Parameters")
    for key, value in params.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            value_text = f"{float(value):.4f}"
        else:
            value_text = str(value)
        pdf.key_value(str(key), value_text)


def _pdf_bytes(pdf: ReportPDF) -> bytes:
    output = io.BytesIO()
    pdf.output(output)
    return output.getvalue()


def create_study1_pdf(df_summary: pd.DataFrame, raw_X, params: dict, standard: str = "ISO 22514-7", metadata: dict | None = None) -> bytes:
    pdf = ReportPDF("MSA Study 1 Report", standard)
    pdf.add_page()
    _write_metadata(pdf, metadata)
    _write_params(pdf, params)
    pdf.section("Summary")
    pdf.table(df_summary)
    pdf.section("Raw Measurements")
    raw_values = ", ".join(f"{float(value):.4f}" for value in np.asarray(raw_X).ravel()[:200])
    pdf.set_app_font(9)
    pdf.multi_cell(0, 6, raw_values if raw_values else "No raw data available.")
    return _pdf_bytes(pdf)


def create_study2_pdf(result: dict, standard: str = "ISO 22514-7", metadata: dict | None = None) -> bytes:
    pdf = ReportPDF("MSA Study 2/3 Report", standard)
    pdf.add_page()
    _write_metadata(pdf, metadata)
    overall = str(result.get("overall_status", "Review required"))
    pdf.banner(f"Overall status: {overall}", bool(result.get("overall_pass", False)))

    metrics = {
        "NDC": result.get("ndc"),
        "Q_MP": result.get("q_mp"),
        "C_MP": result.get("c_mp"),
    }
    pdf.section("Key Metrics")
    for key, value in metrics.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            text = "-"
        else:
            text = f"{float(value):.4f}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
        pdf.key_value(key, text)

    for label in ("judgement", "gage", "anova"):
        value = result.get(label)
        if isinstance(value, pd.DataFrame):
            pdf.section(label.title())
            pdf.table(value.reset_index() if label == "anova" else value)
    return _pdf_bytes(pdf)


def create_study4_pdf(result: dict, params: dict, standard: str = "ISO 22514-7") -> bytes:
    pdf = ReportPDF("MSA Study 4 Report", standard)
    pdf.add_page()
    pdf.banner(
        "Linearity passed" if bool(result.get("linearity_pass", False)) else "Linearity needs review",
        bool(result.get("linearity_pass", False)),
    )
    _write_params(pdf, params)
    for label in ("summary", "uncertainty_components", "linearity_capability"):
        value = result.get(label)
        if isinstance(value, pd.DataFrame):
            pdf.section(label.replace("_", " ").title())
            pdf.table(value)
    return _pdf_bytes(pdf)


def create_study5_pdf(result: dict, metadata: dict | None = None) -> bytes:
    pdf = ReportPDF("MSA Study 5 Report", "AIAG / ISO 22514-7")
    pdf.add_page()
    _write_metadata(pdf, metadata)
    pdf.banner(
        "Stability passed" if bool(result.get("is_stable", False)) else "Stability needs review",
        bool(result.get("is_stable", False)),
    )
    summary = result.get("summary")
    if isinstance(summary, pd.DataFrame):
        pdf.section("Control Chart Summary")
        pdf.table(summary)

    x_bars = result.get("x_bars", [])
    ranges = result.get("ranges", [])
    if len(x_bars) and len(ranges):
        detail_rows = []
        out_x = result.get("out_x", [])
        out_r = result.get("out_r", [])
        for index, (x_bar, rng) in enumerate(zip(x_bars, ranges), start=1):
            detail_rows.append(
                {
                    "Subgroup": index,
                    "X-bar": float(x_bar),
                    "Range": float(rng),
                    "X flag": "Out" if index - 1 < len(out_x) and out_x[index - 1] else "OK",
                    "R flag": "Out" if index - 1 < len(out_r) and out_r[index - 1] else "OK",
                }
            )
        pdf.section("Subgroup Details")
        pdf.table(pd.DataFrame(detail_rows))
    return _pdf_bytes(pdf)


def create_study6_pdf(res_df: pd.DataFrame, fk: dict, metadata: dict | None = None) -> bytes:
    pdf = ReportPDF("MSA Study 6 Report", "AIAG MSA 4th")
    pdf.add_page()
    _write_metadata(pdf, metadata)
    overall_ok = bool(len(res_df)) and all(str(value).strip() in {"OK", "합격", "적합"} for value in res_df.get("종합 판정", []))
    pdf.banner("Attribute agreement summary", overall_ok)
    pdf.section("Appraiser Results")
    pdf.table(res_df)

    if fk:
        pdf.section("Fleiss Kappa")
        for key in ("fleiss_kappa", "aiag_grade", "minitab_grade", "n_raters", "n_subjects"):
            value = fk.get(key)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                text = "-"
            else:
                text = f"{float(value):.4f}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
            pdf.key_value(key, text)
    return _pdf_bytes(pdf)
