from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
REFERENCES = ROOT / "data" / "references"

CHARTERS = RAW / "legal_charters"
SSI_REVIEWS = RAW / "ssi_review_announcements"
BROKER = RAW / "broker_forecast_reports"

REVIEW_PDF = SSI_REVIEWS / "Thay_doi_thanh_phan_chi_so_VN30_Q2_2026.pdf"
NEWSLETTER_PDF = BROKER / "Du_bao_thay_doi_danh_muc_chi_so_VN30_VNFINLEAD_Q3_2026.pdf"
PAPER_PDF = REFERENCES / "1-s2.0-S0952197626009528-main.pdf"

# --- them 2026-10: bo sung 5 PDF moi ---
REVIEW_Q3_PDF = SSI_REVIEWS / "Thay_doi_thanh_phan_chi_so_VN30_Q3_2026.pdf"
REVIEW_VND_Q1_PDF = SSI_REVIEWS / "Thay_doi_thanh_phan_chi_so_VNDiamond_VN30_VNFIN_Q1_2026.pdf"
REVIEW_VND_Q2_PDF = SSI_REVIEWS / "Thay_doi_thanh_phan_chi_so_VNDiamond_VN30_VNFIN_Q2_2026.pdf"
BROKER_Q1_PDF = BROKER / "Du_bao_thay_doi_danh_muc_chi_so_VN30_VNFINLEAD_Q1_2026.pdf"
BROKER_VND_Q2_PDF = BROKER / "Du_bao_thay_doi_danh_muc_chi_so_VNDiamond_Q2_2026.pdf"
