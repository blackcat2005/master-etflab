"""Quy uoc so Viet Nam: cham = phan tach nghin, phay = thap phan."""

from __future__ import annotations

import pytest

from src.vnnum import is_number, normalize_text, parse_number, strip_accents


@pytest.mark.parametrize(
    ("token", "value"),
    [
        ("2.976,9", 2976.9),
        ("12.930.183", 12930183.0),
        ("1.792.841", 1792841.0),
        ("8,75", 8.75),
        ("0", 0.0),
        ("131.977", 131977.0),
    ],
)
def test_tach_nghin_va_thap_phan(token: str, value: float) -> None:
    num = parse_number(token)
    assert num is not None
    assert num.value == pytest.approx(value)


@pytest.mark.parametrize(
    ("token", "value"),
    [
        ("-1,3", -1.3),
        ("-0,99%", -0.99),
        ("-5.946,2", -5946.2),
        ("(69.559)", -69559.0),
        ("(1.792.841)", -1792841.0),
    ],
)
def test_so_am(token: str, value: float) -> None:
    num = parse_number(token)
    assert num is not None
    assert num.value == pytest.approx(value)


def test_dau_am_va_ngoac_khac_nghiep_chu_so() -> None:
    assert parse_number("-1,3").negative_style == "sign"
    assert parse_number("(69.559)").negative_style == "paren"


@pytest.mark.parametrize("token", ["0,87%", "12,5%", "-0,99%"])
def test_phan_tram(token: str) -> None:
    num = parse_number(token)
    assert num is not None
    assert num.is_percent


@pytest.mark.parametrize(
    "token",
    [
        "30/06/2026",
        "Q3/2026",
        "8/5",
        "H1",
        "abc",
        "Mã CK",
        "",
        "   ",
    ],
)
def test_khong_phai_so(token: str) -> None:
    assert parse_number(token) is None
    assert not is_number(token)


def test_dau_xap_xi() -> None:
    num = parse_number("(~1,8")
    assert num is not None
    assert num.value == pytest.approx(-1.8)
    assert num.approx


def test_tu_don_vi_khong_nhan_he_so() -> None:
    """Don vi da nam o header, nhan o day se tinh dop 1000 lan."""
    num = parse_number("3 triệu")
    assert num is not None
    assert num.value == pytest.approx(3.0)
    assert num.scale_word == "triệu"


def test_tieu_de_trieu_khong_bi_doc_thanh_tr() -> None:
    num = parse_number("2 triệu")
    assert num is not None
    assert num.scale_word == "triệu"


def test_chuan_hoa_chu() -> None:
    assert normalize_text("  a   b \n c ") == "a b c"
    assert strip_accents("Tỷ trọng hiện tại") == "Ty trong hien tai"