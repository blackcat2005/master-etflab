"""Test cho hàm khớp đáp án dùng chung (``scripts/eval/metric_common.py``).

Lỗi từng gặp: số âm kiểu kế toán ``(36,2)`` bị regex đọc thành ``36,2``
(dương), nên mọi câu hỏi về ô âm đều bị tính là "mất đáp án" — làm sai cả
bảng kết quả của mọi phương pháp. Test này khoá hành vi đó.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

import metric_common as mc


def test_parenthesized_cell_is_negative() -> None:
    assert -36.2 in mc.numbers_in("MAFM VN30: (36,2)")


def test_thousands_in_parentheses() -> None:
    assert -66531.0 in mc.numbers_in("MSB (66.531)")


def test_sign_and_percent_still_parse() -> None:
    nums = mc.numbers_in("BID -21,1 | TCB 8,75%")
    assert -21.1 in nums and 8.75 in nums


def test_matches_row_with_accounting_negative() -> None:
    chunk = SimpleNamespace(text="Mã: BCM; MAFM VN30: (36,2); TỔNG: (416,1)")
    assert mc.matches(chunk, "BCM", -36.2)
    assert not mc.matches(chunk, "BCM", 36.2)


def test_unit_annotation_is_not_a_zero_value() -> None:
    # "('000 cổ phiếu)" là đơn vị, không phải giá trị 0.
    assert mc.numbers_in("Số lượng CP mua/bán ('000 cổ phiếu)") == set()
    assert 0.0 in mc.numbers_in("Số lượng CP ('000 cổ phiếu): 0,00")
    # Không được phép khớp 0,0 chỉ vì header có '000.
    chunk = SimpleNamespace(text="Mã: VPL; Số lượng CP mua/bán ('000 cổ phiếu): 12,5")
    assert not mc.matches(chunk, "VPL", 0.0)
