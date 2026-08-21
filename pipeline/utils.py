"""Business-day helpers and UTR parsing."""

from __future__ import annotations

import re
from datetime import date, timedelta

UTR_PATTERN = re.compile(r"[A-Z0-9]{16}")


def parse_utr(narration: str) -> str:
    match = UTR_PATTERN.search(narration or "")
    return match.group() if match else ""


def business_days_between(d1: date, d2: date) -> int:
    """Absolute business-day distance (Mon–Fri only; Sat/Sun skipped)."""
    if d1 == d2:
        return 0
    if d1 > d2:
        d1, d2 = d2, d1
    count = 0
    current = d1
    while current < d2:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return count


def signed_business_day_diff(bank_date: date, settled_date: date) -> int:
    """Bank date minus settled date in business days (can be negative)."""
    if bank_date == settled_date:
        return 0
    if bank_date > settled_date:
        return business_days_between(settled_date, bank_date)
    return -business_days_between(bank_date, settled_date)
