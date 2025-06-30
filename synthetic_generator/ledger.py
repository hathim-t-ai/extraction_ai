from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal

from .config import SyntheticConfig
from .data_models import Statement, Transaction
from .faker_providers import get_faker

faker = get_faker()


def _random_decimal(min_value: float, max_value: float, precision: int = 2) -> Decimal:
    value = random.uniform(min_value, max_value)
    return Decimal(f"{value:.{precision}f}")


def build_statement(cfg: SyntheticConfig) -> Statement:
    today = date.today()
    period_start = today.replace(day=1) - timedelta(days=random.randint(0, 180))
    period_end = period_start.replace(day=28) + timedelta(days=4)  # roughly month-end
    account_number = faker.bban()

    txns: list[Transaction] = []
    running_balance = _random_decimal(1000, 100000)
    for _ in range(random.randint(10, 40)):
        # pick a random day within period
        txn_date = period_start + timedelta(days=random.randint(0, (period_end - period_start).days))
        is_debit = random.random() < 0.5
        amt = _random_decimal(1, 10000)
        debit, credit = (amt, None) if is_debit else (None, amt)
        running_balance = running_balance - amt if is_debit else running_balance + amt
        desc = faker.transaction_description()
        txns.append(
            Transaction(
                date=txn_date,
                description=desc,
                debit=debit,
                credit=credit,
                balance=running_balance,
            )
        )

    # sort chronologically
    txns.sort(key=lambda t: t.date)

    return Statement(
        account_number=account_number,
        currency=cfg.locale.currency,
        period_start=period_start,
        period_end=period_end,
        transactions=txns,
    ) 