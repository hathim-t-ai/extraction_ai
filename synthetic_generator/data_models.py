from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from io import StringIO
from typing import List, Optional

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    date: date
    description: str
    debit: Optional[Decimal] = None
    credit: Optional[Decimal] = None
    balance: Decimal

    model_config = {
        "json_encoders": {Decimal: lambda d: str(d)},
    }


class Statement(BaseModel):
    accountNumber: str = Field(alias="account_number")
    currency: str
    periodStart: date = Field(alias="period_start")
    periodEnd: date = Field(alias="period_end")
    transactions: List[Transaction]

    model_config = {
        "json_encoders": {Decimal: lambda d: str(d)},
        "populate_by_name": True,
    }

    # Convenience methods
    def to_csv(self) -> str:
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "date",
            "description",
            "debit",
            "credit",
            "balance",
        ])
        for t in self.transactions:
            writer.writerow([
                t.date.isoformat(),
                t.description.replace("\n", " "),
                t.debit or "",
                t.credit or "",
                t.balance,
            ])
        return output.getvalue()

    def to_jsonl(self) -> str:
        import json

        return json.dumps(self.model_dump(by_alias=True)) + "\n" 