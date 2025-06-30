from __future__ import annotations

import random
from decimal import Decimal
from typing import Any

from faker import Faker
from faker.providers import BaseProvider

faker = Faker()


class BankProvider(BaseProvider):
    bank_names = [
        "Axis Bank",
        "HDFC Bank",
        "Emirates NBD",
        "Mashreq Bank",
        "Qatar National Bank",
        "Bank Muscat",
        "National Bank of Bahrain",
        "Kuwait Finance House",
        "State Bank of India",
        "Habib Bank Limited",
    ]

    def bank_name(self) -> str:
        return random.choice(self.bank_names)

    def iban(self) -> str:
        # crude IBAN generator for Gulf region
        country_code = random.choice(["AE", "QA", "BH", "KW", "OM", "SA"])
        return country_code + faker.bban()

    def swift_code(self) -> str:
        return faker.swift8()

    def transaction_description(self) -> str:
        choices = [
            "SALARY",
            "ATM CASH",
            "POS DUBAI MALL",
            "NEFT INB SUPPLIER",
            "UPI/IPAY" + str(random.randint(100000, 999999)),
            "FX CHARGE",
            "CARD FEE",
            "UTILITY BILL",
            "CHEQUE DEPOSIT",
        ]
        if random.random() < 0.3:
            # make multiline
            desc = random.choice(choices) + "\n" + random.choice(choices)
        else:
            desc = random.choice(choices)
        return desc


# register provider
autofaker = Faker()
autofaker.add_provider(BankProvider)

def get_faker() -> Faker:
    return autofaker 