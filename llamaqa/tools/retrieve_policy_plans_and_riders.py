from typing import List

from .retrieve_premiums import COVERAGE_MEANINGS

POLICY_PLANS_RIDERS = {
    "MediShield Life": {
        "MediShield Life": {
            "coverage": "Basic",
            "riders": [],
        }
    },
    "NTUC Income IncomeShield": {
        "Standard": {
            "coverage": "Standard Class B1",
            "riders": [
                "Deluxe Care",
                "Classic Care",
            ],
        },
        "Enhanced Basic": {
            "coverage": "Class B1",
            "riders": [
                "Deluxe Care",
                "Classic Care",
            ],
        },
        "Enhanced Advantage": {
            "coverage": "Class A",
            "riders": [
                "Deluxe Care",
                "Classic Care",
            ],
        },
        "Enhanced Preferred": {
            "coverage": "Private",
            "riders": [
                "Deluxe Care",
                "Classic Care",
            ],
        },
    },
    "AIA HealthShield Gold Max": {
        "Standard": {
            "coverage": "Standard Class B1",
            "riders": [],
        },
        "Max B Lite": {
            "coverage": "Class B1",
            "riders": [
                "AIA Max VitalHealth B Lite",
            ],
        },
        "Max B": {
            "coverage": "Class A",
            "riders": [
                "AIA Max VitalHealth B",
            ],
        },
        "Max A": {
            "coverage": "Private",
            "riders": [
                "AIA Max VitalHealth A",
                "AIA Max VitalHealth A Value",
                "AIA Max VitalHealth A Cancer Care Booster",
            ],
        },
    },
    "Great Eastern GREAT SupremeHealth": {
        "Standard": {
            "coverage": "Standard Class B1",
            "riders": [
                "TotalCare B",
                "TotalCare B Basic",
                "TotalCare Plus",
            ],
        },
        "B Plus": {
            "coverage": "Class B1",
            "riders": [
                "TotalCare B",
                "TotalCare B Basic",
                "TotalCare Plus",
            ],
        },
        "A Plus": {
            "coverage": "Class A",
            "riders": [
                "TotalCare A",
                "TotalCare A Basic",
                "TotalCare Plus",
            ],
        },
        "P Plus": {
            "coverage": "Private",
            "riders": [
                "TotalCare P Signature",
                "TotalCare P Optimum",
                "TotalCare Plus",
            ],
        },
    },
    "HSBC Life Shield": {
        "Standard": {
            "coverage": "Standard Class B1",
            "riders": ["HSBC Life Enhanced Care Rider"],
        },
        "B": {
            "coverage": "Class A",
            "riders": ["HSBC Life Enhanced Care Rider"],
        },
        "A": {
            "coverage": "Private",
            "riders": ["HSBC Life Enhanced Care Rider"],
        },
    },
    "Prudential PRUShield": {
        "Standard": {
            "coverage": "Standard Class B1",
            "riders": [],
        },
        "Plus": {
            "coverage": "Class A",
            "riders": [
                "PRUExtra Plus CoPay",
                "PRUExtra Plus Lite CoPay",
            ],
        },
        "Premier": {
            "coverage": "Private",
            "riders": [
                "PRUExtra Premier CoPay",
                "PRUExtra Preferred CoPay",
                "PRUExtra Premier Lite CoPay",
            ],
        },
    },
    "Raffles Shield": {
        "Standard": {
            "coverage": "Standard Class B1",
            "riders": ["Raffles Cancer Guard Rider"],
        },
        "B": {
            "coverage": "Class B1",
            "riders": [
                "Premier Rider",
                "Key Rider",
                "High Deductible Option",
                "Raffles Cancer Guard Rider",
            ],
        },
        "A": {
            "coverage": "Class AB1",
            "riders": [
                "Premier Rider",
                "Key Rider",
                "High Deductible Option",
                "Raffles Hospital Option",
                "Raffles Cancer Guard Rider",
            ],
        },
        "Private": {
            "coverage": "Private",
            "riders": [
                "Premier Rider",
                "Key Rider",
                "High Deductible Option",
                "Raffles Cancer Guard Rider",
            ],
        },
    },
    "Singlife Shield": {
        "Standard": {
            "coverage": "Standard Class B1",
            "riders": [],
        },
        "Plan 3": {
            "coverage": "Class B1",
            "riders": [
                "Singlife Health Plus Public Prime",
                "Singlife Health Plus Public Lite",
            ],
        },
        "Plan 2": {
            "coverage": "Class A",
            "riders": [
                "Singlife Health Plus Public Prime",
                "Singlife Health Plus Public Lite",
            ],
        },
        "Plan 1": {
            "coverage": "Private",
            "riders": [
                "Singlife Health Plus Private Prime",
                "Singlife Health Plus Private Lite",
            ],
        },
    },
}


GENERAL_TEMPLATE = """
Each insurance offers several different plans and riders.
An insurance rider is an optional add-on to an insurance policy that provides additional benefits or coverage.
Riders can be used to customize an insurance policy to meet the needs of the insured.

{policies}
"""


POLICY_TEMPLATE = """
The {policy_name} policy comprises the following.
{plans_and_riders}
"""

PLAN_TEMPLATE = """
Plan: {plan_name}
Coverage: {coverage} {coverage_meaning}
Riders:
{riders}
"""


def retrieve_policy_plans_and_riders(policies: List[str]):
    return GENERAL_TEMPLATE.format(
        policies="\n".join([POLICY_TEMPLATE.format(
            policy_name=policy,
            plans_and_riders="".join([PLAN_TEMPLATE.format(
                plan_name=plan_name,
                coverage=plan_info["coverage"],
                coverage_meaning=COVERAGE_MEANINGS[plan_info["coverage"]],
                riders="\n".join([f"- {r}" for r in plan_info["riders"]]) if len(plan_info["riders"]) else "None",
            ) for plan_name, plan_info in POLICY_PLANS_RIDERS[policy].items()])
        ) for policy in policies])
    )
