from typing import Dict, List, Optional

import pandas as pd
from dirtyjson.attributed_containers import AttributedList

from .insurance_plans import INSURANCE_PLANS
from .premiums_data import PREMIUMS_DATA

VALID_COMPANIES = [
    "Income",
    "AIA",
    "Great Eastern",
    "HSBC",
    "Prudential",
    "Raffles",
    "Singlife",
]
VALID_COVERAGE = ["Standard", "Class A", "Class B1", "Private"]
VALID_PLANS = [
    "MediShield Life",
    "Enchanced IncomeShield Preferred",
    "Enchanced IncomeShield Basic Advantage",
    "Enchanced IncomeShield Basic",
    "IncomeShield Standard Plan",
    "AIA HealthShield Gold Max A",
    "AIA HealthShield Gold Max B",
    "AIA HealthShield Gold Max B Lite",
    "AIA HealthShield Gold Standard Plan",
    "GREAT SupremeHealth P Plus",
    "GREAT SupremeHealth A Plus",
    "GREAT SupremeHealth B Plus",
    "GREAT SupremeHealth Standard Plan",
    "HSBC Life Shield Plan A",
    "HSBC Life Shield Plan B",
    "HSBC Life Shield Standard Plan",
    "PRUShield Premier",
    "PRUShield Plus",
    "PRUShield Standard Plan",
    "Raffles Shield Private",
    "Raffles Shield A",
    "Raffles Shield B",
    "Raffles Shield Standard",
    "Singlife Shield Plan 1",
    "Singlife Shield Plan 2",
    "Singlife Shield Plan 3",
    "Singlife Shield Standard Plan",
]


# Filter the data based on the given arguments
def retrieve_premiums(
    age: Optional[List[str]] = None,
    company: Optional[List[str]] = None,
    plan: Optional[List[str]] = None,
    coverage: Optional[List[str]] = None,
    format: str = "list",
):
    data = PREMIUMS_DATA

    # Check input types
    for arg, inp in {
        "age": age,
        "company": company,
        "plan": plan,
        "coverage": coverage,
    }.items():
        if inp and not isinstance(inp, list) and not isinstance(inp, AttributedList):
            raise ValueError(f"{arg} is {type(inp)}, should be {list}")
    if format not in ["list", "table"]:
        raise ValueError(f"format={format} - should be one of ['list', 'table']")

    filtered_data = {}

    # Check input validity
    if age and any(a < 1 for a in age):
        return "The provided age is invalid."
    elif (
        (company and not any(c in VALID_COMPANIES for c in company))
        or (plan and not any(p in VALID_PLANS for p in plan))
        or (coverage and not any(c in VALID_COVERAGE for c in coverage))
    ):
        return "No premium data found for the given query because either the provided company, plan, or coverage level is invalid."

    if age:
        age = ["Over 100" if a > 100 else str(int(a)) for a in age]

    for age_key, age_data in data.items():
        # If age is specified, filter for those ages
        if age and age_key not in age:
            continue

        if age_key not in filtered_data:
            filtered_data[age_key] = {
                "medishield_premium": age_data.get("medishield_premium", "N/A"),
                "additional_withdrawal_limit": age_data.get(
                    "additional_withdrawal_limit", "N/A"
                ),
                "companies": {},
            }

        if plan and plan == ["MediShield Life"]:
            filtered_data[age_key] = {
                "medishield_premium": age_data.get("medishield_premium", "N/A"),
            }
            continue

        for company_key, company_data in age_data["companies"].items():
            # If company is specified and no plan is specified, filter for those companies
            if company and not plan and company_key not in company:
                continue

            plans_data = {}
            # If plan is specified, filter for those plans
            if plan:
                # Find the respective company and coverage in INSURANCE_PLANS
                for plan_key in plan:
                    if plan_key not in INSURANCE_PLANS.get(company_key, {}).values():
                        continue
                    try:
                        coverage = next(
                            coverage
                            for coverage, name in INSURANCE_PLANS[company_key].items()
                            if name == plan_key
                        )
                        plans_data[coverage] = company_data.get(coverage)
                    except StopIteration:
                        # If the plan is not found, skip this company
                        break

            # If coverage is specified but plan is not, filter for those coverages
            elif coverage:
                for coverage_key in coverage:
                    if coverage_key in company_data:
                        plans_data[coverage_key] = company_data.get(coverage_key)

            # If neither plan nor coverage is speciified, include all plans under the company
            else:
                plans_data = company_data

            if company_key not in filtered_data[age_key]["companies"] and plans_data:
                filtered_data[age_key]["companies"][company_key] = {}

            for coverage_key, coverage_data in plans_data.items():
                if coverage_data["additional_insurance_premium"] != "Not available":
                    difference = float(
                        coverage_data.get("additional_insurance_premium", 0)
                    ) - float(age_data.get("additional_withdrawal_limit", 0))
                    coverage_data["cash_outlay"] = max(0, difference)
                else:
                    coverage_data["cash_outlay"] = "Not available"

                # Add the selected plans data (all plans if 'plan' is not specified) to the filtered data dictionary
                filtered_data[age_key]["companies"][company_key][coverage_key] = (
                    coverage_data
                )

    if format == "table":
        return prettify_results_to_table(filtered_data)
    else:
        return prettify_results_to_list(filtered_data)


# Helper function to format currency as Singapore Dollars
def format_currency(value):
    if isinstance(value, (int, float)) and value != "N/A":
        return f"${value:,.2f}"
    return value


def prettify_results_to_list(filtered_data: Dict):
    if not all("companies" in filtered_data[age_key] for age_key in filtered_data):
        message = ""
        for age, age_data in filtered_data.items():
            message += (
                f"Age: {age}\n"
                f"  MediShield Life premium (Fully payable by Medisave): {format_currency(age_data['medishield_premium'])}\n"
            )
        message += "\nAge refers to your age on your next birthday because the insurance will cover your risk for the year ahead."
        return message

    else:
        message = "The premiums for Integrated Shield Plans consist of the MediShield Life premium and the additional private insurance premium.\n\n"
        for age, age_data in filtered_data.items():
            message += (
                f"Age: {age}\n"
                f"  MediShield Life premiums (Fully payable with Medisave): {format_currency(age_data['medishield_premium'])}\n"
                f"  You can use also MediSave to pay for the additional private insurance premium: Up to {format_currency(age_data['additional_withdrawal_limit'])}\n"
            )
            for company, company_data in age_data["companies"].items():
                message += f"  Company: {company}\n"

                for coverage, coverage_data in company_data.items():
                    message += (
                        f"    Plan: {INSURANCE_PLANS[company][coverage]}\n"
                        f"      Additional private insurance premium: {format_currency(coverage_data['additional_insurance_premium'])}\n"
                        f"        Of which, you can pay with MediSave: {format_currency(min(float(coverage_data['additional_insurance_premium']), age_data['additional_withdrawal_limit']) if coverage_data['additional_insurance_premium'] != 'Not available' else 'Up to ' + format_currency(age_data['additional_withdrawal_limit']))}\n"
                        f"        The remaining amount to be paid in cash: {format_currency(coverage_data['cash_outlay'])}\n"
                    )
        message += "\nNote: Age refers to your age on your next birthday because the insurance will cover your risk for the year ahead."
        return message


def prettify_results_to_table(filtered_data: Dict):
    table_results = ""

    if all(
        len(details.get("companies", {})) == 1
        and len(next(iter(details["companies"].values()), {})) == 1
        for details in filtered_data.values()
    ):
        rows = []
        for age, age_data in filtered_data.items():
            for company, company_data in age_data["companies"].items():
                for coverage, coverage_data in company_data.items():
                    plan_title = f"**Plan: {INSURANCE_PLANS[company][coverage]}** \n\n"
                    row = {
                        "Age": age,
                        "MediShield Life premium (Fully payable with MediSave)": format_currency(
                            age_data["medishield_premium"]
                        ),
                        "Additional private insurance premium (Payable with MediSave and cash)": (
                            f"**Total**: {format_currency(coverage_data['additional_insurance_premium'])}<br>"
                            f"**Payable with MediSave**: {format_currency(min(float(coverage_data['additional_insurance_premium']), age_data['additional_withdrawal_limit']) if coverage_data['additional_insurance_premium'] != 'Not available' else 'Up to ' + format_currency(age_data['additional_withdrawal_limit']))}<br>"
                            f"**Cash payment required**: {format_currency(coverage_data['cash_outlay'])}"
                        ),
                    }
                    rows.append(row)

        if rows:
            table = pd.DataFrame(rows).to_markdown(index=False)
            table_results += f"{plan_title}{table}\n"

    else:
        for age, age_data in filtered_data.items():
            age_title = f"**Age: {age}** \n\n"
            rows = []
            for company, company_data in age_data["companies"].items():
                for coverage, coverage_data in company_data.items():
                    row = {
                        "Plan": f"**{INSURANCE_PLANS[company][coverage]}**",
                        "MediShield Life premium (Fully payable by Medisave)": format_currency(
                            age_data["medishield_premium"]
                        ),
                        "Additional private insurance premium (Payable with MediSave and cash)": (
                            f"**Total**: {format_currency(coverage_data['additional_insurance_premium'])}<br>"
                            f"**Payable with MediSave**: {format_currency(min(float(coverage_data['additional_insurance_premium']), age_data['additional_withdrawal_limit']) if coverage_data['additional_insurance_premium'] != 'Not available' else 'Up to ' + format_currency(age_data['additional_withdrawal_limit']))}<br>"
                            f"**Cash payment required**: {format_currency(coverage_data['cash_outlay'])}"
                        ),
                    }
                    rows.append(row)

            if rows:
                table = pd.DataFrame(rows).to_markdown(index=False)
                table_results += f"{age_title}{table}\n\n"

    age_note = "\nNote: Age refers to your age on your next birthday because the insurance will cover your risk for the year ahead."

    return table_results + age_note


def main():
    # This should throw an error
    # retrieve_premiums(age=None, company="foo")
    print(
        retrieve_premiums(
            age=[18, 30, 99],
            company=["AIA"],
            plan=["AIA HealthShield Gold Max A", "IncomeShield Standard Plan"],
            coverage=["Standard"],
            format="table",
        )
    )


if __name__ == "__main__":
    main()
