from typing import Dict, List, Optional

import pandas as pd
from dirtyjson.attributed_containers import AttributedList

from .insurance_plans import INSURANCE_PLANS
from .premiums_data import PREMIUMS_DATA

VALID_COMPANIES = ["Income", "AIA", "GE", "HSBC", "Prudential", "Raffles", "Singlife"]
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

    if age:
        age = [str(a) for a in age]

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
            # If company is specified, filter for those companies
            if company and company_key not in company:
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
                    coverage_data["cash_outlay"] = (
                        difference if difference >= 0 else "None"
                    )
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
                f"  MediShield Life Premiums (Fully payable by Medisave): {format_currency(age_data['medishield_premium'])}\n"
            )
        return message
    elif not filtered_data or not any(
        filtered_data[age_key]["companies"] for age_key in filtered_data
    ):
        return "No premium data found for the given query."
    else:
        message = ""
        for age, age_data in filtered_data.items():
            message += (
                f"Age: {age}\n"
                f"  MediShield Life Premiums (Fully payable by Medisave): {format_currency(age_data['medishield_premium'])}\n"
                f"  Additional Withdrawal Limits (AWLs): {format_currency(age_data['additional_withdrawal_limit'])}\n"
            )
            for company, company_data in age_data["companies"].items():
                message += f"  Company: {company}\n"

                for coverage, coverage_data in company_data.items():
                    message += (
                        f"    Plan: {INSURANCE_PLANS[company][coverage]}\n"
                        f"    Coverage level: {coverage}\n"
                        f"      Additional Insurance Premium: {format_currency(coverage_data['additional_insurance_premium'])}\n"
                        f"      Cash Outlay: {format_currency(coverage_data['cash_outlay'])}\n"
                    )
        return message


def prettify_results_to_table(filtered_data: Dict):
    rows = []
    for age, age_data in filtered_data.items():
        row = {
            "Age Next Birthday": age,
            "MediShield Premium": format_currency(age_data["medishield_premium"]),
            "Additional Withdrawal Limits": format_currency(
                age_data["additional_withdrawal_limit"]
            ),
        }
        for company, company_data in age_data["companies"].items():
            for plan, plan_data in company_data.items():
                row[f"{company} - {plan} - Additional Insurance Premium"] = (
                    format_currency(plan_data["additional_insurance_premium"])
                )
                row[f"{company} - {plan} - Cash Outlay"] = format_currency(
                    plan_data["cash_outlay"]
                )
        rows.append(row)

    return pd.DataFrame(rows).to_markdown(index=False)


def main():
    # This should throw an error
    # retrieve_premiums(age=None, company="foo")
    print(
        retrieve_premiums(
            age=[32],
            company=["AIA"],
            plan=["AIA HealthShield Gold Max B"],
            coverage=["Standard"],
        )
    )


if __name__ == "__main__":
    main()
