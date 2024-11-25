from typing import List, Optional

from dirtyjson.attributed_containers import AttributedList

from .premiums_data import PREMIUMS_DATA

VALID_COMPANIES = ["Income", "AIA", "GE", "HSBC", "Prudential", "Raffles", "Singlife"]
VALID_TIERS = ["Standard", "A", "B", "Private"]


# Filter the data based on the given arguments
def retrieve_premiums(
    age: Optional[List[str]] = None,
    company: Optional[List[str]] = None,
    plan: Optional[List[str]] = None,
):
    data = PREMIUMS_DATA

    print(age, company, plan)

    # Check input types
    for arg, inp in {"age": age, "company": company, "plan": plan}.items():
        if inp and type(inp) != list and type(inp) != AttributedList:
            raise ValueError(f"{arg} is {type(inp)}, should be {list}")

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

        for company_key, company_data in age_data["companies"].items():
            # If company is specified, filter for those companies
            if company and company_key not in company:
                continue

            # If 'plan' is specified, filter for those plans
            plans_data = {}
            if plan:
                for plan_key in plan:
                    if plan_key in company_data:
                        plans_data[plan_key] = company_data.get(plan_key)
            else:
                plans_data = company_data  # Include all plans under the company

            if company_key not in filtered_data[age_key]["companies"]:
                filtered_data[age_key]["companies"][company_key] = {}

            for plan_key, plan_data in plans_data.items():
                if plan_data["additional_insurance_premium"] != "Not available":
                    difference = float(
                        plan_data.get("additional_insurance_premium", 0)
                    ) - float(age_data.get("additional_withdrawal_limit", 0))
                    plan_data["cash_outlay"] = difference if difference >= 0 else "None"
                else:
                    plan_data["cash_outlay"] = "Not available"

                # Add the selected plans data (all plans if 'plan' is not specified) to the filtered data dictionary
                filtered_data[age_key]["companies"][company_key][plan_key] = plan_data

    return prettify_results(filtered_data)


# Helper function to format currency as Singapore Dollars
def format_currency(value):
    if isinstance(value, (int, float)) and value != "N/A":
        return f"${value:,.2f}"
    return value


def prettify_results(filtered_data):
    if not filtered_data:
        return "No data found for the given filters."
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

                for plan, plan_data in company_data.items():
                    message += (
                        f"    Plan: {plan}\n"
                        f"      Additional Insurance Premium: {format_currency(plan_data['additional_insurance_premium'])}\n"
                        f"      Cash Outlay: {format_currency(plan_data['cash_outlay'])}\n"
                    )
        return message


def main():
    # This should throw an error
    # retrieve_premiums(age=None, company="foo")
    print(
        retrieve_premiums(
            age=[32], company=["AIA"], plan=["Standard", "B", "A", "Private"]
        )
    )


if __name__ == "__main__":
    main()
