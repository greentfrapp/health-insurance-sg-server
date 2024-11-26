from typing import List, Optional

from dirtyjson.attributed_containers import AttributedList

from .premiums_data import PREMIUMS_DATA
from .insurance_plans import INSURANCE_PLANS

VALID_COMPANIES = ["Income", "AIA", "GE", "HSBC", "Prudential", "Raffles", "Singlife"]
VALID_TIERS = ["Standard", "A", "B", "Private"]
VALID_PLANS = ['Enchanced IncomeShield Preferred',
 'Enchanced IncomeShield Basic Advantage',
 'Enchanced IncomeShield Basic',
 'IncomeShield Standard Plan',
 'AIA HealthShield Gold Max A',
 'AIA HealthShield Gold Max B',
 'AIA HealthShield Gold Max B Lite',
 'AIA HealthShield Gold Standard Plan',
 'GREAT SupremeHealth P Plus',
 'GREAT SupremeHealth A Plus',
 'GREAT SupremeHealth B Plus',
 'GREAT SupremeHealth Standard Plan',
 'HSBC Life Shield Plan A',
 'HSBC Life Shield Plan B',
 'HSBC Life Shield Standard Plan',
 'PRUShield Premier',
 'PRUShield Plus',
 'PRUShield Standard Plan',
 'Raffles Shield Private',
 'Raffles Shield A',
 'Raffles Shield B',
 'Raffles Shield Standard',
 'Singlife Shield Plan 1',
 'Singlife Shield Plan 2',
 'Singlife Shield Plan 3',
 'Singlife Shield Standard Plan']

# Filter the data based on the given arguments
def retrieve_premiums(
    age: Optional[List[str]] = None,
    company: Optional[List[str]] = None,
    plan: Optional[List[str]] = None,
    tier: Optional[List[str]] = None,
):
    data = PREMIUMS_DATA

    print(age, company, plan, tier)

    # Check input types
    for arg, inp in {"age": age, "company": company, "plan": plan, "tier": tier}.items():
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

            plans_data = {}
            # If plan is specified, filter for those plans
            if plan:  
                # Find the respective company and tier in INSURANCE_PLANS
                for plan_key in plan:
                    if plan_key not in INSURANCE_PLANS.get(company_key, {}).values():
                        print(f"{company_key} does not have {plan_key}")
                        continue
                    try:
                        tier = next(tier for tier, name in INSURANCE_PLANS[company_key].items() if name == plan_key)
                        plans_data[tier] = company_data.get(tier)
                    except StopIteration:
                        # If the plan is not found, skip this company
                        break

            # If tier is specified but plan is not, filter for those tiers
            elif tier:
                for tier_key in tier:
                    if tier_key in company_data:
                        plans_data[tier_key] = company_data.get(tier_key)

            # If neither plan nor tier is speciified, include all plans under the company
            else:
                plans_data = company_data 

            if company_key not in filtered_data[age_key]["companies"] and plans_data:
                filtered_data[age_key]["companies"][company_key] = {}

            for tier_key, tier_data in plans_data.items():
                if tier_data["additional_insurance_premium"] != "Not available":
                    difference = float(
                        tier_data.get("additional_insurance_premium", 0)
                    ) - float(age_data.get("additional_withdrawal_limit", 0))
                    tier_data["cash_outlay"] = difference if difference >= 0 else "None"
                else:
                    tier_data["cash_outlay"] = "Not available"

                # Add the selected plans data (all plans if 'plan' is not specified) to the filtered data dictionary
                filtered_data[age_key]["companies"][company_key][tier_key] = tier_data
    
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

                for tier, tier_data in company_data.items():
                    message += (
                        f"    Plan: {INSURANCE_PLANS[company][tier]}\n"
                        f"    Tier: {tier}\n"
                        f"      Additional Insurance Premium: {format_currency(tier_data['additional_insurance_premium'])}\n"
                        f"      Cash Outlay: {format_currency(tier_data['cash_outlay'])}\n"
                    )
        return message


def main():
    # This should throw an error
    # retrieve_premiums(age=None, company="foo")
    print(
        retrieve_premiums(
            age=[32], company=["AIA"], plan=["AIA HealthShield Gold Max B"], tier=["Standard"]
        )
    )


if __name__ == "__main__":
    main()
