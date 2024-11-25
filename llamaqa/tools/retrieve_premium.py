import json
import argparse
import pandas as pd

# Load the JSON data from the file
def load_data(json_file_path):
    with open(json_file_path, 'r') as jsonfile:
        return json.load(jsonfile)

# Filter the data based on the given arguments
def filter_data(data, age=None, company=None, plan=None):
    filtered_data = {}

    for age_key, age_data in data.items():
        # If age is specified, filter for those ages
        if age and age_key not in age:
            continue

        if age_key not in filtered_data:
            filtered_data[age_key] = {
                "medishield_premium": age_data.get('medishield_premium', 'N/A'),
                "additional_withdrawal_limit": age_data.get('additional_withdrawal_limit', 'N/A'),
                "companies": {}
            }

        for company_key, company_data in age_data['companies'].items():
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
                if plan_data['additional_insurance_premium'] != 'Not available':
                    difference = float(plan_data.get('additional_insurance_premium', 0)) - float(age_data.get('additional_withdrawal_limit', 0))
                    plan_data['cash_outlay'] = difference if difference >= 0 else 'None'
                else: 
                    plan_data['cash_outlay'] = 'Not available'
            
                # Add the selected plans data (all plans if 'plan' is not specified) to the filtered data dictionary
                filtered_data[age_key]["companies"][company_key][plan_key] = plan_data

    return filtered_data

# Helper function to format currency as Singapore Dollars
def format_currency(value):
    if isinstance(value, (int, float)) and value != 'N/A':
        return f"${value:,.2f}"
    return value

# Print the filtered results
def print_results_list(filtered_data):
    if not filtered_data:
        print("No data found for the given filters.")
    else:
        for age, age_data in filtered_data.items():
            print(f"Age: {age}")
            print(f"  MediShield Life Premiums (Fully payable by Medisave): {format_currency(age_data['medishield_premium'])}")
            print(f"  Additional Withdrawal Limits (AWLs): {format_currency(age_data['additional_withdrawal_limit'])}")

            for company, company_data in age_data['companies'].items():
                print(f"  Company: {company}")

                for plan, plan_data in company_data.items():
                    print(f"    Plan: {plan}")
                    print(f"      Additional Insurance Premium: {format_currency(plan_data['additional_insurance_premium'])}")
                    print(f"      Cash Outlay: {format_currency(plan_data['cash_outlay'])}")
            print()

# Print the filtered results in a table format
def print_results_table(filtered_data):
    if not filtered_data:
        print("No data found for the given filters.")
    rows = []
    for age, age_data in filtered_data.items():
        row = {
            "Age Next Birthday": age,
            "MediShield Premium": format_currency(age_data['medishield_premium']),
            "Additional Withdrawal Limits": format_currency(age_data['additional_withdrawal_limit']),
        }
        for company, company_data in age_data['companies'].items():
            for plan, plan_data in company_data.items():
                row[f"{company} - {plan} - Additional Insurance Premium"] = format_currency(plan_data['additional_insurance_premium'])
                row[f"{company} - {plan} - Cash Outlay"] = format_currency(plan_data['cash_outlay'])
        rows.append(row)

    # Create a DataFrame and print it as a table
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print("\n---\n")

def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Filter insurance premium data")
    parser.add_argument('--age', type=str, nargs='+', choices=[str(i) for i in range(1, 101)] + ['Over 100'], help='Age (from 1 to 100, and "Over 100")')
    parser.add_argument('--company', type=str, nargs='+', choices=['Income', 'AIA', 'GE', 'HSBC', 'Prudential', 'Raffles', 'Singlife'], help='Company (Income, AIA, GE, HSBC, Prudential, Raffles, Singlife)')
    parser.add_argument('--plan', type=str, nargs='+', choices=['Private', 'A', 'B', 'Standard'], help='Plan (Private, A, B, Standard)')
    parser.add_argument('--format', type=str, choices=['list', 'table'], default='list', help='Output format (list or table)')

    args = parser.parse_args()

    # Load data from the JSON file
    data = load_data('premiums.json')

    # Filter the data based on the command line arguments
    filtered_data = filter_data(data, age=args.age, company=args.company, plan=args.plan)

    # Print the results in the specified format
    if args.format == 'table':
        print_results_table(filtered_data)
    else:
        print_results_list(filtered_data)

if __name__ == "__main__":
    main()
