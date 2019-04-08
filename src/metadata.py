target_column = 'accepted'
id_column = 'row_id'

categorical_columns = [
    'msa_md',
    'state_code',
    'county_code',
    'lender',
    'loan_type',
    'property_type',
    'loan_purpose',
    'occupancy',
    'preapproval',
    'applicant_ethnicity',
    'applicant_race',
    'applicant_sex',
    'co_applicant'
]

numeric_columns = [
    'loan_amount',
    'applicant_income',
    'population',
    'minority_population_pct',
    'ffiecmedian_family_income',
    'tract_to_msa_md_income_pct',
    'number_of_owner-occupied_units',
    'number_of_1_to_4_family_units'
]

categorical_defaults = {
    'loan_type': 1,  # Conventional
    'applicant_ethnicity': 3,  # Not Provided
    'applicant_race': 6,  # Not Provided
    'applicant_sex': 3  # Not Provided
}

categorical_other_values = {
    'applicant_ethnicity': [3, 4, 5],
    'applicant_race': [6, 7, 8],
    'applicant_sex': [3, 4, 5]
}

categorical_translations = {
    'accepted': {
        0: False,
        1: True
    },
    'loan_type': {
        1: 'Conventional',
        2: 'FHA-Insured',
        3: 'VA-Guaranteed',
        4: 'FSA/RHS'
    },
    'property_type': {
        1: 'One to four-family',
        2: 'Manufactured housing',
        3: 'Multifamily'
    },
    'loan_purpose': {
        1: 'Home purchase',
        2: 'Home improvement',
        3: 'Refinancing'
    },
    'occupancy': {
        1: 'Principal dwelling',
        2: 'Not owner-occupied',
        3: 'Not applicable'
    },
    'preapproval': {
        1: 'Preapproval was requested',
        2: 'Preapproval was not requested',
        3: 'Not applicable'
    },
    'applicant_ethnicity': {
        1: 'Hispanic or Latino',
        2: 'Not Hispanic or Latino',
        3: 'Not provided',
        4: 'Not applicable',
        5: 'No co-applicant'
    },
    'applicant_race': {
        1: 'American Indian or Alaska Native',
        2: 'Asian',
        3: 'Black or African American',
        4: 'Native Hawaiian or Other Pacific Islander',
        5: 'White',
        6: 'Not provided',
        7: 'Not applicable',
        8: 'No co-applicant'
    },
    'applicant_sex': {
        1: 'Male',
        2: 'Female',
        3: 'Not provided',
        4: 'Not applicable',
        5: 'Not applicable'
    }
}
