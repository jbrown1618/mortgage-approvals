"""
Define nice labels for our categorical variables.
"""

accepted_translations = {
    0: False,
    1: True
}

loan_type_translations = {
    1: 'Conventional',
    2: 'FHA-Insured',
    3: 'VA-Guaranteed',
    4: 'FSA/RHS'
}

property_type_translations = {
    1: 'One to four-family',
    2: 'Manufactured housing',
    3: 'Multifamily'
}

loan_purpose_translations = {
    1: 'Home purchase',
    2: 'Home improvement',
    3: 'Refinancing'
}

occupancy_translations = {
    1: 'Principal dwelling',
    2: 'Not owner-occupied',
    3: 'Not applicable'
}

preapproval_translations = {
    1: 'Preapproval was requested',
    2: 'Preapproval was not requested',
    3: 'Not applicable'
}

applicant_ethnicity_translations = {
    1: 'Hispanic or Latino',
    2: 'Not Hispanic or Latino',
    3: 'Not provided',
    4: 'Not applicable',
    5: 'No co-applicant'
}

applicant_race_translations = {
    1: 'American Indian or Alaska Native',
    2: 'Asian',
    3: 'Black or African American',
    4: 'Native Hawaiian or Other Pacific Islander',
    5: 'White',
    6: 'Not provided',
    7: 'Not applicable',
    8: 'No co-applicant'
}

applicant_sex_translations = {
    1: 'Male',
    2: 'Female',
    3: 'Not provided',
    4: 'Not applicable',
    5: 'Not applicable'
}

translations = {
    'accepted': accepted_translations,
    'loan_type': loan_type_translations,
    'property_type': property_type_translations,
    'loan_purpose': loan_purpose_translations,
    'occupancy': occupancy_translations,
    'preapproval': preapproval_translations,
    'applicant_ethnicity': applicant_ethnicity_translations,
    'applicant_race': applicant_race_translations,
    'applicant_sex': applicant_sex_translations
}