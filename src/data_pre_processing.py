## add library imports - not sure how to structure this for the git


import numpy as np
import pandas as pd

tcr_data_raw = pd.read_csv('/Volumes/ritd-ag-project-rd0017-bmcha43/CRUK_datathon_2025/tcrictionary_tabular.csv' )



#---------------------------------------------------------------------------------------------------
# pre-process
#---------------------------------------------------------------------------------------------------

# change some columns
tcr_data= tcr_data_raw.copy()

# remove PTMs
tcr_data['epitope_full'] = tcr_data['epitope']
tcr_data['epitope'] = tcr_data['epitope_full'].str.split('+').str[0].str.strip()

# get lengths of TCRs and epitopes
tcr_data['CDR3A_length'] = tcr_data['CDR3A'].str.len().astype('Int64')
tcr_data['CDR3B_length'] = tcr_data['CDR3B'].str.len().astype('Int64')
tcr_data['epitope_length'] = tcr_data['epitope'].str.len().astype('Int64')


# get middle 5 amino acids
def get_middle_five(aa_string):
    if pd.isna(aa_string):  # Check for NaN/null values
        return aa_string
    length = len(aa_string)
    mid = length // 2 + 1 # integer division will ensure that doesnt matter if even or odd - it is floor so always rounds down
    start = mid - 3 # for even numbers this means that the closer to the back 5 are chosen
    end = mid + 2
    return aa_string[start:end]

tcr_data['CDR3A_middle'] = tcr_data['CDR3A'].apply(get_middle_five)
tcr_data['CDR3B_middle'] = tcr_data['CDR3B'].apply(get_middle_five)
tcr_data['epitope_middle'] = tcr_data['epitope'].apply(get_middle_five)


#---------------------------------------------------------------------------------------------------
# filter
#---------------------------------------------------------------------------------------------------
# 1. only human
tcr_data= tcr_data[ tcr_data['TCR species'] == 'HomoSapiens'].copy()

# 2. only with epitopes
tcr_data= tcr_data[ ~tcr_data['epitope'].isna()].copy()

# 3. keep only class 1
tcr_data = tcr_data[tcr_data['MHC class'] == 1]


#---------------------------------------------------------------------------------------------------
# split
#---------------------------------------------------------------------------------------------------

# remove validation studies
validation_studies = ['PMID:38039963', 'PMID:27959684', 'PMID:32461371']
study_pattern = '|'.join(validation_studies)

tcr_data_validate = tcr_data[tcr_data['Studies'].str.contains(study_pattern, na=False)]  # need to do str match for multiple studies
tcr_data_train = tcr_data[~tcr_data['Studies'].str.contains(study_pattern, na=False)]

# remove leaking epitopes
validation_epitopes = tcr_data_validate['epitope'].unique()
epitope_pattern = '|'.join(validation_epitopes)

leaking_epitopes = tcr_data_train[tcr_data_train['epitope'].str.contains(epitope_pattern, na=False)]

tcr_data_train = tcr_data_train[~tcr_data_train['epitope'].str.contains(epitope_pattern, na=False)]
tcr_data_validate = pd.concat([tcr_data_validate, leaking_epitopes])

#---------------------------------------------------------------------------------------------------
# further filtering (for split)
#---------------------------------------------------------------------------------------------------

# a. filter epitopes
# (Matt found paper supporting epitopes between 8-13 (PMID:26783342) ) 
num_pre_filter = tcr_data['epitope'].nunique()
print('pre length filter:', num_pre_filter)

tcr_data = tcr_data[ tcr_data['epitope_length'] < 14 ]
tcr_data = tcr_data[ tcr_data['epitope_length'] > 7 ]

num_post_filter = tcr_data['epitope'].nunique()
print('post length filter:', num_post_filter)
print('removed:', num_pre_filter - num_post_filter)

# b. filter CDR3A
mean = tcr_data['CDR3A_length'].mean()
std = tcr_data['CDR3A_length'].std()
lower_bound = round(mean - 2*std)
upper_bound = round(mean + 2*std)
print('mean: ', mean)
print('std: ', std)
print('LB: ', lower_bound)
print('UB: ', upper_bound)

num_pre_filter = tcr_data['CDR3A'].nunique()
print('pre length filter:', num_pre_filter)

tcr_data = tcr_data[
    (tcr_data['CDR3A_length'].isna()) |  # ignore NA values
    (tcr_data['CDR3A_length'] >= lower_bound) & 
    (tcr_data['CDR3A_length'] <= upper_bound)
]

num_post_filter = tcr_data['CDR3A'].nunique()
print('post length filter:', num_post_filter)
print('removed:', num_pre_filter - num_post_filter)


# c. filter CDR3B
mean = tcr_data['CDR3B_length'].mean()
std = tcr_data['CDR3B_length'].std()
lower_bound = round(mean - 2*std)
upper_bound = round(mean + 2*std)
print('mean: ', mean)
print('std: ', std)
print('LB: ', lower_bound)
print('UB: ', upper_bound)

num_pre_filter = tcr_data['CDR3B'].nunique()
print('pre length filter:', num_pre_filter)

tcr_data = tcr_data[
    (tcr_data['CDR3B_length'].isna()) |  # ignore NA values
    (tcr_data['CDR3B_length'] >= lower_bound) & 
    (tcr_data['CDR3B_length'] <= upper_bound)
]

num_post_filter = tcr_data['CDR3B'].nunique()
print('post length filter:', num_post_filter)
print('removed:', num_pre_filter - num_post_filter)





