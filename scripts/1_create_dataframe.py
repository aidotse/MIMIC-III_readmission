# Import libraries
import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns

path_to_figures = '/workspace/figures/'
sns.set()

##########################################
## Import data rom SQL database
##########################################

# create a database connection
sqluser = 'postgres'
dbname = 'mimic'
schema_name = 'mimiciii'

# Connect to local postgres version of mimic
con = psycopg2.connect(dbname="mimic", user="lissj17", host="localhost", password="postgres")
cur = con.cursor()

# SQL query (gets all_data view defined elsewhere)
query = \
"""
SELECT 'Hello world'
"""

query = \
"""
select * from all_data
"""

# Perform SQL query
cur.execute('SET search_path to ' + schema_name)
data = pd.read_sql_query(query,con)

##########################################
## Preliminary statistics
##########################################

print('Number of (rows, columns): {}'.format(data.shape))
print('Number of unique patient ids: {} \n'.format(len(data.subject_id.unique())))
print(data[['age']].describe().loc[['mean', 'std']],'\n')
data['age(<300)'] = data.loc[data.age<300][['age']]
print(data[['age(<300)']].describe().loc[['mean', 'std']], '\n')
#print('sapsii\n', data.sapsii.describe()[['mean', 'std']],'\n')
#print('sofa\n', data.sofa.describe()[['mean', 'std']],'\n')
print(data[['urea_n_min','urea_n_mean','urea_n_max']].describe().loc[['mean', 'std']],'\n')
print(data[['urea_n_min','urea_n_mean','urea_n_max']].describe().loc[['mean', 'std']], '\n')
print(data[['magnesium_max','albumin_min','calcium_min']].describe().loc[['mean', 'std']], '\n')
print(data[['resprate_min','resprate_mean','resprate_max']].describe().loc[['mean', 'std']], '\n')
print(data[['glucose_min','glucose_mean','glucose_max']].describe().loc[['mean', 'std']], '\n')
print(data[['hr_min','hr_mean','hr_max']].describe().loc[['mean', 'std']], '\n')
print(data[['sysbp_min','sysbp_mean','sysbp_max']].describe().loc[['mean', 'std']], '\n')
print(data[['diasbp_min','diasbp_mean','diasbp_max']].describe().loc[['mean', 'std']], '\n')
print(data[['temp_min','temp_mean','temp_max']].describe().loc[['mean', 'std']], '\n')
print(data[['urine_min','urine_mean','urine_max']].describe().loc[['mean', 'std']])

fig = plt.figure(figsize=(5,5))
data.gender.value_counts().plot.pie(startangle = 90, autopct='%1.1f%%')
plt.title('Gender split, raw data')
plt.ylabel('')
fig.savefig(path_to_figures + 'raw_gender.png')

fig = plt.figure(figsize=(5,5))
data.marital_status.value_counts().plot.pie(startangle = 0, autopct='%1.1f%%')
plt.title('Marital status split, raw data')
plt.ylabel('')
fig.savefig(path_to_figures + 'raw_marital.png')

fig = plt.figure(figsize=(5,5))
data.insurance.value_counts().plot.pie(startangle = 0, autopct='%1.1f%%')
plt.title('Insurance provider split, raw data')
plt.ylabel('')
fig.savefig(path_to_figures + 'raw_insurance.png')

print(np.sum(data.isnull()))

##########################################
## Extract readmission time information
##########################################

# calculate time delta between subsequent readmissions of the same patient 
data['readmit_dt'] = np.zeros(data.shape[0])
data['next_readmit_dt'] = np.zeros(data.shape[0])
data['readmit_last_careunit'] = None

for idx in np.arange(1,data.shape[0]):
    if data.subject_id[idx] == data.subject_id[idx - 1]:     
        prev_disch = data.dischtime[idx-1]
        curr_adm = data.admittime[idx]
        dt = curr_adm - prev_disch
        dt_hrs_calc = np.round(dt.value/3600.0/1e9,2)

#         data.set_value(idx,'adm_num',data['adm_num'][idx-1] + 1) 
        data.at[idx,'readmit_dt'] = dt_hrs_calc
        data.at[idx-1,'next_readmit_dt'] = dt_hrs_calc
        data.at[idx,'readmit_last_careunit'] = data['last_careunit'][idx-1]

    
print(data.shape)

##########################################
## Clean up missing or invalid values
##########################################

data = data.drop(['urine_min','urine_mean','urine_max'], axis = 1) #Too noisy
data = data[data.readmit_dt >= 0] #Ignore cases where readmit_dt < 0, which result from duplicate records. 
data = data[(data.deathtime.isnull())] #Remove cases where the patient died during stay
data = data.drop(['deathtime'], axis = 1) # Important to drop before dropna otherwise most of the data is lost
data = data.dropna(subset=data.keys()[:-1]).reset_index(drop = True) # Ignore NaN values in readmit_last_careunit

print('Dataframe shape after removal of invalid values: \n{}'.format(data.shape))

##########################################
## Define time threshold and corresponding labels
##########################################

# Define threshold in hours
threshold = 30*24

# Define label column based on threshold
data['future_readmit'] = None
data['future_readmit'] = ['No' if dt == 0.0 else 'Yes' if dt<=threshold else 'No' for dt in data.next_readmit_dt]

print('\nValue counts:')
print(data.future_readmit.value_counts())
print('\nValue proportions:')
print(data.future_readmit.value_counts()/data.shape[0])

##########################################
## Focus on admittions to the MICU
##########################################

MICU_admits_clean = data.loc[data.first_careunit == 'MICU']

# A quick look on the label distribution:
print('Value counts:')
print(MICU_admits_clean.future_readmit.value_counts())
print('\nValue proportions:')
print(MICU_admits_clean.future_readmit.value_counts()/MICU_admits_clean.shape[0])

# Save clean dataset to csv
MICU_admits_clean.to_csv('/workspace/data/MICU_admits_clean.csv', index=False)

print('done')