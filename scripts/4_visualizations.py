import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

path_to_data = '/workspace/data/'
path_to_figures = '/workspace/figures/'

MICU_admits_clean = pd.read_csv(path_to_data + 'MICU_admits_clean.csv')

fig = plt.figure(figsize=(12,6))

ax = sns.violinplot(x="future_readmit", hue="gender", y="age",data=MICU_admits_clean, split=True)
plt.legend(loc='lower left')
plt.xlabel('Re-admission')
plt.ylabel('Age [years]')
plt.title('Age distributions for MICU admissions')
fig.savefig(path_to_figures + 'Age_distribution_all.png')

fig = plt.figure(figsize=(12,6))
ax = sns.violinplot(x="future_readmit", hue="gender", y="age",data=MICU_admits_clean[MICU_admits_clean.age<300], split=True)
plt.legend(loc='lower left')
plt.xlabel('Re-admission')
plt.ylabel('Age [years]')
plt.title('Age distributions for MICU admissions \n (excluding ages > 300)')
fig.savefig(path_to_figures + 'Age_distribution_under300.png')

MICU_single_admit = MICU_admits_clean.loc[MICU_admits_clean.future_readmit == 'No']
MICU_readmit = MICU_admits_clean.loc[MICU_admits_clean.future_readmit == 'Yes']

fig = plt.figure(figsize=(12,6))

plt.subplot(121)
MICU_single_admit.insurance.value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle = 180
    
)
plt.title('Insurance provider, MICU single admission')
plt.ylabel('')

plt.subplot(122)
MICU_readmit.insurance.value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle = 180
)
plt.title('Insurance provider, MICU readmission')
plt.ylabel('')

fig.savefig(path_to_figures + 'insurance.png')

fig = plt.figure(figsize = (10, 5))
plt.subplot(121)
MICU_admits_clean.future_readmit.value_counts().plot.pie(
    labels = ['No readmission', 'Readmitted'],
    startangle = 90,
    autopct='%1.1f%%'
)
plt.ylabel('')
plt.title('Admissions to the MICU')

plt.subplot(122)
MICU_admits_clean.readmit_last_careunit.value_counts().plot.pie(
    labels = MICU_admits_clean.readmit_last_careunit.value_counts().keys(),
    startangle = 90,
    autopct='%1.1f%%',
    explode = (0.1, 0, 0, 0, 0)
)
plt.title('Last care unit\n readmissions to the MICU')
plt.ylabel('')

fig.savefig(path_to_figures + 'MICU_readmit.png')

data = MICU_admits_clean
print(data.shape)
print('Number of unique patient ids: {}'.format(len(data.subject_id.unique())))
print('Number of (rows, columns): {}'.format(data.shape))
print('Number of unique patient ids: {} \n'.format(len(data.subject_id.unique())))
print(data[['age']].describe().loc[['mean', 'std']],'\n')
data['age(<300)'] = data.loc[data.age<300][['age']]
print(data[['age(<300)']].describe().loc[['mean', 'std']], '\n')
data = data.drop(['age(<300)'], axis=1)
#print('sapsii\n', data.sapsii.describe()[['mean', 'std']],'\n')
#print('sofa\n', data.sofa.describe()[['mean', 'std']],'\n')
print(data[['urea_n_min','urea_n_mean','urea_n_max']].describe().loc[['mean', 'std']], '\n')
print(data[['magnesium_max','albumin_min','calcium_min']].describe().loc[['mean', 'std']], '\n')
print(data[['resprate_min','resprate_mean','resprate_max']].describe().loc[['mean', 'std']], '\n')
print(data[['glucose_min','glucose_mean','glucose_max']].describe().loc[['mean', 'std']], '\n')
print(data[['hr_min','hr_mean','hr_max']].describe().loc[['mean', 'std']], '\n')
print(data[['sysbp_min','sysbp_mean','sysbp_max']].describe().loc[['mean', 'std']], '\n')
print(data[['diasbp_min','diasbp_mean','diasbp_max']].describe().loc[['mean', 'std']], '\n')
print(data[['temp_min','temp_mean','temp_max']].describe().loc[['mean', 'std']], '\n')
#print(data[['urine_min','urine_mean','urine_max']].describe().loc[['mean', 'std']])

fig = plt.figure(figsize=(5,5))
data.gender.value_counts().plot.pie(startangle = 90, autopct='%1.1f%%')
plt.title('Gender split, preprocessed data')
plt.ylabel('')
fig.savefig(path_to_figures + 'preprocessed_gender.png')

fig = plt.figure(figsize=(5,5))
data.marital_status.value_counts().plot.pie(startangle = 0, autopct='%1.1f%%')
plt.title('Marital status split, preprocessed data')
plt.ylabel('')
fig.savefig(path_to_figures + 'preprocessed_marital.png')

fig = plt.figure(figsize=(5,5))
data.insurance.value_counts().plot.pie(startangle = 0, autopct='%1.1f%%')
plt.title('Insurance provider split, preprocessed data')
plt.ylabel('')
fig.savefig(path_to_figures + 'preprocessed_insurance.png')

print(np.sum(data.isnull()))

data_neg = MICU_single_admit 
data_pos = MICU_readmit 

fig = plt.figure(figsize=(15,15))
'''
plt.subplot(331)
data_neg.sofa.plot.kde(color = 'red', alpha = 0.5)
data_pos.sofa.plot.kde(color = 'blue', alpha = 0.5)
plt.title('sofa')
plt.legend(labels=['No readmission', 'Readmission'])

plt.subplot(332)
data_neg.sapsii.plot.kde(color = 'red', alpha = 0.5)
data_pos.sapsii.plot.kde(color = 'blue', alpha = 0.5)
plt.title('sapsii')
plt.legend(labels=['No readmission', 'Readmission'])
'''

plt.subplot(333)
data_neg.platelets_min.plot.kde(color = 'red', alpha = 0.5)
data_pos.platelets_min.plot.kde(color = 'blue', alpha = 0.5)
plt.title('platelets_min')
plt.legend(labels=['No readmission', 'Readmission'])

plt.subplot(334)
data_neg.age.plot.kde(color = 'red', alpha = 0.5)
data_pos.age.plot.kde(color = 'blue', alpha = 0.5)
plt.title('Age')
plt.legend(labels=['No readmission', 'Readmission'])

plt.subplot(335)
data_neg.albumin_min.plot.kde(color = 'red', alpha = 0.5)
data_pos.albumin_min.plot.kde(color = 'blue', alpha = 0.5)
plt.title('albumin_min')
plt.legend(labels=['No readmission', 'Readmission'])

plt.subplot(336)
data_neg.resprate_mean.plot.kde(color = 'red', alpha = 0.5)
data_pos.resprate_mean.plot.kde(color = 'blue', alpha = 0.5)
plt.title('resprate_mean')
plt.legend(labels=['No readmission', 'Readmission'])

plt.subplot(337)
data_neg.sysbp_min.plot.kde(color = 'red', alpha = 0.5)
data_pos.sysbp_min.plot.kde(color = 'blue', alpha = 0.5)
plt.title('sysbp_min')
plt.legend(labels=['No readmission', 'Readmission'])

plt.subplot(338)
data_neg.temp_mean.plot.kde(color = 'red', alpha = 0.5)
data_pos.temp_mean.plot.kde(color = 'blue', alpha = 0.5)
plt.title('temp_mean')
plt.legend(labels=['No readmission', 'Readmission'])

plt.subplot(339)
data_neg.resprate_max.plot.kde(color = 'red', alpha = 0.5)
data_pos.resprate_max.plot.kde(color = 'blue', alpha = 0.5)
plt.title('resprate_max')
plt.legend(labels=['No readmission', 'Readmission'])
fig.savefig(path_to_figures + 'most_important_kdes.png')

print('done')