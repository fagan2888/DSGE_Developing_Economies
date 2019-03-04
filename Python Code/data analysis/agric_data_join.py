# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:29:34 2018

@author: rodri
"""

### Apppend datasets.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col
os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/phd tesi/data')
from data_functions_albert import remove_outliers, gini
pd.options.display.float_format = '{:,.2f}'.format

dollars = 2586.89

# Import data
data13 = pd.read_csv('agric_data13.csv')
crop_codes = pd.read_csv('crop_codes.csv')
crop_codes.columns = ['cropID','crop_name']
data13 = pd.merge(data13, crop_codes, on='cropID', how='left' )
data13['crop_name'] = data13['crop_name'].str.upper()
data13['wave'] = '2013-14'

data11 = pd.read_csv('agric_data11.csv')
data11['crop_name'] = data11['cropID'].str.upper()
data11['wave'] = '2011-12'
data10 = pd.read_csv('agric_data10.csv')
data10.rename(columns={'cropID':'crop_name'}, inplace=True)
data10['wave'] = '2010-11'
del data13['cropID'], data11['cropID']


data = data13.append(data11)
data = data.append(data10)

#data = data[~data.index.duplicated()]


data['l'].dropna(inplace=True)
data['l'].replace(0, np.nan).dropna(inplace=True)


sumdata = data.describe()
# For some variables differences between max and 75% are too high. Trim at the 1%.


## Remove outliers
data[['chem_fert', 'k', 'l', 'org_fert', 'pesticides', 'seed_cost', 'y']] = remove_outliers(data[['chem_fert', 'k', 'l', 'org_fert', 'pesticides', 'seed_cost', 'y']], lq=0, hq=0.99)
sumdata2 = data.describe()


data['m'] = data['org_fert'].fillna(0)+ data['chem_fert'].fillna(0)+ data['pesticides'].fillna(0)+ data['seed_cost'].fillna(0)
data['m'].replace(0, np.nan, inplace=True)

data['y_over_A'] = (data['y']/data['A']).replace([-np.inf, np.inf], np.nan)
data['y_over_AL'] = (data['y_over_A']/data['l']).replace([-np.inf, np.inf], np.nan)

sumdata3 = data.describe()

variables = ['k', 'm', 'l', 'A', 'y', 'y_over_A', 'y_over_AL']
for var in variables:
    data['ln'+var] = np.log(data[var]).replace(-np.inf, np.nan)


sumdata3 = data.describe()


#%% Plot distributions
lnk = np.log(data['k'].dropna())
lnk = lnk.replace(-np.inf, np.nan)
lnk = lnk.dropna()

lnA = np.log(data['A'].dropna())
lnA = lnA.replace(-np.inf, np.nan)
lnA = lnA.dropna()

lny = (np.log(data['y'].dropna()).replace(-np.inf, np.nan)).dropna()

lnm = (np.log(data['m'].dropna()).replace(-np.inf, np.nan)).dropna()
lny_over_A = (np.log(data['y_over_A'].dropna()).replace([-np.inf,np.inf], np.nan)).dropna()
lny_over_AL = data['lny_over_AL'].dropna()
#Plot Capital distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnk, label="K")
plt.title('Distribution of Farm Capital in Uganda 2013-2014')
plt.xlabel('Log of Farm Capital')
plt.ylabel("Density")
plt.legend()
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/K_distribution.png')

#Plot hours distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(data['l'].dropna(), label="L")
plt.title('Distribution of Labour in farm in Uganda 2013-2014')
plt.xlabel('Farm Labour')
plt.ylabel("Density")
plt.legend()
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/L_distribution.png')


#Plot Area plot distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnA, label="A")
plt.title('Distribution of Plots Area in Uganda 2013-2014')
plt.xlabel('Plot Area (in Acres)')
plt.ylabel("Density")
plt.legend()
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/A_distribution.png')

#Plot Inputs distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnm, label="m")
plt.title('Distribution of inputs in Uganda 2013-2014')
plt.xlabel('log of inputs')
plt.ylabel("Density")
plt.legend()
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/M_distribution.png')


#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny, label="y")
plt.title('Distribution of Production in Uganda 2013-2014')
plt.xlabel('Agricultural Production')
plt.ylabel("Density")
plt.legend()
plt.show()

fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/output_distribution.png')


#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny_over_A, label="y/A")
plt.title('Distribution of Production per Acre in Uganda')
plt.xlabel('Agricultural Production per Acre')
plt.ylabel("Density")
plt.legend()
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/yield_distribution.png')

#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny_over_AL, label="y/AL")
plt.title('Distribution of Production per Acre and per hour Worked in Uganda')
plt.xlabel('Agricultural Production per Acre and per Hour worked')
plt.ylabel("Density")
plt.legend()
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/yieldAL_distribution.png')


#%% Plot distributions separating by wave, season, region

def plot_cond_log_distr(data, variable1, variable2, folder='C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/', save=False):
        fig, ax = plt.subplots()
        a = data[variable2].unique()
        for value in a:           
            sns.distplot((np.log(data.loc[data[variable2] == value][variable1]).replace([-np.inf, np.inf], np.nan)).dropna()-np.mean((np.log(data[variable1]).replace([-np.inf, np.inf], np.nan)).dropna()), label=variable2+str(value))
           
        plt.title('Distribution of '+variable1+' in Uganda')
        plt.xlabel(variable1)
        ax.legend()
        if save == True:
            fig.savefig(folder+'distr'+variable1+variable2+'.png')
            return plt.show()


var_list = ['y','y_over_A','y_over_AL','A','m','l']
for var in var_list:
    plot_cond_log_distr(data,variable1=var, variable2='season')
    
    
var_list = ['y','y_over_A','y_over_AL','A','m','l']
for var in var_list:
    plot_cond_log_distr(data,variable1=var, variable2='region')
    
var_list = ['y','y_over_A','y_over_AL','A','m','l']
for var in var_list:
    plot_cond_log_distr(data,variable1=var, variable2='wave')



#%% OLS: Does Production function follows a cobb-douglas form?
count_crops = pd.value_counts(data['crop_name']).to_frame()
count_crops = count_crops.reset_index()

list_crops = count_crops.iloc[0:18,0]
del list_crops[12], list_crops[15]

list_ols = []
list_ftest = []
list_n = []


for item in list_crops:
     print(item)
     ols= sm.ols(formula=" lny ~ lnk + lnA +lnm +lnl", data=data.loc[data['crop_name']==item, :]).fit()
     print(ols.summary())  
     ftest = ols.f_test(" lnk +lnm +lnA +lnl = 1")
     list_ols.append(ols)
     list_ftest.append(ftest)
     n = len(ols.fittedvalues)
     list_n.append(n)

results_1 = summary_col([list_ols[0], list_ols[1], list_ols[2], list_ols[3], list_ols[4], list_ols[5], list_ols[6], list_ols[7]],stars=True)
results_1 = summary_col([ list_ols[1], list_ols[2], list_ols[4], list_ols[5], list_ols[6], list_ols[7]],stars=True)

print(results_1)
print(results_1.as_latex())


results_2 = summary_col([list_ols[8], list_ols[9], list_ols[11],  list_ols[12], list_ols[13], list_ols[14]],stars=True)
print(results_2.as_latex())




ftests= pd.DataFrame(np.array([list_ftest[0].fvalue[0,0], list_ftest[0].pvalue, list_ftest[1].fvalue[0,0], list_ftest[1].pvalue, list_ftest[2].fvalue[0,0], list_ftest[2].pvalue, list_ftest[3].fvalue[0,0], list_ftest[3].pvalue, list_ftest[4].fvalue[0,0], list_ftest[4].pvalue, list_ftest[5].fvalue[0,0], list_ftest[5].pvalue, list_ftest[6].fvalue[0,0], list_ftest[6].pvalue, list_ftest[7].fvalue[0,0], list_ftest[7].pvalue]))
ftests2= pd.DataFrame(np.array([list_ftest[8].fvalue[0,0], list_ftest[8].pvalue, list_ftest[9].fvalue[0,0], list_ftest[9].pvalue, list_ftest[10].fvalue[0,0], list_ftest[10].pvalue, list_ftest[11].fvalue[0,0], list_ftest[11].pvalue, list_ftest[12].fvalue[0,0], list_ftest[12].pvalue, list_ftest[13].fvalue[0,0], list_ftest[13].pvalue, list_ftest[14].fvalue[0,0], list_ftest[14].pvalue, list_ftest[15].fvalue[0,0], list_ftest[15].pvalue]))

list_f1 = [list_ftest[1].fvalue[0,0], list_ftest[2].fvalue[0,0],  list_ftest[4].fvalue[0,0],  list_ftest[5].fvalue[0,0], list_ftest[6].fvalue[0,0],  list_ftest[7].fvalue[0,0]]
list_pvalue1 = [list_ftest[1].pvalue, list_ftest[2].pvalue, list_ftest[4].pvalue,  list_ftest[5].pvalue, list_ftest[6].pvalue, list_ftest[7].pvalue ]


ftests= pd.DataFrame(np.array([list_ftest[0].fvalue[0,0], list_ftest[0].pvalue, list_ftest[1].fvalue[0,0], list_ftest[1].pvalue, list_ftest[2].fvalue[0,0], list_ftest[2].pvalue, list_ftest[3].fvalue[0,0], list_ftest[3].pvalue, list_ftest[4].fvalue[0,0], list_ftest[4].pvalue, list_ftest[5].fvalue[0,0], list_ftest[5].pvalue, list_ftest[6].fvalue[0,0], list_ftest[6].pvalue, list_ftest[7].fvalue[0,0], list_ftest[7].pvalue]))
ftests2= pd.DataFrame(np.array([list_ftest[8].fvalue[0,0], list_ftest[8].pvalue, list_ftest[9].fvalue[0,0], list_ftest[9].pvalue, list_ftest[10].fvalue[0,0], list_ftest[10].pvalue, list_ftest[11].fvalue[0,0], list_ftest[11].pvalue, list_ftest[12].fvalue[0,0], list_ftest[12].pvalue, list_ftest[13].fvalue[0,0], list_ftest[13].pvalue, list_ftest[14].fvalue[0,0], list_ftest[14].pvalue, list_ftest[15].fvalue[0,0], list_ftest[15].pvalue]))

list_f2 = [list_ftest[8].fvalue[0,0], list_ftest[9].fvalue[0,0],  list_ftest[11].fvalue[0,0],  list_ftest[12].fvalue[0,0], list_ftest[13].fvalue[0,0],  list_ftest[14].fvalue[0,0]]
list_pvalue2 = [list_ftest[8].pvalue, list_ftest[9].pvalue, list_ftest[11].pvalue,  list_ftest[12].pvalue, list_ftest[13].pvalue, list_ftest[14].pvalue ]

pd.options.display.float_format = '{:,.4f}'.format
ftests = pd.concat([ftests, ftests2], axis=1)


#%% As in the model
count_crops = pd.value_counts(data['crop_name']).to_frame()
count_crops = count_crops.reset_index()

list_crops = count_crops.iloc[0:18,0]
del list_crops[12], list_crops[15]

list_ols_short = []
list_ftest_short = []
list_n_short = []


for item in list_crops:
     print(item)
     ols= sm.ols(formula=" lny_over_AL ~ lnm + lnk ", data=data.loc[data['crop_name']==item, :]).fit()
     print(ols.summary())  
     list_ols_short.append(ols)
     n = len(ols.fittedvalues)
     list_n_short.append(n)

results_1_short = summary_col([list_ols_short[0], list_ols_short[1], list_ols_short[2], list_ols_short[3], list_ols_short[4], list_ols_short[5], list_ols_short[6], list_ols_short[7]],stars=True)
print(results_1_short)
results_2_short = summary_col([list_ols_short[8], list_ols_short[9], list_ols_short[10], list_ols_short[11], list_ols_short[12], list_ols_short[13], list_ols_short[14], list_ols_short[15]],stars=True)
print(results_2_short)



pd.options.display.float_format = '{:,.4f}'.format
ftests = pd.concat([ftests, ftests2], axis=1)

olsjoin = sm.ols(formula=" lny_over_AL ~ lnm + lnk ", data=data).fit()
print(olsjoin.summary())  

#%% Risk analysis

list_crops = count_crops.iloc[0:18,0]
del list_crops[12], list_crops[15]

list_avg_prod = []
for item in list_crops:
    avg_prod = data.loc[data['crop_name']==item, :].groupby(by=['wave','season'])['y'].sum()
    list_avg_prod.append(avg_prod)

list_var=[]
for i in range(len(list_avg_prod)):
    var = np.std(list_avg_prod[i])
    list_var.append(var)
    
list_var=[]
for i in range(len(list_avg_prod)):
    var = np.std(list_avg_prod[i])
    list_var.append(var)


#%% Sample statistics per crop
pd.options.display.float_format = '{:,.2f}'.format

count_crops = pd.value_counts(data['crop_name']).to_frame()
count_crops = count_crops.reset_index()
list_crops = count_crops.iloc[0:18,0]
del list_crops[12], list_crops[15]

crop_summary = []
yield_mean = []
yield_sd = []
yield_cv = []

yield_mean_agg = []
yield_sd_agg = []
yield_cv_agg = []
yield_gini_agg = []
A_agg = []
k_agg = []
m_agg = []
l_agg = []

inputs = []
inputs_list = []
for item in list_crops:
     #Get data by crop
     data_crop=data.loc[data['crop_name']==item, ['y','y_over_AL','A','k','m','l']]
     data_xharvest = data.loc[data['crop_name']==item, ['y_over_AL','wave','season']]
     data_inputs = data.loc[data['crop_name']==item, ['k','l','wave','season']]
     
     #Compute aggregate statistics
     summary = data_crop.describe()
     mean_agg = np.mean(data_crop['y_over_AL'])
     A_agg_temp = data_crop['A'].mean()
     k_agg_temp = data_crop['k'].mean()
     m_agg_temp = data_crop['m'].mean()
     l_agg_temp = data_crop['l'].mean()
     
     sd_agg = np.var(data_crop['y_over_AL'])
     gini_agg = gini(np.array(data_crop['y_over_AL'].dropna()))
     cv_agg = sd_agg / mean_agg    
     # Append them to list
     yield_mean_agg.append(mean_agg)
     yield_sd_agg.append(sd_agg)
     yield_cv_agg.append(cv_agg)
     yield_gini_agg.append(gini_agg)
     A_agg.append(A_agg_temp)
     k_agg.append(k_agg_temp)
     m_agg.append(m_agg_temp)
     l_agg.append(l_agg_temp)
     
     #Compute statistics per survey wave
     mean = data_xharvest.groupby(by=["wave"]).mean()  
     sd = data_xharvest.groupby(by=["wave"]).std() 
     inputs = data_inputs.groupby(by=["wave"]).mean()
     cv =  sd/mean 
     
     #Append them to list
     crop_summary.append(summary)
     yield_mean.append(mean)
     yield_sd.append(sd)
     inputs_list.append(inputs)
     yield_cv.append(cv)


# Summary Statistics ------------------------------
sum_data = data[['y','y_over_AL' ,'A','k','m']].describe()
sum_cassava = crop_summary[0]
sum_swpotatoes = crop_summary[1]
sum_beans = crop_summary[2]
sum_bananafood = crop_summary[3]
sum_maize = crop_summary[4]
sum_groundnuts = crop_summary[5]
sum_sorghum = crop_summary[6]
sum_fingermillet = crop_summary[7]
sum_simsim = crop_summary[8]
sum_irishpotatoes = crop_summary[9]
sum_coffee = crop_summary[10]
sum_rice = crop_summary[11]
sum_sunflower = crop_summary[12]
sum_soyabean =crop_summary[13]
sum_fieldpeas = crop_summary[14]
sum_cotton = crop_summary[15]


## Plot mean and Variance across waves--------------------------
## mean crops
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(yield_mean[0].index, yield_mean[0].iloc[:,0] ,label="Cassava")
ax.plot(yield_mean[1].index, yield_mean[1].iloc[:,0] ,label="Sw. Potatoes")
ax.plot(yield_mean[2].index, yield_mean[2].iloc[:,0] ,label="Beans")
ax.plot(yield_mean[3].index, yield_mean[3].iloc[:,0] ,label="Banana Food")
ax.plot(yield_mean[4].index, yield_mean[4].iloc[:,0] ,label="Maize")
ax.plot(yield_mean[5].index, yield_mean[5].iloc[:,0] ,label="Sorghum")
ax.plot(yield_mean[6].index, yield_mean[6].iloc[:,0] ,label="Finger Millet")
ax.plot(yield_mean[7].index, yield_mean[7].iloc[:,0] ,label="SimSim")
ax.plot(yield_mean[8].index, yield_mean[8].iloc[:,0] ,label="Irish Potatoes")
ax.plot(yield_mean[9].index, yield_mean[9].iloc[:,0] ,label="Coffee")
ax.plot(yield_mean[10].index, yield_mean[10].iloc[:,0] ,linestyle='dashed', label="Rice")
ax.plot(yield_mean[11].index, yield_mean[11].iloc[:,0] , linestyle='dashed', label="Sunflower")
ax.plot(yield_mean[12].index, yield_mean[12].iloc[:,0] ,linestyle='dashed', label="Field Peas")
ax.plot(yield_mean[13].index, yield_mean[13].iloc[:,0] , linestyle='dashed', label="Cotton")
plt.title('Agricultural Yields Uganda')
plt.xlabel('Wave')
plt.ylim([1,11])
plt.ylabel("Yield: Output per Acre and per Labor Hour")
plt.legend(loc=2)
plt.show()

## standard deviations of crops
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(yield_sd[0].index, yield_sd[0].iloc[:,0] ,label="Cassava")
ax.plot(yield_sd[1].index, yield_sd[1].iloc[:,0] ,label="Sw. Potatoes")
ax.plot(yield_sd[2].index, yield_sd[2].iloc[:,0] ,label="Beans")
ax.plot(yield_sd[3].index, yield_sd[3].iloc[:,0] ,label="Banana Food")
ax.plot(yield_sd[4].index, yield_sd[4].iloc[:,0] ,label="Maize")
ax.plot(yield_sd[5].index, yield_sd[5].iloc[:,0] ,label="Sorghum")
ax.plot(yield_sd[6].index, yield_sd[6].iloc[:,0] ,label="Finger Millet")
ax.plot(yield_sd[7].index, yield_sd[7].iloc[:,0] ,label="SimSim")
ax.plot(yield_sd[8].index, yield_sd[8].iloc[:,0] ,label="Irish Potatoes")
ax.plot(yield_sd[9].index, yield_sd[9].iloc[:,0] ,label="Coffee")
ax.plot(yield_sd[10].index, yield_sd[10].iloc[:,0] ,linestyle='dashed', label="Rice")
ax.plot(yield_sd[11].index, yield_sd[11].iloc[:,0] , linestyle='dashed', label="Sunflower")
ax.plot(yield_sd[12].index, yield_sd[12].iloc[:,0] ,linestyle='dashed', label="Field Peas")
ax.plot(yield_sd[13].index, yield_sd[13].iloc[:,0] , linestyle='dashed', label="Cotton")
plt.title('Agricultural Risk in Uganda: Standard Deviations of Yields in Uganda')
plt.xlabel('Wave')
plt.ylim([1,40])
plt.ylabel(" Standard Deviation Yield")
plt.legend(loc=2)
plt.show()

## CV of crops
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(yield_cv[0].index, yield_cv[0].iloc[:,0] ,label="Cassava")
ax.plot(yield_cv[1].index, yield_cv[1].iloc[:,0] ,label="Sw. Potatoes")
ax.plot(yield_cv[2].index, yield_cv[2].iloc[:,0] ,label="Beans")
ax.plot(yield_cv[3].index, yield_cv[3].iloc[:,0] ,label="Banana Food")
ax.plot(yield_cv[4].index, yield_cv[4].iloc[:,0] ,label="Maize")
ax.plot(yield_cv[5].index, yield_cv[5].iloc[:,0] ,label="Sorghum")
ax.plot(yield_cv[6].index, yield_cv[6].iloc[:,0] ,label="Finger Millet")
ax.plot(yield_cv[7].index, yield_cv[7].iloc[:,0] ,label="SimSim")
ax.plot(yield_cv[8].index, yield_cv[8].iloc[:,0] ,label="Irish Potatoes")
ax.plot(yield_cv[9].index, yield_cv[9].iloc[:,0] ,label="Coffee")
ax.plot(yield_cv[10].index, yield_cv[10].iloc[:,0] ,linestyle='dashed', label="Rice")
ax.plot(yield_cv[11].index, yield_cv[11].iloc[:,0] , linestyle='dashed', label="Sunflower")
ax.plot(yield_cv[12].index, yield_cv[12].iloc[:,0] ,linestyle='dashed', label="Field Peas")
ax.plot(yield_cv[13].index, yield_cv[13].iloc[:,0] , linestyle='dashed', label="Cotton")
plt.title('Agricultural Risk in Uganda: Standard Deviations of Yields in Uganda')
plt.xlabel('Wave')
#plt.ylim([1,40])
plt.ylabel(" Standard Deviation Yield")
plt.legend(loc=2)
plt.show()



## Capital
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(inputs_list[0].index, inputs_list[0].iloc[:,0] ,label="Cassava")
ax.plot(inputs_list[1].index, inputs_list[1].iloc[:,0] ,label="Sw. Potatoes")
ax.plot(inputs_list[2].index, inputs_list[2].iloc[:,0] ,label="Beans")
ax.plot(inputs_list[3].index, inputs_list[3].iloc[:,0] ,label="Banana Food")
ax.plot(inputs_list[4].index, inputs_list[4].iloc[:,0] ,label="Maize")
ax.plot(inputs_list[5].index, inputs_list[5].iloc[:,0] ,label="Sorghum")
ax.plot(inputs_list[6].index, inputs_list[6].iloc[:,0] ,label="Finger Millet")
ax.plot(inputs_list[7].index, inputs_list[7].iloc[:,0] ,label="SimSim")
ax.plot(inputs_list[8].index, inputs_list[8].iloc[:,0] ,label="Irish Potatoes")
ax.plot(inputs_list[9].index, inputs_list[9].iloc[:,0] ,label="Coffee")
ax.plot(inputs_list[10].index, inputs_list[10].iloc[:,0] ,linestyle='dashed', label="Rice")
ax.plot(inputs_list[11].index, inputs_list[11].iloc[:,0] , linestyle='dashed', label="Sunflower")
ax.plot(inputs_list[12].index, inputs_list[12].iloc[:,0] ,linestyle='dashed', label="Field Peas")
ax.plot(inputs_list[13].index, inputs_list[13].iloc[:,0] , linestyle='dashed', label="Cotton")
plt.title('Average Capital per Plot in Uganda')
plt.xlabel('Wave')
#plt.ylim([1,40])
plt.ylabel(" k (in US$)")
plt.legend(loc=2)
plt.show()


## Inputs
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(inputs_list[0].index, inputs_list[0].iloc[:,1] ,label="Cassava")
ax.plot(inputs_list[1].index, inputs_list[1].iloc[:,1] ,label="Sw. Potatoes")
ax.plot(inputs_list[2].index, inputs_list[2].iloc[:,1] ,label="Beans")
ax.plot(inputs_list[3].index, inputs_list[3].iloc[:,1] ,label="Banana Food")
ax.plot(inputs_list[4].index, inputs_list[4].iloc[:,1] ,label="Maize")
ax.plot(inputs_list[5].index, inputs_list[5].iloc[:,1] ,label="Sorghum")
ax.plot(inputs_list[6].index, inputs_list[6].iloc[:,1] ,label="Finger Millet")
ax.plot(inputs_list[7].index, inputs_list[7].iloc[:,1] ,label="SimSim")
ax.plot(inputs_list[8].index, inputs_list[8].iloc[:,1] ,label="Irish Potatoes")
ax.plot(inputs_list[9].index, inputs_list[9].iloc[:,1] ,label="Coffee")
ax.plot(inputs_list[10].index, inputs_list[10].iloc[:,1] ,linestyle='dashed', label="Rice")
ax.plot(inputs_list[11].index, inputs_list[11].iloc[:,1] , linestyle='dashed', label="Sunflower")
ax.plot(inputs_list[12].index, inputs_list[12].iloc[:,1] ,linestyle='dashed', label="Field Peas")
ax.plot(inputs_list[13].index, inputs_list[13].iloc[:,1] , linestyle='dashed', label="Cotton")
plt.title('Average Intermediate Goods per Plot in Uganda')
plt.xlabel('Wave')
#plt.ylim([1,40])
plt.ylabel(" m (in US$)")
plt.legend(loc=2)
plt.show()

## Statistics aggregating across waves -------------------------------------

### Productivity and Risk aggregating waves
data = [('Crop', list_crops),
         ('Mean', yield_mean_agg),
         ('SD', yield_sd_agg),
         ('CV', yield_cv_agg),
         ('Gini', yield_gini_agg)
         ]
data_crops_agg = pd.DataFrame.from_items(data)

print(data_crops_agg.to_latex())

corr_mean_sd = data_crops_agg['Mean'].corr(data_crops_agg['SD'])
corr_mean_cv = data_crops_agg['Mean'].corr(data_crops_agg['CV'])
corr_mean_gini = data_crops_agg['Mean'].corr(data_crops_agg['Gini'])

## Input choices vs productivity across crops
data_inp = [('Crop', list_crops),
         ('y/AL', yield_mean_agg),
         ('CV(y/AL)', yield_cv_agg),
         ('A', A_agg),
         ('k', k_agg),
         ('m', m_agg),
         ('l', l_agg),]


data_inp_crops_agg = pd.DataFrame.from_items(data_inp)
print(data_inp_crops_agg.to_latex())

## Correlation productivity and inputs
corr_y_A = data_inp_crops_agg['y/AL'].corr(data_inp_crops_agg['A'])
corr_y_k = data_inp_crops_agg['y/AL'].corr(data_inp_crops_agg['k'])
corr_y_m = data_inp_crops_agg['y/AL'].corr(data_inp_crops_agg['m'])
corr_y_l = data_inp_crops_agg['y/AL'].corr(data_inp_crops_agg['l'])

## Correlation risk and inputs
corr_risk_A = data_inp_crops_agg['CV(y/AL)'].corr(data_inp_crops_agg['A'])
corr_risk_k = data_inp_crops_agg['CV(y/AL)'].corr(data_inp_crops_agg['k'])
corr_risk_m = data_inp_crops_agg['CV(y/AL)'].corr(data_inp_crops_agg['m'])
corr_risk_l = data_inp_crops_agg['CV(y/AL)'].corr(data_inp_crops_agg['l'])

#frame it
data_corr = [('input', ['A','k','m','l']),
         ('Productivity (y/AL)', [corr_y_A, corr_y_k, corr_y_m, corr_y_l]),
         ('Risk (CV(y/AL))', [corr_risk_A, corr_risk_k, corr_risk_m, corr_risk_l]),]
data_corr_agg = pd.DataFrame.from_items(data_corr)
print(data_corr_agg.to_latex())

#%% Plot distribution functions per crop

#Plot Agricultural production distribution


for item in list_crops:
    data_crop = data.loc[data['crop_name']==item, ['y', 'y_over_A', 'y_over_AL']]
    crop_serie = (np.log(data_crop['y'].dropna()).replace(-np.inf, np.nan)).dropna()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.distplot(crop_serie, label="y")
    plt.title('Distribution of Production of' +item)
    plt.xlabel('logarithm of Output of '+item)
    plt.ylabel("Density")
    plt.legend()
    plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/yieldAL_distribution.png')


for item in list_crops:
    data_crop = data.loc[data['crop_name']==item, [ 'y_over_A']]
    crop_serie = (np.log(data_crop['y_over_A'].dropna()).replace([-np.inf, np.inf], np.nan)).dropna()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.distplot(crop_serie, label="y/A")
    plt.title('Distribution of Production per Acre of' +item)
    plt.xlabel('logarithm of Output per Acre of '+item)
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    

for item in list_crops:
    data_crop = data.loc[data['crop_name']==item, ['y_over_AL']]
    crop_serie = (np.log(data_crop['y_over_AL'].dropna()).replace([-np.inf, np.inf], np.nan)).dropna()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.distplot(crop_serie, label="y/AL")
    plt.title('Distribution of production per acre and per hour worked of ' +item)
    plt.xlabel('logarithm of output per acre and per hour worked of '+item)
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/yieldAL_distribution_'+item+'.png')
