# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 12:33:05 2019

@author: rodri
"""

#### Agriculture productivity Analysis Uganda 2010-11

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/Research/python Albert')
from data_functions_albert import remove_outliers
os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/Master tesi/Data & Code (Uganda)/data10')
pd.options.display.float_format = '{:,.2f}'.format

dollars = 2586.89



# AGRICULTURAL SEASON 1:

#Omit rents for evaluate agricultural productivity.


# =============================================================================
# Fertilizers & labor inputs
# =============================================================================
ag3a = pd.read_stata('agsec3a.dta')
ag3a = ag3a[["HHID", 'pltid', "a3aq8", "a3aq19", 'a3aq31', 'a3aq38', 'a3aq39', 'a3aq42a', 'a3aq42b','a3aq42c']]
ag3a['hhlabor'] = ag3a["a3aq39"].fillna(0) 
ag3a['hired_labor'] = ag3a["a3aq42a"].fillna(0)+ag3a["a3aq42b"].fillna(0)+ag3a["a3aq42c"].fillna(0) #Sum over hours men, women and kids. We assume all of them equally productive.

ag3a = ag3a[["HHID", 'pltid', "a3aq8", "a3aq19", "a3aq31",'hhlabor', 'hired_labor' ]]
ag3a.columns = ["HHID", 'pltid','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor']


# =============================================================================
# Crop choice and Seeds costs
# =============================================================================
ag4a = pd.read_stata('agsec4a.dta')
ag4a = ag4a[["HHID",'pltid', 'cropID' ,'a4aq8', 'a4aq11']]
ag4a = ag4a[["HHID", 'pltid','cropID' , 'a4aq8',  "a4aq11"]]
ag4a.columns = ["HHID", 'pltid','cropID' , 'area_planted', 'seed_cost']


# =============================================================================
# Output
# =============================================================================
ag5a = pd.read_stata('agsec5a.dta')
ag5a = ag5a[["HHID",'pltid',"cropID","a5aq6a","a5aq6c","a5aq6d","a5aq7a","a5aq7c","a5aq8","a5aq10","a5aq12","a5aq13","a5aq14a","a5aq14b","a5aq15","a5aq22"]]
ag5a.columns = ["HHID", 'pltid', "cropID", "total","unit", "tokg", "sell", "unit2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]


# Convert all quantitites to kilos:
#1.1 get median conversations (self-reported values)
conversion_kg = ag5a.groupby(by="unit")[["tokg"]].median()
conversion_kg.reset_index(inplace=True)
conversion_kg.loc[conversion_kg.unit==1, "tokg"] = 1
conversion_kg.columns = ["unit","kgconverter"]
ag5a = ag5a.merge(conversion_kg, on="unit", how="left")

# Convert to kg
ag5a[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]] = ag5a[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]].multiply(ag5a["kgconverter"], axis="index")


#1.2 Check reported quantities
ag5a["total"] = ag5a["total"].fillna(0)
ag5a["total2"] =  ag5a.loc[:,["sell","gift","cons","food_prod","animal", "seeds", "stored"]].sum(axis=1)
ag5a["diff_totals"] = ag5a.total -ag5a.total2
count_equal = len(ag5a.loc[ag5a.total==ag5a.total2])
count_bigger = len(ag5a.loc[ag5a.total>ag5a.total2])
count_smaller = len(ag5a.loc[ag5a.total<ag5a.total2])

#Prices
ag5a["prices"] = ag5a.value_sells.div(ag5a.sell, axis=0) 
prices = ag5a.groupby(by=["cropID"])[["prices"]].median()

#Check price values in Dollars: they make sense!!!!
prices_usd = ag5a.groupby(by=["cropID"])[["prices"]].median()/dollars
prices.reset_index(inplace=True)
prices.columns=["cropID","p_sell"]


ag5a = ag5a.merge(prices, on="cropID", how="left")

quant = ["total","total2","sell","gift","cons","food_prod","animal","seeds","stored"]
priceslist = ["p_sell"] 
#to check production value for the 3 type of prices uncomment:
#priceslist = ["p_sell", "p_c", "p_c_gate"] 
values_ag5a = ag5a[["HHID", 'pltid', 'cropID', "trans_cost"]]
#Generate values for each quantities and each type of price. Now I only use for sellings prices since the consumption ones where to big.
for q in quant:
    for p in priceslist:
        values_ag5a[q+"_value_"+p] = ag5a[q]*ag5a[p]

ag5a = values_ag5a.groupby(by=["HHID",  'pltid']).sum()
ag5a = ag5a.reset_index()


# =============================================================================
# Capital of the plots/households
# =============================================================================
ag10 = pd.read_stata('agsec10.dta')
ag10 = ag10[['HHID', 'a10q2','a10q3','a10q7','a10q8']]
ag10 = ag10[['HHID', 'a10q2']]
ag10 = ag10.groupby(by="HHID").sum()
ag10 = ag10.reset_index()
ag10.columns = ['HHID','farm_capital']


# Merge datasets -------------------------------------------
agrica = pd.merge(ag3a, ag4a, on=['HHID','pltid'], how='outer')
agrica = pd.merge(agrica, ag5a, on=['HHID','pltid'], how='right')
agrica = pd.merge(agrica, ag10, on='HHID', how='right')
agrica.set_index(['HHID','pltid'], inplace=True)
agrica = agrica.reset_index()


del ag3a, ag4a, ag5a, ag10,  conversion_kg, count_bigger, count_equal, count_smaller, p, prices, prices_usd, priceslist, q, quant, values_ag5a

agrica = agrica.groupby(by=["HHID"])[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'seed_cost', 'trans_cost', 'area_planted',  'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell', 'stored_value_p_sell', 'farm_capital']].sum()

# Pass monetary variables to US2013 $
agrica[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell','stored_value_p_sell', 'farm_capital']] = agrica[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell','stored_value_p_sell', 'farm_capital']]/dollars

# Remove outliers: top 0.5%
agrica[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost','pesticides',  'total2_value_p_sell', 'farm_capital', 'area_planted', 'hhlabor', 'hired_labor']] = remove_outliers(agrica[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'farm_capital','area_planted', 'hhlabor', 'hired_labor']], lq=0, hq=0.975)

# computing productivity levels
agrica['season'] = 1
agrica['k'] = agrica['farm_capital']
agrica['m'] = agrica['org_fert'].fillna(0)+ agrica['chem_fert'].fillna(0)+ agrica['pesticides'].fillna(0)+ agrica['seed_cost'].fillna(0)
agrica['l'] = agrica['hhlabor'].fillna(0)+ agrica['hired_labor'].fillna(0)
agrica['A'] = agrica['area_planted']
agrica['y'] = agrica['total2_value_p_sell'] - agrica['trans_cost'] 

agrica['y_over_A'] = (agrica['y']/agrica['A']).replace([-np.inf, np.inf], np.nan)
agrica['y_over_AL'] = (agrica['y_over_A']/agrica['l']).replace([-np.inf, np.inf], np.nan)

variables = ['k', 'm', 'l', 'A', 'y', 'y_over_A', 'y_over_AL']
for var in variables:
    agrica['ln'+var] = np.log(agrica[var].dropna()+np.abs(np.min(agrica[var]))).replace(-np.inf, np.nan)



# AGRICULTURAL SEASON 2:


# =============================================================================
# Fertilizers & labor inputs
# =============================================================================
ag3b = pd.read_stata('agsec3b.dta')
ag3b = ag3b[["HHID", 'pltid', "a3bq8", "a3bq19", 'a3bq31', 'a3bq38', 'a3bq39', 'a3bq42a', 'a3bq42b','a3bq42c']]
ag3b['hhlabor'] = ag3b["a3bq39"].fillna(0) 
ag3b['hired_labor'] = ag3b["a3bq42a"].fillna(0)+ag3b["a3bq42b"].fillna(0)+ag3b["a3bq42c"].fillna(0) #Sum over hours men, women and kids. We assume all of them equally productive.

ag3b = ag3b[["HHID", 'pltid', "a3bq8", "a3bq19", "a3bq31",'hhlabor', 'hired_labor']]
ag3b.columns = ["HHID", 'pltid','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor']



# =============================================================================
# Crop choice and Seeds costs
# =============================================================================
ag4b = pd.read_stata('agsec4b.dta')
ag4b = ag4b[["HHID",'pltid', 'cropID' ,'a4bq8', 'a4bq11']]
ag4b = ag4b[["HHID", 'pltid','cropID' , 'a4bq8', 'a4bq11']]
ag4b.columns = ["HHID", 'pltid','cropID' , 'area_planted', 'seed_cost']
#COST


# =============================================================================
# Output
# =============================================================================
ag5b = pd.read_stata('agsec5b.dta')
ag5b = ag5b[["HHID",'pltid',"cropID","a5bq6a","a5bq6c","a5bq6d","a5bq7a","a5bq7c","a5bq8","a5bq10","a5bq12","a5bq13","a5bq14a","a5bq14b","a5bq15","a5bq22"]]
ag5b.columns = ["HHID", 'pltid', "cropID", "total","unit", "tokg", "sell", "unit2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]

# Convert all quantitites to kilos:
#1.1 get median conversations (self-reported values)
conversion_kg = ag5b.groupby(by="unit")[["tokg"]].median()
conversion_kg.reset_index(inplace=True)
conversion_kg.loc[conversion_kg.unit==1, "tokg"] = 1
conversion_kg.columns = ["unit","kgconverter"]
ag5b = ag5b.merge(conversion_kg, on="unit", how="left")

# Convert to kg
ag5b[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]] = ag5b[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]].multiply(ag5b["kgconverter"], axis="index")

#1.2 Check reported quantities
ag5b["total"] = ag5b["total"].fillna(0)
ag5b["total2"] =  ag5b.loc[:,["sell","gift","cons","food_prod","animal", "seeds", "stored"]].sum(axis=1)
ag5b["diff_totals"] = ag5b.total -ag5b.total2
count_equal = len(ag5b.loc[ag5b.total==ag5b.total2])
count_bigger = len(ag5b.loc[ag5b.total>ag5b.total2])
count_smaller = len(ag5b.loc[ag5b.total<ag5b.total2])

#Prices
ag5b["prices"] = ag5b.value_sells.div(ag5b.sell, axis=0) 
prices = ag5b.groupby(by=["cropID"])[["prices"]].median()

#Check price values in Dollars: they make sense!!!!
prices_usd = ag5b.groupby(by=["cropID"])[["prices"]].median()/dollars
prices.reset_index(inplace=True)
prices.columns=["cropID","p_sell"]


ag5b = ag5b.merge(prices, on="cropID", how="left")


quant = ["total","total2","sell","gift","cons","food_prod","animal","seeds","stored"]
priceslist = ["p_sell"] 
#to check production value for the 3 type of prices uncomment:
#priceslist = ["p_sell", "p_c", "p_c_gate"] 
values_ag5b = ag5b[["HHID", 'pltid', 'cropID', "trans_cost"]]
#Generate values for each quantities and each type of price. Now I only use for sellings prices since the consumption ones where to big.
for q in quant:
    for p in priceslist:
        values_ag5b[q+"_value_"+p] = ag5b[q]*ag5b[p]


# Summarize results for crop
'''
ag5bcrop = values_ag5b.groupby(by=["HHID", 'cropID']).sum()
ag5bcrop= ag5bcrop.replace(0, np.nan)
ag5bcrop = ag5bcrop.dropna(subset=['total_value_p_sell'])
ag5bcrop = ag5bcrop.reset_index()
crop_count = pd.value_counts(ag5bcrop['cropID'])/ len(ag5bcrop)
crop_sum = ag5bcrop.groupby(by=['cropID']).mean()/dollars
#Diversification of crops per household
hh_count = pd.value_counts(ag5bcrop['HHID']).to_frame() 
hh_count = pd.value_counts(hh_count['HHID'])/len(hh_count)

ag5bcrop = ag5bcrop.reset_index()
sumag5bcrop = ag5bcrop.describe()/dollars
'''

ag5b = values_ag5b.groupby(by=["HHID",  'pltid']).sum()
ag5b = ag5b.reset_index()


# =============================================================================
# Capital of the plots/households
# =============================================================================
ag10 = pd.read_stata('agsec10.dta')
ag10 = ag10[['HHID', 'a10q2','a10q3','a10q7','a10q8']]
ag10 = ag10[['HHID', 'a10q2']]
ag10 = ag10.groupby(by="HHID").sum()
ag10 = ag10.reset_index()
ag10.columns = ['HHID','farm_capital']


# Merge datasets -------------------------------------------
agricb= pd.merge(ag3b, ag4b, on=['HHID','pltid'], how='outer')
agricb = pd.merge(agricb, ag5b, on=['HHID','pltid'], how='right')
agricb = pd.merge(agricb, ag10, on='HHID', how='right')
agricb.set_index(['HHID','pltid'], inplace=True)
agricb = agricb.reset_index()

del ag3b, ag4b, ag5b,  conversion_kg, count_bigger, count_equal, count_smaller, p, prices, prices_usd, priceslist, q, quant, values_ag5b

agricb = agricb.groupby(by=["HHID"])[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'seed_cost', 'trans_cost', 'area_planted',  'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell', 'stored_value_p_sell', 'farm_capital']].sum()


# Pass monetary variables to US2013 $
agricb[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell','stored_value_p_sell', 'farm_capital']] = agricb[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell','stored_value_p_sell', 'farm_capital']]/dollars

# Remove outliers: top 0.5%
agricb[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost','pesticides',  'total2_value_p_sell', 'farm_capital', 'area_planted', 'hhlabor', 'hired_labor']] = remove_outliers(agricb[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'farm_capital', 'area_planted', 'hhlabor', 'hired_labor']], lq=0, hq=0.975)


# computing productivity levels
agricb['season'] = 2
agricb['k'] = agricb['farm_capital']/dollars
agricb['m'] = (agricb['org_fert'].fillna(0)+ agricb['chem_fert'].fillna(0)+ agricb['pesticides'].fillna(0)+ agricb['seed_cost'].fillna(0))
agricb['l'] = agricb['hhlabor'].fillna(0)+ agricb['hired_labor'].fillna(0)
agricb['A'] = agricb['area_planted']
agricb['y'] = (agricb['total2_value_p_sell'] - agricb['trans_cost']) 

agricb['y_over_A'] = (agricb['y']/agricb['A']).replace([-np.inf, np.inf], np.nan)
agricb['y_over_AL'] = (agricb['y_over_A']/agricb['l']).replace([-np.inf, np.inf], np.nan)

variables = ['k', 'm', 'l', 'A', 'y', 'y_over_A', 'y_over_AL']
for var in variables:
    agricb['ln'+var] = np.log(agricb[var].dropna()+np.abs(np.min(agricb[var]))).replace(-np.inf, np.nan)



#%% Merge datasets
data = agrica.append(agricb)
####### PASS EVERYTHING TO DOLLARS



data.to_csv('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/data/agric_data10_hhlevel.csv', index=False)
print('data_saved')



#### Summary statistics data
sum_inp = data[['k','m','org_fert','chem_fert','pesticides','seed_cost']].describe()
sum_labor = data[['A','l','hhlabor','hired_labor', 'trans_cost']].describe()
sum_y = data[['y', 'y_over_A', 'y_over_AL','sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell']].describe()


lnk = np.log(agricb['k'].dropna())
lnk = lnk.replace(-np.inf, np.nan)
lnk = lnk.dropna()

lnA = np.log(agricb['A'].dropna())
lnA = lnA.replace(-np.inf, np.nan)
lnA = lnA.dropna()

lny = (np.log(agricb['y'].dropna()).replace(-np.inf, np.nan)).dropna()

lnm = (np.log(agricb['m'].dropna()).replace(-np.inf, np.nan)).dropna()
lnl = (np.log(agricb['l'].dropna()).replace(-np.inf, np.nan)).dropna()

lny_over_A = (np.log(agricb['y_over_A'].dropna()).replace([-np.inf,np.inf], np.nan)).dropna()
lny_over_AL = (np.log(agricb['y_over_AL'].dropna()).replace([-np.inf,np.inf], np.nan)).dropna()




#Plot Capital distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnk, label="K")
plt.title('Distribution of Farm Capital in Uganda 2010-2011')
plt.xlabel('Log of Farm Capital (in $)')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot hours distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnl, label="L")
plt.title('Distribution of Labour in farm in Uganda 2010-2011')
plt.xlabel('log of Farm Labour (in hours)')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Area plot distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnA, label="A")
plt.title('Distribution of Plots Area in Uganda 2010-2011')
plt.xlabel('log of Plot Area (in Acres)')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Area plot distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnm, label="m")
plt.title('Distribution of inputs in Uganda 2010-2011')
plt.xlabel('log of inputs (in US 2013 $)')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny, label="y")
plt.title('Distribution of Production in Uganda 2010-2011')
plt.xlabel('log of Agricultural Production (in US 2013 $)')
plt.ylabel("Density")
plt.legend()
plt.show()


#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny_over_A, label="y/A")
plt.title('Distribution of Production per Acre in Uganda 2010-2011')
plt.xlabel('log of Agricultural Production per Acre')
plt.ylabel("Density")
plt.legend()
plt.show()


fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny_over_A, label="y/AL")
plt.title('Distribution of Production per Acre and per Hour in Uganda 2010-2011')
plt.xlabel('log of Agricultural Production per Acre and per Hour')
plt.ylabel("Density")
plt.legend()
plt.show()

