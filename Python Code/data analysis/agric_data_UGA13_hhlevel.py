# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:17:10 2018

@author: Albert
"""

#### Agriculture productivity Analysis Uganda 2013-14

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/Research/python Albert')
from data_functions_albert import remove_outliers


os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/Master tesi/Data & Code (Uganda)/data13')
pd.options.display.float_format = '{:,.2f}'.format

dollars = 2586.89


# AGRICULTURAL SEASON 1:
#Omit rents for evaluate agricultural productivity.

# =============================================================================
# Fertilizers & labor inputs
# =============================================================================
ag3a = pd.read_csv('agsec3a.csv', header=0, na_values='NA')
ag3a = ag3a[["HHID", 'parcelID','plotID', 'a3aq5', "a3aq8", 'a3aq15',"a3aq18",'a3aq24a','a3aq24b',"a3aq27", 'a3aq33a_1', 'a3aq33b_1', 'a3aq33c_1', 'a3aq33d_1', 'a3aq33e_1', 'a3aq35a', 'a3aq35b','a3aq35c' ,"a3aq36"]]
ag3a['hhlabor'] = ag3a["a3aq33a_1"].fillna(0) +ag3a["a3aq33b_1"].fillna(0) +ag3a["a3aq33c_1"].fillna(0) +ag3a["a3aq33d_1"].fillna(0)+ ag3a["a3aq33e_1"].fillna(0)
ag3a['hired_labor'] = ag3a["a3aq35a"].fillna(0) +ag3a["a3aq35b"].fillna(0)   +ag3a["a3aq35c"].fillna(0)
ag3a = ag3a[["HHID", 'plotID', "a3aq8", "a3aq18", "a3aq27",'hhlabor', 'hired_labor']]
ag3a.columns = ["HHID", 'plotID','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor']


# =============================================================================
# Crop choice and Seeds costs
# =============================================================================
ag4a = pd.read_csv('agsec4a.csv', header=0, na_values='NA')
ag4a = ag4a[["HHID", 'parcelID','plotID', 'ACropCode', 'cropID' , 'a4aq7', 'a4aq9', "a4aq15", 'a4aq13']]
ag4a = ag4a[["HHID", 'plotID','ACropCode', 'cropID', 'a4aq7',  "a4aq15"]]
ag4a.columns = ["HHID", 'plotID','ACropCode', 'cropID', 'area_planted', 'seed_cost']
#COST


# =============================================================================
# Output
# =============================================================================
ag5a = pd.read_csv('agsec5a.csv', header=0, na_values='NA')
ag5a = ag5a[["HHID",'plotID',"cropID","a5aq6a","a5aq6c","a5aq6d","a5aq7a","a5aq7c","a5aq7d","a5aq8","a5aq10","a5aq12","a5aq13","a5aq14a","a5aq14b","a5aq15","a5aq21"]]
ag5a.columns = ["HHID", 'plotID', "cropID", "total","unit", "tokg", "sell", "unit2","tokg2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]


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
values_ag5a = ag5a[["HHID", 'plotID', 'cropID', "trans_cost"]]
#Generate values for each quantities and each type of price. Now I only use for sellings prices since the consumption ones where to big.
for q in quant:
    for p in priceslist:
        values_ag5a[q+"_value_"+p] = ag5a[q]*ag5a[p]

'''
# Summarize results for crop
ag5acrop = values_ag5a.groupby(by=["HHID", 'cropID']).sum()
ag5acrop= ag5acrop.replace(0, np.nan)
ag5acrop = ag5acrop.dropna(subset=['total_value_p_sell'])
ag5acrop = ag5acrop.reset_index()
crop_count = pd.value_counts(ag5acrop['cropID'])/ len(ag5acrop)
crop_sum = ag5acrop.groupby(by=['cropID']).mean()/dollars
#Diversification of crops per household
hh_count = pd.value_counts(ag5acrop['HHID']).to_frame()
hh_count = pd.value_counts(hh_count['HHID'])/len(hh_count)
ag5acrop = ag5acrop.reset_index()
sumag5acrop = ag5acrop.describe()/dollars
'''
#Problem summing over the code of the crop
ag5a = values_ag5a.groupby(by=["HHID",  'plotID']).sum()
ag5a = ag5a.reset_index()



# =============================================================================
# Capital of the plots/households
# =============================================================================
ag10 = pd.read_csv('agsec10.csv')
ag10 = ag10[['HHID','A10itemcod', 'a10q2','a10q3','a10q7','a10q8']]
ag10 = ag10[['HHID', 'a10q2']]
ag10 = ag10.groupby(by="HHID").sum()
ag10 = ag10.reset_index()
ag10.columns = ['HHID','farm_capital']


# Merge datasets -------------------------------------------
agrica = pd.merge(ag3a, ag4a, on=['HHID','plotID'], how='outer')
agrica = pd.merge(agrica, ag5a, on=['HHID','plotID'], how='right')
agrica = pd.merge(agrica, ag10, on='HHID', how='right')
agrica.set_index(['HHID','plotID'], inplace=True)
agrica = agrica.reset_index()

#Sum at household level
agrica = agrica.groupby(by=["HHID"])[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'seed_cost', 'trans_cost', 'area_planted',  'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell', 'stored_value_p_sell', 'farm_capital']].sum()

# Pass monetary variables to US2013 $
agrica[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell','stored_value_p_sell', 'farm_capital']] = agrica[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell','stored_value_p_sell', 'farm_capital']]/dollars

# Remove outliers: top 0.5%
agrica[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost','pesticides',  'total2_value_p_sell', 'farm_capital','area_planted', 'hhlabor', 'hired_labor']] = remove_outliers(agrica[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'farm_capital','area_planted', 'hhlabor', 'hired_labor']], lq=0, hq=0.995)

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



#%% AGRICULTURAL SEASON 2:


# =============================================================================
# Fertilizers & labor inputs
# =============================================================================
ag3b = pd.read_csv('agsec3b.csv', header=0, na_values='NA')
ag3b = ag3b[["HHID", 'parcelID','plotID', 'a3bq5', "a3bq8", 'a3bq15',"a3bq18",'a3bq24a','a3bq24b',"a3bq27", 'a3bq33a_1', 'a3bq33b_1', 'a3bq33c_1', 'a3bq33d_1', 'a3bq33e_1', 'a3bq35a', 'a3bq35b','a3bq35c' ,"a3bq36"]]
ag3b['hhlaborb'] = ag3b["a3bq33a_1"].fillna(0) +ag3b["a3bq33b_1"].fillna(0) +ag3b["a3bq33c_1"].fillna(0) +ag3b["a3bq33d_1"].fillna(0)+ ag3b["a3bq33e_1"].fillna(0)
ag3b['hired_laborb'] = ag3b["a3bq35a"].fillna(0) +ag3b["a3bq35b"].fillna(0)   +ag3b["a3bq35c"].fillna(0)
ag3b = ag3b[["HHID", 'plotID', "a3bq8", "a3bq18", "a3bq27",'hhlaborb', 'hired_laborb', "a3bq36"]]
ag3b.columns = ["HHID", 'plotID','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'labor_payment']


# =============================================================================
# Crop choice and Seeds costs
# =============================================================================
ag4b = pd.read_csv('agsec4b.csv', header=0, na_values='NA')
ag4b = ag4b[["HHID", 'parcelID','plotID', 'ACrop2_ID', 'cropID', 'a4bq7', 'a4bq9', "a4bq15", 'a4bq13']]
ag4b = ag4b[["HHID", 'plotID', 'cropID', 'a4bq7',  "a4bq15"]]
ag4b.columns = ["HHID", 'plotID', 'cropID', 'area_planted', 'seed_cost']
#COST


# =============================================================================
# Output
# =============================================================================
ag5b = pd.read_csv('agsec5b.csv', header=0, na_values='NA')
ag5b = ag5b[["HHID",'plotID',"cropID","a5bq6a","a5bq6c","a5bq6d","a5bq7a","a5bq7c","a5bq7d","a5bq8","a5bq10","a5bq12","a5bq13","a5bq14a","a5bq14b","a5bq15","a5bq21"]]
ag5b.columns = ["HHID", 'plotID', "cropID", "total","unit", "tokg", "sell", "unit2","tokg2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]

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
values_ag5b = ag5b[["HHID", 'plotID', 'cropID', "trans_cost"]]
#Generate values for each quantities and each type of price. Now I only use for sellings prices since the consumption ones where to big.
for q in quant:
    for p in priceslist:
        values_ag5b[q+"_value_"+p] = ag5b[q]*ag5b[p]

ag5b = values_ag5b.groupby(by=["HHID",  'plotID']).sum()

ag5b = ag5b.reset_index()


# =============================================================================
# Capital of the plots/households
# ===========================================================================
ag10 = pd.read_csv('agsec10.csv')
ag10 = ag10[['HHID','A10itemcod', 'a10q2','a10q3','a10q7','a10q8']]
ag10 = ag10[['HHID', 'a10q2']]
ag10 = ag10.groupby(by="HHID").sum()
ag10 = ag10.reset_index()
ag10.columns = ['HHID','farm_capital']

# Merge datasets -------------------------------------------
agricb = pd.merge(ag3b, ag4b, on=['HHID','plotID'], how='outer')
agricb = pd.merge(agricb, ag5b, on=['HHID','plotID'], how='right')
agricb = pd.merge(agricb, ag10, on='HHID', how='right')
agricb.set_index(['HHID','plotID'], inplace=True)
agricb = agricb.reset_index()

del ag3b, ag4b, ag5b, conversion_kg, count_bigger, count_equal, count_smaller, p, prices, prices_usd, priceslist, q, quant, values_ag5b

# Sum at hh level
agricb = agricb.groupby(by=["HHID"])[['org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor', 'seed_cost', 'trans_cost', 'area_planted',  'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell', 'stored_value_p_sell', 'farm_capital']].sum()

# Pass monetary variables to US2013 $
agricb[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell','stored_value_p_sell', 'farm_capital']] = agricb[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'sell_value_p_sell', 'cons_value_p_sell', 'gift_value_p_sell', 'food_prod_value_p_sell', 'animal_value_p_sell', 'seeds_value_p_sell','stored_value_p_sell', 'farm_capital']]/dollars

# Remove outliers: top 0.5%
agricb[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost','pesticides',  'total2_value_p_sell', 'farm_capital','area_planted', 'hhlabor', 'hired_labor']] = remove_outliers(agricb[['org_fert', 'chem_fert', 'seed_cost', 'trans_cost', 'pesticides', 'total2_value_p_sell', 'farm_capital','area_planted', 'hhlabor', 'hired_labor']], lq=0, hq=0.995)

#Aggregate variables
agricb['season'] = 2
agricb['k'] = agricb['farm_capital']
agricb['m'] = agricb['org_fert'].fillna(0)+ agricb['chem_fert'].fillna(0)+ agricb['pesticides'].fillna(0)+ agricb['seed_cost'].fillna(0)
agricb['l'] = agricb['hhlabor'].fillna(0)+ agricb['hired_labor'].fillna(0)
agricb['A'] = agricb['area_planted']
agricb['y'] = agricb['total2_value_p_sell'] - agricb['trans_cost'] 


agricb['y_over_A'] = (agricb['y']/agricb['A']).replace([-np.inf, np.inf], np.nan)
agricb['y_over_AL'] = (agricb['y_over_A']/agricb['l']).replace([-np.inf, np.inf], np.nan)

variables = ['k', 'm', 'l', 'A', 'y', 'y_over_A', 'y_over_AL']
for var in variables:
    agricb['ln'+var] = np.log(agricb[var].dropna()+np.abs(np.min(agricb[var]))).replace(-np.inf, np.nan)




#%% Both seasons together

data = agrica.append(agricb)


data.to_csv('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/data/agric_data13_hhlevel.csv', index=False)
print('data saved')


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
plt.title('Distribution of Farm Capital in Uganda 2013-2014')
plt.xlabel('Log of Farm Capital (in $)')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot hours distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnl, label="L")
plt.title('Distribution of Labour in farm in Uganda 2013-2014')
plt.xlabel('log of Farm Labour (in hours)')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Area plot distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnA, label="A")
plt.title('Distribution of Plots Area in Uganda 2013-2014')
plt.xlabel('log of Plot Area (in Acres)')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Area plot distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnm, label="m")
plt.title('Distribution of inputs in Uganda 2013-2014')
plt.xlabel('log of inputs (in US 2013 $)')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny, label="y")
plt.title('Distribution of Production in Uganda 2013-2014')
plt.xlabel('log of Agricultural Production (in US 2013 $)')
plt.ylabel("Density")
plt.legend()
plt.show()


#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny_over_A, label="y/A")
plt.title('Distribution of Production per Acre in Uganda 2013-2014')
plt.xlabel('log of Agricultural Production per Acre')
plt.ylabel("Density")
plt.legend()
plt.show()


fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny_over_A, label="y/AL")
plt.title('Distribution of Production per Acre and per Hour in Uganda 2013-2014')
plt.xlabel('log of Agricultural Production per Acre and per Hour')
plt.ylabel("Density")
plt.legend()
plt.show()

