# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 19:45:34 2022

@author: Bruno_Oshiro
"""



import pandas as pd

def scaling(data): #drop_features sao as features numericas que nao serao normalizadas
    
    from ds_charts import get_variable_types
    variable_types = get_variable_types(data)
    numeric_vars = variable_types['Numeric']
    print(numeric_vars)
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']
    
    df_nr = data[numeric_vars]
    df_sb = data[symbolic_vars]
    df_bool = data[boolean_vars]

    from sklearn.preprocessing import StandardScaler
    from pandas import DataFrame, concat
    
    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
    tmp_1 = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
    norm_data_zscore = concat([tmp_1, df_sb,  df_bool ], axis=1)
    #norm_data_zscore.to_csv(f'{file}_scaled_zscore.csv', index=False)
    
    
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame, concat
    
    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
    norm_data_minmax = concat([tmp, df_sb,  df_bool ], axis=1)
    #norm_data_minmax.to_csv(f'data/{file}_scaled_minmax.csv', index=False)
    print(norm_data_minmax.describe())
    
    from matplotlib.pyplot import subplots, show
    
    fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
    axs[0, 0].set_title('Original data')
    data.boxplot(ax=axs[0, 0])
    axs[0, 0].tick_params(labelrotation=70)
    axs[0, 1].set_title('Z-score normalization')
    norm_data_zscore.boxplot(ax=axs[0, 1])
    axs[0, 1].tick_params(labelrotation=70)
    axs[0, 2].set_title('MinMax normalization')
    norm_data_minmax.boxplot(ax=axs[0, 2])
    axs[0, 2].tick_params(labelrotation=70)
    # show()
    
    norm_data_zscore = concat([tmp_1, df_sb,  df_bool], axis=1)
    norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1)
    
    return norm_data_zscore, norm_data_minmax


age = pd.read_csv ('Life_expectancy.csv')



# GDP
pib = pd.read_csv ('GDP.csv')

pib = pib.drop(['Country Code', 'Indicator Name', 'Indicator Code'],axis=1)

pib = pd.melt(pib, id_vars=['Country Name'],var_name='year', value_name='pib')

pib = pib.dropna()

pib['year']=pib['year'].astype(int)



# obesity
fat = pd.read_csv ('obesity.csv')

# new data frame with split value columns
obesity_column = fat["Obesity (%)"].str.split(" ", n = 1, expand = True)

fat = fat.merge(obesity_column, how = 'left',left_index=True, right_index=True)

fat.drop(['Obesity (%)', 1, 'Unnamed: 0'], axis=1, inplace=True)

fat.rename(columns={0: 'Obesity'}, inplace=True)

fat = fat[fat.Obesity != 'No']

fat['Obesity']=fat['Obesity'].astype(float)

fat = fat[fat.Sex == 'Both sexes']

fat.drop(['Sex'], axis=1, inplace=True)


# population
pop = pd.read_csv ('population.csv')

pop = pop.drop(['Country Code', 'Indicator Name', 'Indicator Code'],axis=1)

pop = pd.melt(pop, id_vars=['Country Name'],var_name='year', value_name='population')

pop = pop.dropna()

pop['year']=pop['year'].astype(int)


# rural
rural = pd.read_csv ('rural.csv')

rural = rural.drop(['Country Code', 'Indicator Name', 'Indicator Code'],axis=1)

rural = pd.melt(rural, id_vars=['Country Name'],var_name='year', value_name='rural pct')

rural = rural.dropna()

rural['year']=rural['year'].astype(int)

# misery
poor = pd.read_csv ('miseria.csv')

poor = poor.drop(['Country Code', 'Indicator Name', 'Indicator Code'],axis=1)

poor = pd.melt(poor, id_vars=['Country Name'],var_name='year', value_name='poverty')

poor = poor.dropna()

poor['year']=poor['year'].astype(int)


# electry
ele = pd.read_csv ('electry.csv')

ele = ele.drop(['Country Code', 'Indicator Name', 'Indicator Code'],axis=1)

ele = pd.melt(ele, id_vars=['Country Name'],var_name='year', value_name='electricity')

ele = ele.dropna()

ele['year']=ele['year'].astype(int)




#sanitation
sani = pd.read_csv('sanitation.csv')
sani.drop(['Country Code', 'Indicator Name','Indicator Code','Unnamed: 65'], axis=1, inplace=True)
sani = pd.melt(sani, id_vars=['Country Name'], var_name = ['year'], value_name = 'sanitation')

sani = sani.dropna()
sani['year']=sani['year'].astype(int)
#sani.rename(columns={'san_sm': 'safe sanitation','san_bas_minus_sm':'basic sanitation','san_od':'open defecation'}, inplace=True)
#sani.drop(['san_lim', 'san_unimp','Code'],axis=1,inplace=True)


# final df
df = age.merge(pib, how='left', right_on= ['Country Name', 'year'], left_on=['Entity', 'Year'])
df.drop(['Country Name','year'],axis=1,inplace=True)
df = df.merge(fat, how='left', right_on= ['Country', 'Year'], left_on=['Entity', 'Year'])
df.drop(['Country'],axis=1,inplace=True)
df = df.merge(pop, how='left', right_on= ['Country Name', 'year'], left_on=['Entity', 'Year'])
df.drop(['Country Name','year'],axis=1,inplace=True)
df = df.merge(rural, how='left', right_on= ['Country Name', 'year'], left_on=['Entity', 'Year'])
df.drop(['Country Name','year'],axis=1,inplace=True)
#df = df.merge(poor, how='left', right_on= ['Country Name', 'year'], left_on=['Entity', 'Year'])
#4df.drop(['Country Name','year'],axis=1,inplace=True)
df = df.merge(ele, how='left', right_on= ['Country Name', 'year'], left_on=['Entity', 'Year'])
df.drop(['Country Name','year'],axis=1,inplace=True)
#df = df.merge(sani, how='left', right_on= ['Country Name', 'year'], left_on=['Entity', 'Year'])
#df.drop(['Country Name','year'],axis=1,inplace=True)

df.rename(columns={'Entity': 'Country'}, inplace=True)
df.dropna(inplace=True)



#selecting country
br = df[df.Country == 'Australia']



df = br






from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

## Import pearsonr function from scipy -- calculate the correlation and p-value between two columns
from scipy.stats import pearsonr



plt.scatter(br['Year'],br['Life expectancy'])
plt.show() #



## Calculate the correlations between the columns
df_corrs = df.corr(method='pearson')

## Calculate the p-value, as the second element of the response from the pearsonr function. 
pval = df.corr(method=lambda x, y: pearsonr(x, y)[1])

## Establish the mask, to hide values without a given statistical significance
ptg_stat_sig = 0.1/100
mask = pval > ptg_stat_sig

## Plot the correlation matrix using seaborn's heatmap function
plt.subplots(figsize=(15, 15))
heatmap = sns.heatmap(df_corrs, mask = mask, square = True, cmap = 'coolwarm', annot = True)




# Prepare the input data. Manually create the interaction elements' columns
X_df = df.drop(columns = ['Life expectancy','Country','Year'])

X_df, df_minmax = scaling(X_df)

X = X_df.values

y = df['Life expectancy'].values.reshape(-1, 1)

# Prepare the column names to be printed
names = list(X_df.columns.values)
names.insert(0,'Intercept')

# Fit the linear regression model the data
regr = LinearRegression()
regr.fit(X, y)

# Extract the coefficients and print them
values = list(regr.coef_[0])
values.insert(0, regr.intercept_[0])
s = ['{} :: {:.4f}'.format(names[i].rjust(18), values[i]) for i in range(len(values))]
listToStr = '\n'.join([str(elem) for elem in s]) 
print(listToStr)

# Calculate and print the R^2 score
sales_pred = regr.predict(X)
r2_value = r2_score(y, sales_pred)
rmse = mean_squared_error(y, sales_pred)
print('\n{} :: {:.4f}'.format('R^2'.rjust(18), r2_value))
print('\n{} :: {:.4f}'.format('RMSE'.rjust(18), rmse))

plt.figure
plt.plot(br['Year'],sales_pred)
plt.show()




'''
from numpy import argsort, arange
from ds_charts import horizontal_bar_chart
from matplotlib.pyplot import Axes
import matplotlib.pyplot as plt

variables = X.columns

importances = regr.feature_importances_
indices = argsort(importances)[::-1]
elems = []
imp_values = []
for f in range(len(variables)-1):
    elems += [variables[indices[f]]]
    imp_values += [importances[indices[f]]]
    print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')


#horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance', ylabel='variables')
plt.figure(figsize=(4,4))
plt.barh(elems, imp_values, align='center', height=0.3, color="skyblue")
plt.xlabel('Importance')
plt.ylabel('Variables')
plt.title('Decision Tree Features importance')

plt.tight_layout()
'''



