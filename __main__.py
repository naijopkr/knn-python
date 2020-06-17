import pandas as pd

df = pd.read_csv('data/classified_data.csv')
df.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_predictors = df.drop('TARGET CLASS', axis=1)
scaler.fit(df_predictors)

scaled_predictors = scaler.transform(df_predictors)

X = pd.DataFrame(scaled_predictors, columns=df.columns[:-1])
y = df['TARGET CLASS']
