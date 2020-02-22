import pandas as pd
import quandl, datetime
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')
#Get data from quandl
df = quandl.get('WIKI/GOOGL')
#Data transformation
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'] / df['Adj. Close'] * 100)
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open'] / df['Adj. Open'] * 100)
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
#Math.ceil runder op
forecast_out = int(math.ceil(0.01*len(df)))

#Specificere label (svar) 1% af df længde ude i fremtiden
#Shift er en pandas funktion svare ca til lead/lack
df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'], 1))
#Normalisere data - hjælper med træning, men kan komplicere produktion da nyt data skal scales med det gamle
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] #Data efter det nyeste 10% (Der er ikke y værdier for disse)
X = X[:-forecast_out] # Data til og med det seneste 10% data


df.dropna(inplace=True) #De nyeste 10% kolonner vil mangle label, da den bliver beregnet i forecast_col. derfor droppes de til træning
y = np.array(df['label'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state = 0)

clf = LinearRegression(n_jobs=10)
clf.fit(X_train, y_train)
with open('linearregression-pickle', 'wb') as o:
    pickle.dump(clf, o)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_day = df.iloc[-1].name
last_unix = last_day.timestamp() #Unix timestamp
one_day = 86400 #antal sekunder i en dag
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for k in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


