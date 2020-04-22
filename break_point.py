import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load the data
players = pd.read_csv("tennis_stats.csv")

# exploratory analysis
plt.scatter(players['BreakPointsOpportunities'],players['Winnings'])
plt.title('BreakPointsOpportunities vs Winnings')
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Winnings')
plt.show()
plt.clf()

#select features and value to predict
features = players[['BreakPointsOpportunities']]
winnings = players[['Winnings']]

#train test split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

#create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

#score model on test data
print("Predicting Winnings with Break points opportunities test score:", model.score(features_test,winnings_test))

#make predictions with model
winnings_prediction = model.predict(features_test)

#plot prediction
plt.scatter(winnings_test, winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs Actual Winnings')
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.show()
plt.clf()
