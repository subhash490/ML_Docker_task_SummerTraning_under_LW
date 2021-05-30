import pandas
data = pandas.read_csv('SalaryData.csv')
x = data['YearsExperience']
x=x.values.reshape(-1,1)
y = data['Salary']
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
model.predict([[1]])
import joblib
joblib.dump(model,"project1.pki")
