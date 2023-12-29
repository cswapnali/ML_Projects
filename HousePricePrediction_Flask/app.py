from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

df = pd.read_csv('cleaned_data.csv') 

le = LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])

X = df.drop('Price', axis=1)
y = df['Price']

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = le.transform([request.form['location']])[0]
        bhk = int(request.form['bhk'])
        furnishing = int(request.form['furnishing'])
        sqft = int(request.form['sqft'])
        old_years = int(request.form['old_years'])
        floor = int(request.form['floor'])

        input_data = [[location, bhk, furnishing, sqft, old_years, floor]]
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return render_template('index.html', error="Error in input. Please check your values.")

if __name__ == '__main__':
    app.run(debug=True)
