import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from art import text2art
import datetime
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.dash_table import DataTable  # Correct import statement

today = datetime.date.today()

# Generate ASCII art as logo
ascii_art = text2art("BCAS")

print("============================================================")

# Print the generated ASCII art
print(ascii_art)
print("Bitcoin price prediction")
print("============================================================")
print("Created by: Muzarrif Ahamed")
print("BCAS_Kalmunai_Campus")
print("Network Engineering")
print("CSD-17 Batch")
print("Support my work:")
print("BTC,ETH & BNB (ERC20): 0xeab77fbd758df735ac79e11e95c649e9883ca10f")
print("USDT (TRC20): TRtEtci9t6SXSzqW8SyudVXvM1FeQSow65")
print("============================================================")





# Download historical data
print("Downloading historical Bitcoin data for training...")
bitcoin = yf.download('BTC-USD', start='2010-07-17', end=today)
print("Downloaded.")
print("Training...")

# Prepare data for model
bitcoin['Prediction'] = bitcoin['Close'].shift(-1)
bitcoin.dropna(inplace=True)
X = np.array(bitcoin.drop(['Prediction'], axis=1))
Y = np.array(bitcoin['Prediction'])

# Split data into training set and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Make predictions
bitcoin['Prediction'] = model.predict(np.array(bitcoin.drop(['Prediction'], axis=1)))
print("Training complete.")

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout of the dashboard
app.layout = html.Div([
    html.H1("Bitcoin Price Prediction Dashboard"),
    
    dcc.Graph(
        id='price-prediction-chart',
        figure={
            'data': [
                {'x': bitcoin.index, 'y': bitcoin['Close'], 'type': 'line', 'name': 'Actual Price'},
                {'x': bitcoin.index, 'y': bitcoin['Prediction'], 'type': 'line', 'name': 'Predicted Price'}
            ],
            'layout': {
                'title': 'Bitcoin Price Prediction vs Actual'
            }
        }
    ),

    html.Label("Select a date to see Bitcoin price:"),
    dcc.Input(id='input-date', type='text', value='', placeholder='YYYY-MM-DD'),
    html.Div(id='output-date-price'),

    html.Hr(),

    html.H3("Detailed Predictions Table:"),
    DataTable(
        id='predictions-table',
        columns=[{'name': col, 'id': col} for col in ['Close', 'Prediction']],
        data=bitcoin[['Close', 'Prediction']].tail(10).to_dict('records')
    )
])

# Define callback to update date price output
@app.callback(
    Output('output-date-price', 'children'),
    [Input('input-date', 'value')]
)
def update_date_price(input_date_str):
    try:
        input_date = pd.to_datetime(input_date_str, format='%Y-%m-%d')
        input_date_price = bitcoin.loc[input_date, 'Close']
        return f"Bitcoin price on {input_date_str}: {input_date_price}"
    except KeyError:
        return f"No data available for {input_date_str}. Please enter a valid date."
    except ValueError:
        return "Invalid date format. Please enter the date in YYYY-MM-DD format."

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
