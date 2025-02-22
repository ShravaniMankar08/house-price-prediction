# import pandas as pd
# import dash
# from dash import dcc, html
# import plotly.express as px

# # Load your dataset
# df = pd.read_csv("Housing.csv")  # Replace with actual CSV file path

# # Create a scatter plot
# fig = px.scatter(df, x="GrLivArea", y="SalePrice", title="Living Area vs House Price")

# # Show the plot
# fig.show()
# # Create a Dash app
# app = dash.Dash(__name__)

# # Sample visualization
# fig = px.scatter(df, x="GrLivArea", y="SalePrice", title="Living Area vs House Price")

# app.layout = html.Div([
#     html.H1("House Price Dashboard"),
#     dcc.Graph(figure=fig)
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True)


# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import plotly.express as px
# import pandas as pd

# # Load dataset
# df = pd.read_csv("Housing.csv")  # Replace with your actual file

# #add
# df["price_lakhs"] = df["price"] / 100000  # Convert price to lakhs


# # Initialize Dash app
# app = dash.Dash(__name__)

# # Layout of Dashboard
# app.layout = html.Div([
#     html.H1("House Price Prediction Dashboard", style={"textAlign": "center"}),
    
#     # Dropdown to select X-axis feature
#     html.Label("Select Feature to Compare with Price:"),
#     dcc.Dropdown(
#         id="feature-dropdown",
#         options=[
#             {"label": "Area", "value": "area"},
#             {"label": "Bedrooms", "value": "bedrooms"},
#             {"label": "Bathrooms", "value": "bathrooms"},
#             {"label": "Stories", "value": "stories"},
#             {"label": "Parking", "value": "parking"},
#         ],
#         value="area",
#         clearable=False
#     ),
    
#     # Scatter plot
#     dcc.Graph(id="scatter-plot"),
    
#     # Histogram of House Prices
#     html.H3("Price Distribution"),
#     dcc.Graph(id="histogram"),
# ])

# @app.callback(
#     Output("scatter-plot", "figure"),
#     Input("feature-dropdown", "value")
# )
# def update_scatter(selected_feature):
#     fig = px.scatter(df, x=selected_feature, y="price_lakhs", 
#                      title=f"{selected_feature} vs Price (₹ Lakhs)",
#                      labels={selected_feature: selected_feature, "price_lakhs": "House Price (₹ Lakhs)"},
#                      color="stories")

#     fig.update_layout(yaxis_tickprefix="₹", yaxis_title="House Price (Lakhs ₹)")
#     return fig

# @app.callback(
#     Output("histogram", "figure"),
#     Input("feature-dropdown", "value")
# )
# def update_histogram(_):
#     fig = px.histogram(df, x="price_lakhs", nbins=30, title="Distribution of House Prices (₹ Lakhs)")
#     fig.update_layout(xaxis_tickprefix="₹", xaxis_title="House Price (Lakhs ₹)")
#     return fig


# # Run the app
# if __name__ == "__main__":
#     app.run_server(debug=True)


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Housing.csv")  # Replace with your actual file

# Convert price to lakhs for better readability
df["price_lakhs"] = df["price"] / 100000

# Selecting features and target variable
features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
X = df[features]
y = df["price"]

# Train a simple Linear Regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "house_price_model.pkl")

# Load trained model
model = joblib.load("house_price_model.pkl")

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of Dashboard
app.layout = html.Div([
    html.H1("House Price Prediction Dashboard", style={"textAlign": "center"}),
    
    # User inputs for prediction
    html.Div([
        html.Label("Area (sq ft):"),
        dcc.Input(id="input-area", type="number", value=1000),

        html.Label("Bedrooms:"),
        dcc.Input(id="input-bedrooms", type="number", value=2),

        html.Label("Bathrooms:"),
        dcc.Input(id="input-bathrooms", type="number", value=1),

        html.Label("Stories:"),
        dcc.Input(id="input-stories", type="number", value=1),

        html.Label("Parking Spaces:"),
        dcc.Input(id="input-parking", type="number", value=1),

        html.Button("Predict Price", id="predict-button", n_clicks=0),
        html.H3(id="prediction-output", style={"color": "blue"}),
    ], style={"display": "flex", "flexDirection": "column", "width": "50%", "margin": "auto"}),
    
    # Dropdown to select X-axis feature
    html.Label("Select Feature to Compare with Price:"),
    dcc.Dropdown(
        id="feature-dropdown",
        options=[
            {"label": "Area", "value": "area"},
            {"label": "Bedrooms", "value": "bedrooms"},
            {"label": "Bathrooms", "value": "bathrooms"},
            {"label": "Stories", "value": "stories"},
            {"label": "Parking", "value": "parking"},
        ],
        value="area",
        clearable=False
    ),
    
    # Scatter plot
    dcc.Graph(id="scatter-plot"),
    
    # Histogram of House Prices
    html.H3("Price Distribution"),
    dcc.Graph(id="histogram"),
])

# Callback to update scatter plot
@app.callback(
    Output("scatter-plot", "figure"),
    Input("feature-dropdown", "value")
)
def update_scatter(selected_feature):
    fig = px.scatter(df, x=selected_feature, y="price_lakhs", 
                     title=f"{selected_feature} vs Price (₹ Lakhs)",
                     labels={selected_feature: selected_feature, "price_lakhs": "House Price (Lakhs ₹)"},
                     color="stories")
    fig.update_layout(yaxis_tickprefix="₹")  # Add INR symbol
    return fig

# Callback to update histogram
@app.callback(
    Output("histogram", "figure"),
    Input("feature-dropdown", "value")
)
def update_histogram(_):
    fig = px.histogram(df, x="price_lakhs", nbins=30, title="Distribution of House Prices (₹ Lakhs)")
    fig.update_layout(xaxis_tickprefix="₹")  # Add INR symbol
    return fig

# Callback for price prediction
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    Input("input-area", "value"),
    Input("input-bedrooms", "value"),
    Input("input-bathrooms", "value"),
    Input("input-stories", "value"),
    Input("input-parking", "value")
)
def predict_price(n_clicks, area, bedrooms, bathrooms, stories, parking):
    if n_clicks > 0:
        user_input = np.array([[area, bedrooms, bathrooms, stories, parking]])
        predicted_price = model.predict(user_input)[0]
        return f"Predicted Price: ₹{predicted_price:,.2f}"
    return ""

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
