# Air Passengers Forecasting Project

Forecasts future monthly air passenger numbers using machine learning.

GitHub URL : https://github.com/harekrishna10/air-passengers-forecasting

Streamlit Web app Live Demo :

## Dataset

- `airline_passengers.csv` — 144 months (Jan 1949 to Dec 1960)
- Target: `Passengers` (monthly passenger count)

## Steps

1. Loaded the dataset and inspected it
2. Data cleaning — converted dates, set monthly frequency, checked missing values and outliers
3. Data transformation — created Month_Num, Year, Lag_1 and Lag_12 features
4. EDA — trend plot, seasonal decomposition, monthly and yearly averages
5. Model development — chronological train/test split, Linear Regression on lag features
6. Model optimization — Linear Regression (MAE 15.17) beat Random Forest (MAE 34.73)
7. Final model — Linear Regression retrained on all data, saved as `forecast_model.pkl`
8. Streamlit web app (`app.py`) with forecast plot and CSV download

## Model Results

Final model test MAE: **15.17** passengers

## How to Run

1. Install packages: `pip install -r requirements.txt`
2. Open `time_series_forecasting.ipynb` and run all cells
3. Create new Terminal
4. Start the app: `streamlit run app.py`

## How to add to GitHub

1. `git init`
2. `git add .`
3. `git commit -m "Initial commit"`
4. `git remote add origin https://github.com/harekrishna10/air-passengers-forecasting.git`
5. `git push -u origin main`
