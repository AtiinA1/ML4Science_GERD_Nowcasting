# ReadMe

## Project Overview
This report presents a new approach to nowcasting R&D expenditure. Having at our disposal GERD data, macroeconomic variables, and Google Trends, we developed models to predict R&D expenditure on a year basis and on a quarterly
basis. Best results were achieved using only constantly up-to-date Google Trends data with a MAPE (Mean Absolute Percentage Error) of approximately 2.5% for the yearly basis, and 4% for the quarterly basis.

## Key Findings
- **Yearly Predictions**: Achieved a Mean Absolute Percentage Error (MAPE) of approximately 2.5% using only  Google Trends data, highlighting its effectiveness in capturing annual R&D expenditure trends.
- **Quarterly Predictions**: Demonstrated a MAPE of around 4.2%, indicating the potential of Google Trends data in predicting more frequent, quarterly R&D expenditure fluctuations.

## Structure of the Repo
- **R&D_prediction_Macroeconomics.ipynb : contains Comparative Model Analysis**. This notebook presents a  comparison of various predictive models, including Linear Regression,Neural networks, LSTM, and KNN. It examines their performance using the last 2 previous GERD values as features, both alone and in combination with macroeconomic variables.

- **R&D_prediction_google_trends.ipynb: contains R&D predictions using Google Trends data**. This notebook exploits utilizing Google Trends data for predicting R&D expenditure. It splits the analysis into yearly and quarterly predictions, optimizing a Neural Network model tailored to these intervals. The notebook underscores the dynamic nature of Google Trends data as a valuable predictor for R&D spending.

- **google_trends_data_fetching.ipynb: notebook for Fetching and Preprocessing Google Trends Data**. Details the process of extracting relevant Google Trends data.
  
- **NN.py**: notebook containing several NN models with different depths and widths.

- **data**: This folder contains all the csv files for GERD data, macroeconomic data and Google Trends data
