
# Life Expectancy Prediction

This project investigates how various social, health, environmental, and economic factors influence life expectancy across countries. It leverages multiple machine learning models to predict life expectancy based on real-world data sourced from organizations like the WHO and World Bank.

---

## Overview

The main objectives of this project are:

- To build a clean, comprehensive dataset by filtering and merging publicly available tables  
- To predict life expectancy using six different supervised learning methods  
- To evaluate which features most influence life expectancy across countries  
- To visualize the results in an interactive Power BI dashboard

---

## 🗃️ Project Structure

```
life-expectancy-prediction/
│
├── data/
│   ├── raw/                            # Original CSVs from WHO, World Bank, etc.
│   └── processed/                      # Cleaned data used for modeling
│
├── dashboard/
│   └── Project.pbix                    # Final visualization dashboard
│
├── notebooks/
│   ├── export_tables.ipynb             # SQL table creation and export logic
│   ├── predictors.ipynb                # Model definitions, training, evaluation
│   └── linear_regression_func.ipynb    # Linear regression previous problem study
│
├── scripts/
│   ├── config.py                       # Config file
│   ├── ds_charts.py                    # Modular functions for charts
│   └── main.py                         # Helpers
│
└── README.md                           # This file
```

---

## Data Sources

The data used for this project comes from:

- [World Health Organization (WHO)](https://www.who.int/)
- [World Bank Open Data](https://data.worldbank.org/)
- Other public health and development databases

A total of **40+ CSVs** were considered. After evaluating completeness and relevance, **25 datasets** were selected and merged. Only columns covering **at least 100 countries across 20+ years** were included.

---

## Features Considered

Examples of selected features include:

- GDP per capita  
- Adult mortality  
- Sanitation access  
- CO2 emissions  
- BMI  
- Obesity rates  
- Literacy rate  
- Suicide and homicide statistics  
- Urban and rural population share  
- Exposure to air pollution  
- Drug and alcohol use  
- Caloric supply  

All data was transformed to match a consistent format, with **"Country", "Year", and "Life Expectancy"** as central reference fields.

---

## Models Used

| Model               | Type              | Description |
|--------------------|-------------------|-------------|
| Linear Regression   | Baseline          | Simple and interpretable |
| Ridge Regression    | Regularized       | Handles multicollinearity with L2 penalty |
| Lasso Regression    | Regularized       | Performs feature selection with L1 penalty |
| Decision Tree       | Tree-based        | Captures non-linear interactions, prone to overfitting |
| Random Forest       | Ensemble          | Combines many trees to reduce variance |
| Gradient Boosting   | Ensemble          | Boosts weak learners sequentially |
| Neural Network      | Deep learning     | MLP with several hidden layers (Keras)

---

## Results & Discussion

After training and evaluating the models, the following patterns emerged:

### Best Performers:
- **Linear Regression**, **Ridge**, and **Lasso** consistently achieved the best **R² scores**, especially for countries with steady, gradual improvement in life expectancy (e.g., Brazil).
- **R² scores exceeded 0.9** in many cases, showing strong fit when training data was complete and stable.

### Tree-Based Models:
- **Decision Tree**, **Random Forest**, and **Gradient Boosting** performed poorly on generalization.
- Their test R² values were often low or negative, especially for countries with noisy or irregular life expectancy trends.
- They were prone to **overfitting**, capturing exact training points but failing to predict unseen data correctly.

### Neural Network:
- The MLP neural net achieved strong fit but required longer training and careful tuning.  
- Results were comparable to ridge and lasso but offered no major advantage over simpler models.

---

## Dashboard

An interactive **Power BI** dashboard was created to:

- Select countries and compare model results
- Display **real vs predicted** values by year
- Highlight **R² and MSE** scores by method
- Allow country-by-country model comparison

---

## Contributors

- **Giovanna Mazzali**  
- **Bruno Oshiro**

> Note: Internal IDs and other authors have been omitted for privacy.  
> For access to the full technical report, please contact the project owner.

---

## License

This project is for academic purposes and is not licensed for commercial use.
