# **🌾 Paddy Yield Prediction — Agricultural Yield Intelligence System**
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Overview**
A machine learning project that predicts **paddy crop yield (in Kg)** 
based on agronomic, soil, fertilizer, and weather data collected from 
farms across different Regions.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Problem Statement**
Agricultural productivity is influenced by multiple environmental and 
agronomic factors. Predicting yield in advance remains a major challenge 
due to the complex interaction of these variables.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **🚀 Live Demo**
👉 [Click here to try the app](#)  ← replace with your Streamlit Cloud URL
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Project Structure**
```
paddy-yield-prediction/
├── ML_Paddy_pipeline(Regression).ipynb   ← main notebook
├── app.py                                 ← Streamlit web app
├── paddy_yield_model.pkl                  ← saved best model
├── paddydataset.csv                       ← dataset
├── requirements.txt                       ← dependencies
└── README.md
```
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **ML Pipeline**
```
Data Loading → Cleaning → EDA → Outlier Handling
→ Train-Test Split → Preprocessing → Feature Selection
→ Model Training → Evaluation → Deployment
```
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Models Compared**
| Model | Tuning |
|---|---|
| Linear Regression | - |
| Ridge | alpha |
| Lasso | alpha |
| Decision Tree | max_depth, min_samples_split |
| Random Forest | n_estimators, max_depth |
| Gradient Boosting | n_estimators, learning_rate |
| KNN | n_neighbors |
| SVR | C, kernel |
| XGBoost | n_estimators, max_depth, learning_rate |

All models tuned using **GridSearchCV with 5-fold cross-validation**.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Preprocessing Pipeline**
- **Outlier Handling** — IQR clipping (fit on train only)
- **Scaling** — StandardScaler for numerical features
- **Encoding** — OneHotEncoder for categorical features
- **Feature Selection** — SelectKBest with f_regression
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Key EDA Findings**
- `Hectares` is the strongest predictor of yield
- Rainfall shows weak relationship with yield
- No feature crosses ±1 skewness threshold — StandardScaler is appropriate
- Highly correlated fertilizer features were dropped to reduce multicollinearity
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Streamlit App Features**
- Input all farm and weather details
- Predicts yield in Kg, Bags, and Tonnes
- High / Medium / Low yield classification
- Clean UI with sidebar model info
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Tech Stack**
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit, Joblib
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Run Locally**
```bash
git clone https://github.com/yourusername/paddy-yield-prediction
cd paddy-yield-prediction
pip install -r requirements.txt
streamlit run app.py
```
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
