# 🛒 Smart Product Recommendation System

The **Smart Product Recommendation System** is a Machine Learning solution that cleans customer interaction data, engineers intent features like cart-to-view ratios, and trains a Random Forest model. It accepts a Customer ID to predict purchase probabilities and outputs top 5 personalized recommendations.

---

## 📌 Project Overview

Online retail platforms face the challenge of showing relevant products to diverse customers. This project processes historical interaction telemetry—including product views, cart additions, item categories, and past purchase counts—to predict the likelihood of a customer buying specific items. Using a **Random Forest Classifier**, the system dynamically ranks candidate products and generates the **Top 5 personalized recommendations** for any specified Customer ID.

---

## ✨ Key Features

* **Automated Data Preprocessing:** Removes duplicate interaction records and handles missing numerical values through median imputation.
* **Behavioral Feature Engineering:** Computes interaction metrics such as `cart_to_view_ratio` and aggregates historical customer purchasing frequency (`previous_purchases`).
* **Probabilistic Classification:** Utilizes a Random Forest Classifier to estimate purchase probabilities (0 to 100%) for customer-product pairs.
* **Personalized Recommendation Engine:** Tailors recommendations dynamically based on individual customer browsing and purchase histories.
* **Cold-Start Handling:** Integrates fallback dataset baseline metrics to serve reliable recommendations to new customers without historical data.
* **Interactive Web App:** Deployed via Streamlit for instant interaction and real-time recommendation generation.

---

## 🛠️ Technologies Used

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `LabelEncoder`, evaluation metrics)
* **Web Framework:** Streamlit
* **Version Control & Hosting:** Git, GitHub, Streamlit Cloud

---

## 📁 Repository Structure

```text
├── notebook.ipynb       # Jupyter Notebook containing data prep, EDA, model training & evaluation
├── app.py              # Streamlit web application script for interactive deployment
├── requirements.txt    # List of dependencies required for cloud deployment
└── README.md           # Project documentation
