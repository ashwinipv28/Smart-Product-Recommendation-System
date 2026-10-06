import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

st.title("🛒 Smart Product Recommendation System")
st.write(
    "Enter a Customer ID to get personalized Top 5 product recommendations."
)


# Load & preprocess data
@st.cache_data
def load_and_train():
  np.random.seed(42)
  num_samples = 1000
  data = {
      "customer_id": np.random.randint(1001, 1050, size=num_samples),
      "product_id": np.random.randint(5001, 5020, size=num_samples),
      "product_category": np.random.choice(
          ["Electronics", "Clothing", "Home", "Books"], size=num_samples
      ),
      "product_price": np.random.uniform(10, 500, size=num_samples).round(2),
      "views": np.random.randint(1, 20, size=num_samples),
      "cart_adds": np.random.randint(0, 10, size=num_samples),
      "purchased": np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3]),
  }
  df = pd.DataFrame(data)
  df = df.drop_duplicates()
  df["product_price"] = df["product_price"].fillna(df["product_price"].median())
  df["cart_to_view_ratio"] = df["cart_adds"] / (df["views"] + 1e-5)

  customer_prev = df.groupby("customer_id")["purchased"].sum().reset_index()
  customer_prev.columns = ["customer_id", "previous_purchases"]
  df = df.merge(customer_prev, on="customer_id", how="left")

  le_category = LabelEncoder()
  df["category_encoded"] = le_category.fit_transform(df["product_category"])

  features = [
      "product_price",
      "views",
      "cart_adds",
      "cart_to_view_ratio",
      "previous_purchases",
      "category_encoded",
  ]
  X, y = df[features], df["purchased"]

  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X, y)

  return df, model, le_category, features


df, model, le_category, features = load_and_train()

# UI Inputs
customer_id_input = st.number_input(
    "Enter Customer ID:", min_value=1001, max_value=1050, value=1005, step=1
)

if st.button("Generate Recommendations"):
  customer_history = df[df["customer_id"] == customer_id_input]

  if not customer_history.empty:
    avg_views = customer_history["views"].mean()
    avg_cart_adds = customer_history["cart_adds"].mean()
    prev_purchases = customer_history["previous_purchases"].iloc[0]
  else:
    avg_views = df["views"].median()
    avg_cart_adds = df["cart_adds"].median()
    prev_purchases = 0

  product_catalog = (
      df[["product_id", "product_category", "product_price"]]
      .drop_duplicates()
      .reset_index(drop=True)
  )
  candidate_products = product_catalog.copy()
  candidate_products["views"] = avg_views
  candidate_products["cart_adds"] = avg_cart_adds
  candidate_products["cart_to_view_ratio"] = avg_cart_adds / (avg_views + 1e-5)
  candidate_products["previous_purchases"] = prev_purchases
  candidate_products["category_encoded"] = le_category.transform(
      candidate_products["product_category"]
  )

  X_pred = candidate_products[features]
  candidate_products["purchase_probability"] = model.predict_proba(X_pred)[
      :, 1
  ]

  top_5 = candidate_products.sort_values(
      by="purchase_probability", ascending=False
  ).head(5)

  st.subheader(
      f"Top 5 Recommendations for Customer ID: {int(customer_id_input)}"
  )
  st.dataframe(
      top_5[[
          "product_id",
          "product_category",
          "product_price",
          "purchase_probability",
      ]]
  )
