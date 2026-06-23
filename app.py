import streamlit as st
import pandas as pd
import re
import plotly.express as px
import pickle

from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    r2_score,
    mean_absolute_error
)

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AutoML Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AutoML Dashboard")

st.markdown("""
### Automated Machine Learning Platform

Upload a CSV dataset, select the target column, and let FLAML automatically find the best machine learning model.
""")

st.write(
    "Upload any CSV dataset and automatically train a Machine Learning model."
)

# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# -----------------------------
# PROCESS DATA
# -----------------------------

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # -----------------------------
    # DATASET METRICS
    # -----------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])
    col3.metric("Missing Values", int(data.isnull().sum().sum()))

    # -----------------------------
    # TARGET COLUMN SUGGESTION
    # -----------------------------

    possible_targets = [
        "target",
        "label",
        "class",
        "survived",
        "attrition",
        "churn",
        "saleprice",
        "price",
        "gradeclass",
        "target_binary",
        "outcome",
        "diagnosis",
        "fraud",
        "is_fraud",
        "default",
        "loan_status",
        "approved",
        "result"
    ]

    suggested_target = None

    for col in data.columns:
        if col.lower() in possible_targets:
            suggested_target = col
            break

    if suggested_target:
        st.info(
            f"💡 Suggested Target Column: {suggested_target}"
        )

    target_options = [
        "-- Choose Target Column --"
    ] + list(data.columns)

    target = st.selectbox(
        "🎯 Select Target Column",
        target_options
    )

    # -----------------------------
    # WAIT UNTIL TARGET SELECTED
    # -----------------------------

    if target != "-- Choose Target Column --":

        st.success(
            f"✅ Selected Target: {target}"
        )

        # -----------------------------
        # TARGET DISTRIBUTION
        # -----------------------------

        try:
            if data[target].nunique() < 20:

                st.subheader("📈 Target Distribution")

                fig = px.histogram(
                    data,
                    x=target
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )
        except:
            pass

        # -----------------------------
        # TRAINING TIME
        # -----------------------------

        time_budget = st.slider(
            "⏱ Training Time (seconds)",
            min_value=30,
            max_value=600,
            value=60
        )

        # -----------------------------
        # TRAIN BUTTON
        # -----------------------------

        if st.button("🚀 Train Model"):

            try:

                # -----------------------------
                # FEATURES & TARGET
                # -----------------------------

                X = data.drop(columns=[target])
                y = data[target]

                # -----------------------------
                # HANDLE MISSING VALUES
                # -----------------------------

                for col in X.select_dtypes(include=["object"]).columns:
                    X[col] = X[col].fillna("Missing")

                for col in X.select_dtypes(
                    include=["int64", "float64"]
                ).columns:
                    X[col] = X[col].fillna(
                        X[col].median()
                    )

                # -----------------------------
                # ENCODE CATEGORICAL DATA
                # -----------------------------

                X = pd.get_dummies(X)

                # -----------------------------
                # CLEAN COLUMN NAMES
                # -----------------------------

                X.columns = [
                    re.sub(
                        r"[^A-Za-z0-9_]+",
                        "_",
                        str(col)
                    )
                    for col in X.columns
                ]

                # -----------------------------
                # TASK DETECTION
                # -----------------------------

                # Better task detection

                # -----------------------------
                # SMART TASK DETECTION
                # -----------------------------

                y_numeric = pd.to_numeric(
                    y.astype(str)
                     .str.replace("$", "", regex=False)
                     .str.replace("₹", "", regex=False)
                     .str.replace(",", "", regex=False),
                    errors="coerce"
                )

                if y_numeric.notna().sum() >= 0.9 * len(y):

                    y = y_numeric
                    task = "regression"

                elif pd.api.types.is_object_dtype(y):

                    task = "classification"

                else:

                    unique_ratio = y.nunique() / len(y)

                    if y.nunique() <= 20 and unique_ratio < 0.05:
                        task = "classification"
                    else:
                        task = "regression"

                st.success(
                    f"Detected Task: {task.upper()}"
                )

                # -----------------------------
                # SPLIT DATA
                # -----------------------------

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )

                # -----------------------------
                # TRAIN AUTOML
                # -----------------------------

                with st.spinner(
                    f"Training AutoML for {time_budget} seconds..."
                ):

                    automl = AutoML()

                    automl.fit(
                        X_train=X_train,
                        y_train=y_train,
                        task=task,
                        time_budget=time_budget
                    )

                # -----------------------------
                # PREDICTIONS
                # -----------------------------

                preds = automl.predict(X_test)

                # -----------------------------
                # SAVE MODEL
                # -----------------------------

                pickle.dump(
                    automl,
                    open(
                        "models/best_model.pkl",
                        "wb"
                    )
                )

                # -----------------------------
                # FEATURE IMPORTANCE
                # -----------------------------

                importance_df = None

                try:

                    model = automl.model

                    if hasattr(model, "feature_importances_"):

                        importances = model.feature_importances_

                        if len(importances) == len(X.columns):

                            importance_df = pd.DataFrame({
                                "Feature": X.columns,
                                "Importance": importances
                            })

                            importance_df = (
                                importance_df
                                .sort_values(
                                    by="Importance",
                                    ascending=False
                                )
                                .head(15)
                            )

                except Exception:
                    pass

                # -----------------------------
                # RESULTS
                # -----------------------------

                st.subheader("🏆 Results")

                st.success(
                    f"🏆 Best Model: {automl.best_estimator}"
                )

                with open(
                    "models/best_model.pkl",
                    "rb"
                ) as f:

                    st.download_button(
                        "📥 Download Model",
                        f,
                        file_name="best_model.pkl"
                    )

                if importance_df is not None:

                    st.subheader(
                        "📈 Top 15 Important Features"
                    )

                    st.bar_chart(
                        importance_df.set_index(
                            "Feature"
                        )
                    )

                # -----------------------------
                # CLASSIFICATION
                # -----------------------------

                if task == "classification":

                    acc = accuracy_score(
                        y_test,
                        preds
                    )

                    st.metric(
                        "Accuracy",
                        f"{acc*100:.2f}%"
                    )

                    st.subheader(
                        "Classification Report"
                    )

                    st.text(
                        classification_report(
                            y_test,
                            preds
                        )
                    )

                # -----------------------------
                # REGRESSION
                # -----------------------------

                else:

                    r2 = r2_score(
                        y_test,
                        preds
                    )

                    mae = mean_absolute_error(
                        y_test,
                        preds
                    )

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "R² Score",
                        round(r2, 4)
                    )

                    c2.metric(
                        "MAE",
                        round(mae, 2)
                    )

            except Exception as e:

                st.error(
                    f"Error during training: {str(e)}"
                )

    else:

        st.info(
            "👆 Please select a target column from the dropdown."
        )

st.markdown("---")
st.caption("Built using Streamlit + FLAML")