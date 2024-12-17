import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load resampled data
smote_data = pd.read_csv("smote_data.csv")

target_columns = ['Heart Disease', 'Kidney Disease', 'Lung Cancer']

Y = smote_data[target_columns]

X = smote_data.drop(columns=target_columns)

# Split data into train and test sets (use stratify if needed)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Initialize dictionary for models and results
gradient_boosting_models = {}
gb_results = {}

# Loop through target columns
for target in target_columns:
    if target in Y.columns:
        # Ensure target column is integer for model compatibility
        Y_train[target] = Y_train[target].astype(int)
        Y_test[target] = Y_test[target].astype(int)

        # Create and fit the Gradient Boosting model
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_model.fit(X_train, Y_train[target].values.ravel())

        # Store the trained model
        gradient_boosting_models[target] = gb_model

        # Make predictions
        Y_train_pred = gb_model.predict(X_train)
        Y_test_pred = gb_model.predict(X_test)

        # Evaluate the model
        train_accuracy = accuracy_score(Y_train[target], Y_train_pred)
        test_accuracy = accuracy_score(Y_test[target], Y_test_pred)
        precision = precision_score(Y_test[target], Y_test_pred, average='binary')
        recall = recall_score(Y_test[target], Y_test_pred, average='binary')
        f1 = f1_score(Y_test[target], Y_test_pred, average='binary')

        # Store results for display
        gb_results[target] = {
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        # Display results in Streamlit
        st.markdown(f"### {target} Results")
        st.write(f"**Train Accuracy:** {train_accuracy:.2f}")
        st.write(f"**Test Accuracy:** {test_accuracy:.2f}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write(f"**Recall:** {recall:.2f}")
        st.write(f"**F1 Score:** {f1:.2f}")

# Save all models in one file
joblib.dump(gradient_boosting_models, "all_target_models.pkl")
st.write("All models have been saved as 'all_target_models.pkl'")
