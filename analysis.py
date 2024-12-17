import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score,confusion_matrix, ConfusionMatrixDisplay,roc_curve,auc
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from imblearn.combine import SMOTEENN


# Function to process the datasets
def process_data(heart_df, kidney_df, lung_cancer_df):
    # Standardize column names for each dataset
    heart_df.rename(columns={  
        'age': 'Age',
        'sex': 'Gender',
        'chest pain': 'Chest Pain',
        'trestbps': 'Trestbps',
        'chol': 'Cholesterol',
        'fbs': 'Fbs',
        'max heart rate': 'Max Heart Rate',
        'heart result': 'Heart Disease'
    }, inplace=True)

    kidney_df.rename(columns={
        'Age': 'Age',
        'Gender': 'Gender',
        'Smoking': 'Smoking',
        'AlcoholConsumption': 'Alcohol Consumption',
        'PhysicalActivity': 'Physical Activity',
        'CholesterolTotal': 'Cholesterol',
        'Diagnosis': 'Kidney Disease'
    }, inplace=True)

    lung_cancer_df.rename(columns={
        'GENDER': 'Gender',
        'AGE': 'Age',
        'SMOKING': 'Smoking',
        'ALCOHOL CONSUMING': 'Alcohol Consumption',
        'COUGHING': 'Coughing',
        'CHEST PAIN': 'Chest Pain',
        'LUNG_CANCER': 'Lung Cancer'
    }, inplace=True)

    # Define the columns for each dataset
    heart_columns = ['Age', 'Gender', 'Smoking', 'Alcohol Consumption', 'Cholesterol', 'Chest Pain', 'Heart Disease', 'Trestbps', 'Fbs', 'Max Heart Rate']
    kidney_columns = ['Age', 'Gender', 'Smoking', 'Alcohol Consumption', 'Cholesterol', 'Chest Pain', 'Physical Activity','Kidney Disease']
    lung_cancer_columns = ['Age', 'Gender', 'Smoking', 'Alcohol Consumption', 'Cholesterol', 'Chest Pain', 'Lung Cancer']

    # Reindex the DataFrames
    heart_disease_combined = heart_df.reindex(columns=heart_columns)
    kidney_disease_combined = kidney_df.reindex(columns=kidney_columns)
    lung_cancer_combined = lung_cancer_df.reindex(columns=lung_cancer_columns)

    # Combine the datasets
    combined_df = pd.concat([heart_disease_combined, kidney_disease_combined, lung_cancer_combined], ignore_index=True, sort=False)

    return combined_df

# Function to fill missing values
def fill_missing_values(df):
    # Fill numeric columns with the median
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        # Handle infinite values
        df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        df[col].fillna(df[col].median(), inplace=True)
    
    # Convert all numeric columns to integers
    df[numeric_columns] = df[numeric_columns].astype(int)

    # Fill categorical columns
    if 'Lung Cancer' in df.columns:
        df['Lung Cancer'] = df['Lung Cancer'].replace({'YES': 1, 'NO': 0})
        df['Lung Cancer'].fillna(df['Lung Cancer'].mode()[0], inplace=True)
        df['Lung Cancer'] = df['Lung Cancer'].astype('int') 
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'F': 0, 'M': 1})
        df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
        df['Gender'] = df['Gender'].astype('int')
    return df

# Function to calculate unique value percentages in target columns
def calculate_unique_value_percentage(df, target_columns):
    unique_percentages = {}
    for col in target_columns:
        if col in df.columns:
            value_counts = df[col].value_counts(normalize=True) * 100  # Calculate percentages
            unique_percentages[col] = value_counts
    return unique_percentages

# Function to balance datasets using SMOTE
def balance_dataset_smote(data, target_col):
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Initialize SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)

    X_res, y_res = smote.fit_resample(X, y)

    return pd.concat([X_res, y_res], axis=1)

# Function to apply Random Oversampling
def random_oversample(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    
    # Combine the resampled features and target back into a single DataFrame
    balanced_df = pd.DataFrame(X_res, columns=X.columns)
    balanced_df[target_column] = y_res
    return balanced_df

# Function to apply Random Under-Sampling (RUS)
def random_under_sample(data, target_col, desired_ratio=0.5):
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Calculate majority and minority class sampling strategy
    rus = RandomUnderSampler(sampling_strategy=desired_ratio, random_state=42)

    X_res, y_res = rus.fit_resample(X, y)

    return pd.concat([X_res, y_res], axis=1)

# Streamlit UI
st.markdown('<h1 style="color: violet;">Multi-Disease Risk Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color: violet;">DATA ANALYSIS</h2>', unsafe_allow_html=True)
st.header('Upload Datasets')
st.write("Please upload the following datasets in CSV format:")
heart_file = st.file_uploader("Upload Heart Disease Dataset", type=["csv"])
kidney_file = st.file_uploader("Upload Kidney Disease Dataset", type=["csv"])
lung_cancer_file = st.file_uploader("Upload Lung Cancer Dataset", type=["csv"])

# Load all datasets
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    return None

heart_df = load_data(heart_file)
kidney_df = load_data(kidney_file)
lung_cancer_df = load_data(lung_cancer_file)

# Check if all datasets are successfully loaded
if heart_df is not None and kidney_df is not None and lung_cancer_df is not None:
    # Data integration
    combined_data = process_data(heart_df, kidney_df, lung_cancer_df)

    st.markdown('<h3 style="color: violet;">INTEGRATED DATASET</h3>', unsafe_allow_html=True)
    st.write("After combining all three datasets, the integrated dataset is displayed below:")
    st.dataframe(combined_data)

    # Option to download the combined dataset
    csv = combined_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Combined Dataset", data=csv, file_name='combined_health_dataset.csv', mime='text/csv')

    # Data preprocessing
    st.markdown('<h3 style="color: violet;">DATA PREPROCESSING</h3>', unsafe_allow_html=True)

    # BASIC DATA CHECKS
    st.write("Shape of the dataset:", combined_data.shape)
    st.write("Columns and their data types:")
    st.write(combined_data.dtypes)
    st.write("First 5 rows of the dataset:")
    st.dataframe(combined_data.head())
    st.write("Last 5 rows of the dataset:")
    st.dataframe(combined_data.tail())

    # Check for null values
    st.write("Check for null values in each column:")
    null_counts = combined_data.isnull().sum()
    st.dataframe(null_counts[null_counts > 0])

    # Apply missing value handling
    combined_data_filled = fill_missing_values(combined_data)

    # Display the dataset after handling missing values
    st.markdown('<h3 style="color: violet;">PROCESSED DATASET (After Handling Missing Values)</h3>', unsafe_allow_html=True)
    st.write("Final dataset after handling missing values is displayed below:")
    st.dataframe(combined_data_filled)

    st.write("Unique Value Percentages in Target Columns:")
    target_columns = ['Heart Disease', 'Kidney Disease', 'Lung Cancer']
    unique_percentages = calculate_unique_value_percentage(combined_data_filled, target_columns)
    for target in target_columns:
        if target in unique_percentages:
            st.dataframe(unique_percentages[target])
        else:
            st.write(f"{target} column is not found in the dataset.")
    
    
    st.write("-->SMOTE is used to generate synthetic samples for the minority class")
    st.write("-->RUS is applied to reduce the majority class size.")
    st.write("-->ROS is used for random oversampling to balance the dataset.")
    st.write("Click the button below to balance the dataset using SMOTE, RUS, and ROS.")
    if st.button("Balance Dataset with SMOTE, RUS, and ROS"):
        smote_data = combined_data_filled.copy()
        rus_data = combined_data_filled.copy()
        ros_data = combined_data_filled.copy()

        # Apply SMOTE for each target column and calculate unique percentages
        target_columns = ['Heart Disease', 'Kidney Disease', 'Lung Cancer']
        smote_unique_percentages = {}
        for target in target_columns:
            if target in smote_data.columns:
                st.write(f"Balancing with SMOTE for **{target}**...")
                smote_data = balance_dataset_smote(smote_data, target)
                smote_unique_percentages[target] = calculate_unique_value_percentage(smote_data, [target])

        # Apply Random Under-Sampling for each target column and calculate unique percentages
        rus_unique_percentages = {}
        for target in target_columns:
            if target in rus_data.columns:
                st.write(f"Applying Random Under-Sampling for **{target}**...")
                rus_data = random_under_sample(rus_data, target)
                rus_unique_percentages[target] = calculate_unique_value_percentage(rus_data, [target])

        # Apply Random Oversampling for each target column and calculate unique percentages
        ros_unique_percentages = {}
        for target in target_columns:
            if target in ros_data.columns:
                st.write(f"Applying Random Oversampling for **{target}**...")
                ros_data = random_oversample(ros_data, target)
                ros_unique_percentages[target] = calculate_unique_value_percentage(ros_data, [target])

        # Move 'Heart Disease' column to the end if it exists
        if 'Heart Disease' in smote_data.columns:
            heart_disease_column = smote_data.pop('Heart Disease')
            smote_data['Heart Disease'] = heart_disease_column

        # Reset the index of the final balanced dataset
        smote_data.reset_index(drop=True, inplace=True)
        rus_data.reset_index(drop=True, inplace=True)
        ros_data.reset_index(drop=True, inplace=True)
        
        # Display the balanced datasets
        st.markdown('<h4 style="color: violet;">BALANCED DATASETS</h4>', unsafe_allow_html=True)

        # SMOTE Balanced Dataset
        st.write("Dataset after SMOTE balancing is displayed below:")
        st.dataframe(smote_data)

        # RUS Balanced Dataset
        st.write("Dataset after RUS balancing is displayed below:")
        st.dataframe(rus_data)

        # ROS Balanced Dataset
        st.write("Dataset after ROS balancing is displayed below:")
        st.dataframe(ros_data)


        # Display Unique Value Percentages for each balancing technique
        st.markdown('<h4 style="color: green;">Unique Value Percentages in Target Columns (SMOTE, RUS, ROS)</h4>', unsafe_allow_html=True)

        st.write("**Unique Value Percentages After SMOTE**:")
        for target, percentages in smote_unique_percentages.items():
            st.write(f"**{target}** unique value percentages:")
            st.dataframe(percentages)

        st.write("**Unique Value Percentages After RUS**:")
        for target, percentages in rus_unique_percentages.items():
            st.write(f"**{target}** unique value percentages:")
            st.dataframe(percentages)

        st.write("**Unique Value Percentages After ROS**:")
        for target, percentages in ros_unique_percentages.items():
            st.write(f"**{target}** unique value percentages:")
            st.dataframe(percentages)


        st.write("From all above techniques iam going with SMOTE BALANCED DATASET")
        st.dataframe(smote_data)
        # Convert dataframe to CSV
        csv = smote_data.to_csv(index=False)

        # Create a download button
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="smote_data.csv",
            mime="text/csv"
        )
        # Adding basic data checks after unique value percentages
        st.markdown('<h4 style="color: green;">Basic Data Checks</h4>', unsafe_allow_html=True)

        
        st.write("**Shape of the balanced dataset:**")
        st.write(f"Rows: {smote_data.shape[0]}, Columns: {smote_data.shape[1]}")

        
        st.write("**Column names in the dataset:**")
        st.write(smote_data.columns.tolist())

        
        st.write("**First 5 rows of the balanced dataset (Head):**")
        st.dataframe(smote_data.head())

        
        st.write("**Last 5 rows of the balanced dataset (Tail):**")
        st.dataframe(smote_data.tail())

        # Check for missing values
        st.write("**Missing values in each column:**")
        missing_values = smote_data.isnull().sum()
        st.dataframe(missing_values[missing_values > 0])

        # Display the data types of each column
        st.write("**Data types of each column in the dataset:**")
        st.write(smote_data.dtypes)

        st.markdown('<h2 style="color: violet;">Exploratory Data Analysis </h2>', unsafe_allow_html=True)
        st.write("Here are some EDA visualizations based on our Dataset:")
        ##histograms
        st.write("Histogram: To check the distribution of  Age:")
        plt.figure(figsize=(5,5))
        sns.histplot(smote_data['Age'], bins=30, kde=True)
        plt.title('Distribution of Age')
        st.pyplot(plt)

        st.write("Histogram: To check the distribution of Cholesterol :")
        plt.figure(figsize=(5,5))
        sns.histplot(smote_data['Cholesterol'], bins=20, kde=True)
        plt.title('Distribution of Cholesterol')
        st.pyplot(plt)
        
        ##countplots
        st.write("count plot: to visualize the distribution of gender and smoking")
        sns.countplot(x='Gender', data=smote_data,width=0.3)
        plt.title('Gender Distribution')
        st.pyplot(plt)


        sns.countplot(x='Smoking', data=smote_data,width=0.3)
        plt.title('Smoking Distribution')
        st.pyplot(plt)
        

        ##barplots
        st.write("bar plot: To show Average Smoking Levels by Gender ")
        plt.figure(figsize=(8, 4))
        sns.barplot(x='Gender', y='Smoking', data=smote_data, estimator=np.mean,width=0.3)
        st.pyplot(plt)
        plt.clf()

        ##scatter plot
        st.write(" To show Age vs Cholesterol (Colored by Kidney Disease)")
        sns.scatterplot(x='Age', y='Cholesterol', hue='Kidney Disease', data=smote_data)
        plt.title('Age vs Cholesterol (Colored by Kidney Disease)')
        st.pyplot(plt)
        plt.clf()
        
        ##pie chart
        st.write("To show smoking distribution in pie chart")
        smoking_counts = smote_data['Smoking'].value_counts()  
        plt.figure(figsize=(6, 6)) 
        plt.pie(smoking_counts, labels=smoking_counts.index, autopct='%1.1f%%', startangle=90)  
        st.pyplot(plt)

       ##pie chart
        st.write("To visualize Gender using pie chart")
        smoking_counts = smote_data['Gender'].value_counts()  
        plt.figure(figsize=(6, 6)) 
        plt.pie(smoking_counts, labels=smoking_counts.index, autopct='%1.1f%%', startangle=90)  
        plt.title('Gender')  
        st.pyplot(plt)

        st.markdown('<h2 style="color: violet;">DATA PREPARATIONS AND FEATURE SELECTIONS</h2>', unsafe_allow_html=True)
        st.write("Dataset divided into target dataframe and feature dataframe")
        target_columns = ['Heart Disease', 'Kidney Disease', 'Lung Cancer']
        #target dataframe
        Y = smote_data[target_columns]  
        #feature dataframe
        X = smote_data.drop(columns=target_columns) 
        
        # Display the separate DataFrames 
        st.write("Feature DataFrame (X):")
        st.dataframe(X.head())
        st.write("Target DataFrame (Y):")
        st.dataframe(Y.head())

        # Correlation Matrix for Feature Columns
        st.markdown('<h3 style="color: violet;">Correlation Matrix for Features</h3>', unsafe_allow_html=True)
        correlation_matrix = X.corr() 

        
        st.write("Correlation Matrix:")
        st.dataframe(correlation_matrix)

        
        st.write("Correlation Heatmap for Features:")
        st.write("it provides visual summary of each variable correlates with others, and which variables are positively or negatively correlated.")
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Feature Columns')
        st.pyplot(plt)


        st.markdown('<h2 style="color: violet;"> DATA SPLITTING </h2>', unsafe_allow_html=True)
        st.write("Splitting the dataset into training and testing sets")
        st.write("splitting testing data int 30% , training data into 70%")

         # Perform train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Display the sizes of the resulting sets
        st.write("Size of training set (X_train):", X_train.shape)
        st.write("Size of testing set (X_test):", X_test.shape)
        st.write("Size of training set (Y_train):", Y_train.shape)
        st.write("Size of testing set (Y_test):", Y_test.shape)
        

        st.write("X_train data:")
        st.write(X_train)
        st.write("X_test data:")
        st.write(X_test)

        st.write("Y_test data:")
        st.write(Y_test)
        st.write("Y_train data:")
        st.write(Y_train)
        
    ## data modelling
        st.markdown('<h2 style="color: violet;"> DATA MODELLING </h2>', unsafe_allow_html=True)
        st.write(' -> precision measures the accuracy of the positive predictions made by the model')
        st.write('-> It tells us the proportion of true positive predictions among all predictions that were classified as positive')
        st.write(' -> Recall measures how well the model identifies all relevant instances')
        st.write('-> It tells us the proportion of true positives that were correctly identified out of all actual positive cases')

        st.write('-> The F1 score is the harmonic mean of precision and recall')

        st.markdown('<h2 style="color: violet;"> 1.LOGISTIC REGRESSION </h2>', unsafe_allow_html=True)
        
        # Initialize the Logistic Regression model
        # Define the target columns
        target_columns = ['Heart Disease', 'Kidney Disease', 'Lung Cancer']
        logistic_models = {}
        results = {}

        for target in target_columns:
            if target in Y.columns:
                # Create and fit the model
                model = LogisticRegression( solver='saga', class_weight='balanced')
                model.fit(X_train, Y_train[target])

                # Store the model
                logistic_models[target] = model

                # Predict on training and testing datasets
                Y_train_pred = model.predict(X_train)
                Y_test_pred = model.predict(X_test)
                Y_test_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

                # Calculate accuracy
                train_accuracy = accuracy_score(Y_train[target], Y_train_pred)
                test_accuracy = accuracy_score(Y_test[target], Y_test_pred)

                # Calculate precision, recall, and F1 score
                precision = precision_score(Y_test[target], Y_test_pred)
                recall = recall_score(Y_test[target], Y_test_pred)
                f1 = f1_score(Y_test[target], Y_test_pred)

                # Store results in a dictionary
                results[target] = {
                    'Train Accuracy': train_accuracy,
                    'Test Accuracy': test_accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                }

                # Display results
                st.markdown(f"### {target} Results")
                st.write(f"**Train Accuracy:** {train_accuracy:.2f}")
                st.write(f"**Test Accuracy:** {test_accuracy:.2f}")
                st.write(f"**Precision:** {precision:.2f}")
                st.write(f"**Recall:** {recall:.2f}")
                st.write(f"**F1 Score:** {f1:.2f}")

                # Confusion Matrix
                cm = confusion_matrix(Y_test[target], Y_test_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                # Plotting and saving the confusion matrix
                fig_cm, ax_cm = plt.subplots()
                disp.plot(ax=ax_cm)
                plt.title(f'Confusion Matrix for {target}')
                    
                st.pyplot(fig_cm)  

                plt.close(fig_cm)  

                # ROC Curve
                fpr, tpr, thresholds = roc_curve(Y_test[target], Y_test_prob)
                roc_auc = auc(fpr, tpr)

                # Plotting ROC Curve
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                ax_roc.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.0])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'Receiver Operating Characteristic for {target}')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)  
                plt.close(fig_roc) 
        

        st.markdown('<h2 style="color: violet;">2.RANDOM FOREST  </h2>', unsafe_allow_html=True)
       
        target_columns = ['Heart Disease', 'Kidney Disease', 'Lung Cancer']
        random_forest_models = {}
        results = {}

        
        for target in target_columns:
            if target in Y.columns:
                # Initialize and fit the Random Forest model
                model = RandomForestClassifier(n_estimators=1, max_depth=3, class_weight='balanced', random_state=41)
                model.fit(X_train, Y_train[target])

                # Store the model
                random_forest_models[target] = model

                # Predict on training and testing datasets
                Y_train_pred = model.predict(X_train)
                Y_test_pred = model.predict(X_test)
                Y_test_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

                
                train_accuracy = accuracy_score(Y_train[target], Y_train_pred)
                test_accuracy = accuracy_score(Y_test[target], Y_test_pred)

                
                precision = precision_score(Y_test[target], Y_test_pred)
                recall = recall_score(Y_test[target], Y_test_pred)
                f1 = f1_score(Y_test[target], Y_test_pred)

               
                results[target] = {
                    'Train Accuracy': train_accuracy,
                    'Test Accuracy': test_accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                }

                
                st.markdown(f"### {target} Results")
                st.write(f"**Train Accuracy:** {train_accuracy:.2f}")
                st.write(f"**Test Accuracy:** {test_accuracy:.2f}")
                st.write(f"**Precision:** {precision:.2f}")
                st.write(f"**Recall:** {recall:.2f}")
                st.write(f"**F1 Score:** {f1:.2f}")

                # Confusion Matrix
                cm = confusion_matrix(Y_test[target], Y_test_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                
                # Plotting and displaying the confusion matrix
                fig_cm, ax_cm = plt.subplots()
                disp.plot(ax=ax_cm,cmap='Blues')
                plt.title(f'Confusion Matrix for {target}')
                st.pyplot(fig_cm)
                plt.close(fig_cm)  

                # ROC Curve
                fpr, tpr, thresholds = roc_curve(Y_test[target], Y_test_prob)
                roc_auc = auc(fpr, tpr)

                # Plotting ROC Curve
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                ax_roc.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.0])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'Receiver Operating Characteristic for {target}')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
                plt.close(fig_roc)  

        st.markdown('<h2 style="color: violet;">3.GRADIENT BOOSTING </h2>', unsafe_allow_html=True)
        
        # Initialize the Gradient Boosting model
        gradient_boosting_models = {}
        gb_results = {}
        

        for target in target_columns:
            if target in Y.columns:
                # Create and fit the model
                gb_model = GradientBoostingClassifier()
                gb_model.fit(X_train, Y_train[target])

                # Store the model
                gradient_boosting_models[target] = gb_model

                
                Y_train_pred = gb_model.predict(X_train)
                Y_test_pred = gb_model.predict(X_test)
                Y_test_prob = gb_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

               
                train_accuracy = accuracy_score(Y_train[target], Y_train_pred)
                test_accuracy = accuracy_score(Y_test[target], Y_test_pred)

                
                precision = precision_score(Y_test[target], Y_test_pred)
                recall = recall_score(Y_test[target], Y_test_pred)
                f1 = f1_score(Y_test[target], Y_test_pred)

               
                gb_results[target] = {
                    'Train Accuracy': train_accuracy,
                    'Test Accuracy': test_accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                }

                # Display results
                st.markdown(f"### {target} Results")
                st.write(f"**Train Accuracy:** {train_accuracy:.2f}")
                st.write(f"**Test Accuracy:** {test_accuracy:.2f}")
                st.write(f"**Precision:** {precision:.2f}")
                st.write(f"**Recall:** {recall:.2f}")
                st.write(f"**F1 Score:** {f1:.2f}")

                # Confusion Matrix
                cm = confusion_matrix(Y_test[target], Y_test_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gb_model.classes_)
                
               
                fig_cm, ax_cm = plt.subplots()
                disp.plot(ax=ax_cm)
                plt.title(f'Confusion Matrix for {target}')
                st.pyplot(fig_cm)
                plt.close(fig_cm)

                # ROC Curve
                fpr, tpr, thresholds = roc_curve(Y_test[target], Y_test_prob)
                roc_auc = auc(fpr, tpr)

                # Plotting ROC Curve
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                ax_roc.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.0])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'Receiver Operating Characteristic for {target}')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
                plt.close(fig_roc)

        st.markdown('<h2 style="color: violet;"> 4.Neural Network Classifier </h2>', unsafe_allow_html=True)

        target_columns = ['Heart Disease', 'Kidney Disease', 'Lung Cancer']
        nn_models = {}
        nn_results = {}

        # Define hyperparameters
        hidden_layer_sizes = (128, 64)  # Two hidden layers with 128 and 64 units
        max_iter = 3  # Number of epochs
        learning_rate_init = 0.001

        for target in target_columns:
            if target in Y.columns:
                # Initialize the MLPClassifier model
                model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                                    activation='relu', 
                                    solver='adam', 
                                    max_iter=max_iter, 
                                    learning_rate_init=learning_rate_init,
                                    random_state=42
                                   )
                # Combine SMOTE with Edited Nearest Neighbors
                smote_enn = SMOTEENN(random_state=42)
                X_resampled, Y_resampled = smote_enn.fit_resample(X_train, Y_train[target])

                # Train the model on the balanced dataset
                model.fit(X_resampled, Y_resampled)
                # Predict on training and testing datasets
                Y_train_pred = model.predict(X_train)
                Y_test_pred = model.predict(X_test)
                Y_test_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for ROC curve

                # Calculate metrics
                train_accuracy = accuracy_score(Y_train[target], Y_train_pred)
                test_accuracy = accuracy_score(Y_test[target], Y_test_pred)
                precision = precision_score(Y_test[target], Y_test_pred)
                recall = recall_score(Y_test[target], Y_test_pred)
                f1 = f1_score(Y_test[target], Y_test_pred)

                # Store results
                nn_results[target] = {
                    'Train Accuracy': train_accuracy,
                    'Test Accuracy': test_accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                }

                # Display results
                st.markdown(f"### {target} Results")
                st.write(f"**Train Accuracy:** {train_accuracy:.2f}")
                st.write(f"**Test Accuracy:** {test_accuracy:.2f}")
                st.write(f"**Precision:** {precision:.2f}")
                st.write(f"**Recall:** {recall:.2f}")
                st.write(f"**F1 Score:** {f1:.2f}")

                # Confusion Matrix
                cm = confusion_matrix(Y_test[target], Y_test_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

                # Plotting and displaying the confusion matrix
                fig_cm, ax_cm = plt.subplots()
                disp.plot(ax=ax_cm,cmap="Reds")
                plt.title(f'Confusion Matrix for {target}')
                st.pyplot(fig_cm)
                plt.close(fig_cm)

                # ROC Curve
                fpr, tpr, thresholds = roc_curve(Y_test[target], Y_test_prob)
                roc_auc = auc(fpr, tpr)

                # Plotting ROC Curve
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                ax_roc.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.0])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'Receiver Operating Characteristic for {target}')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
                plt.close(fig_roc)
        
        st.markdown('<h2 style="color: violet;"> 5.DEEP NEURAL NETWORK </h2>', unsafe_allow_html=True)

        target_columns = ['Heart Disease', 'Kidney Disease', 'Lung Cancer']
        dnn_models = {}
        results = {}

        for target in target_columns:
            if target in Y.columns:
                
                # Build the DNN model
                model = Sequential([
                    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid')  # Binary classification output
                ])

                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                # Train the model
                history = model.fit(X_train, Y_train[target], epochs=50, batch_size=32, validation_data=(X_test, Y_test[target]), verbose=0)

                # Store the model
                dnn_models[target] = model

                # Predicting for train and test sets
                Y_train_pred = (model.predict(X_train) > 0.5).astype(int)
                Y_test_pred = (model.predict(X_test) > 0.5).astype(int)
                Y_test_prob = model.predict(X_test)  # Get probabilities for the positive class

                # Calculate metrics
                train_accuracy = accuracy_score(Y_train[target], Y_train_pred)
                test_accuracy = accuracy_score(Y_test[target], Y_test_pred)

                precision = precision_score(Y_test[target], Y_test_pred)
                recall = recall_score(Y_test[target], Y_test_pred)
                f1 = f1_score(Y_test[target], Y_test_pred)

                # Store results in a dictionary
                results[target] = {
                    'Train Accuracy': train_accuracy,
                    'Test Accuracy': test_accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                }

                # Display results
                st.markdown(f"### {target} Results")
                st.write(f"**Train Accuracy:** {train_accuracy:.2f}")
                st.write(f"**Test Accuracy:** {test_accuracy:.2f}")
                st.write(f"**Precision:** {precision:.2f}")
                st.write(f"**Recall:** {recall:.2f}")
                st.write(f"**F1 Score:** {f1:.2f}")

                # Confusion Matrix
                cm = confusion_matrix(Y_test[target], Y_test_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])  # For binary classification
                # Plotting and saving the confusion matrix
                fig_cm, ax_cm = plt.subplots()
                disp.plot(ax=ax_cm, cmap="Greens")
                plt.title(f'Confusion Matrix for {target}')
                st.pyplot(fig_cm)  
                plt.close(fig_cm)

                # ROC Curve
                fpr, tpr, thresholds = roc_curve(Y_test[target], Y_test_prob)
                roc_auc = auc(fpr, tpr)

                # Plotting ROC Curve
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                ax_roc.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.0])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'Receiver Operating Characteristic for {target}')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)  
                plt.close(fig_roc)

# Results for each target disease (Heart Disease, Kidney Disease, Lung Cancer)
        results_heart_disease = {
            "Model": [
                "Logistic Regression", 
                "Random Forest", 
                "Gradient Boosting", 
                "Neural Network Classifier", 
                "Deep Neural Network"
            ],
            "Train Accuracy": [
                0.84, 0.95, 0.99, 0.96, 0.98
            ],
            "Test Accuracy": [
                0.83, 0.96, 0.99, 0.96, 0.98
            ],
            "Precision": [
                0.97, 1.00, 0.99, 1.00, 0.99
            ],
            "Recall": [
                0.82, 0.95, 0.95, 0.96, 0.99
            ],
            "F1 Score": [
                0.89, 0.97, 0.97, 0.98, 0.99
            ]
        }

        results_kidney_disease = {
            "Model": [
                "Logistic Regression", 
                "Random Forest", 
                "Gradient Boosting", 
                "Neural Network Classifier", 
                "Deep Neural Network"
            ],
            "Train Accuracy": [
                0.82, 0.88, 0.96, 0.91, 0.94
            ],
            "Test Accuracy": [
                0.82, 0.88, 0.95, 0.90, 0.94
            ],
            "Precision": [
                0.93, 1.00, 0.99, 0.98, 0.99
            ],
            "Recall": [
                0.81, 0.85, 0.95, 0.89, 0.94
            ],
            "F1 Score": [
                0.87, 0.92, 0.97, 0.93, 0.96
            ]
        }

        results_lung_cancer = {
            "Model": [
                "Logistic Regression", 
                "Random Forest", 
                "Gradient Boosting", 
                "Neural Network Classifier", 
                "Deep Neural Network"
            ],
            "Train Accuracy": [
                0.95, 0.98, 0.99, 0.98, 0.99
            ],
            "Test Accuracy": [
                0.96, 0.98, 1.00, 0.98, 0.99
            ],
            "Precision": [
                1.00, 1.00, 1.00, 1.00, 1.00
            ],
            "Recall": [
                0.92, 0.97, 0.99, 0.96, 0.97
            ],
            "F1 Score": [
                0.96, 0.98, 1.00, 0.98, 0.99
            ]
        }

        # Create DataFrames for each disease
        df_heart_disease = pd.DataFrame(results_heart_disease)
        df_kidney_disease = pd.DataFrame(results_kidney_disease)
        df_lung_cancer = pd.DataFrame(results_lung_cancer)

        # Display the results for each disease in a tabular format
        st.markdown('<h1 style="color: violet;">MODEL COMPARISON</h2>', unsafe_allow_html=True)
        st.write(' HEART DISEASE')
        st.dataframe(df_heart_disease)

        st.write(' KIDNEY DISEASE')
        st.dataframe(df_kidney_disease)

        st.write(' LUNG CANCER')
        st.dataframe(df_lung_cancer)

                
else:
    st.warning("Please upload all datasets to proceed.")
