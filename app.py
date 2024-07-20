import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Function to configure the model based on user selection
def get_model(model_choice):
    if model_choice == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
        params = {'C': [0.01, 0.1, 1, 10]}
    elif model_choice == 'Random Forest':
        model = RandomForestClassifier()
        params = {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}
    elif model_choice == 'Gradient Boosting':
        model = GradientBoostingClassifier()
        params = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    elif model_choice == 'SVM':
        model = SVC()
        params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif model_choice == 'SGD Classifier':
        model = SGDClassifier()
        params = {'alpha': [0.0001, 0.001, 0.01], 'loss': ['hinge', 'log']}
    return model, params

# Main title
st.title('Happiness Classification Dashboard')

# File uploader for user to add CSV data
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file:
    # Load and display data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Preprocess data
    X = data.drop('happy', axis=1)
    y = data['happy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model selection dropdown
    model_choice = st.selectbox("Choose a classification model:", ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'SGD Classifier'])

    # GridSearch toggle
    use_gridsearch = st.selectbox("Use GridSearchCV?", ['Yes', 'No'])

    # Model setup and training
    model, params = get_model(model_choice)
    if use_gridsearch == 'Yes':
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        st.write("Best Model Parameters:", grid_search.best_params_)
    else:
        best_model = model
        best_model.fit(X_train_scaled, y_train)

    # Model evaluation
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)

    # Display metrics
    st.subheader("Model Performance:")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:", pd.DataFrame(clf_report).transpose())
    st.write("Confusion Matrix:", conf_mat)

    # Button to generate EDA based on selected model
    if st.button("Generate EDA Graphs"):
        st.subheader("Exploratory Data Analysis")
        # Pairplot
        sns.pairplot(data, hue='happy', palette='viridis')
        st.pyplot()
        
        # Heatmap of correlations
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        st.pyplot()

        # Distribution of each feature
        fig, ax = plt.subplots(1, len(data.columns)-1, figsize=(20, 5))
        for i, col in enumerate(data.columns[:-1]):
            sns.histplot(data[col], kde=True, ax=ax[i])
            ax[i].set_title(f'Distribution of {col}')
        st.pyplot()

else:
    st.warning("Please upload a CSV file to continue.")
