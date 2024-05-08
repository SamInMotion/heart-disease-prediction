# %%
#Required Libraries
import pandas as pd
#Load Dataset

dataset_path = "C:/Users/samue/Desktop/Projects/data/heart.csv" #  Adjust file path if needed
df = pd.read_csv(dataset_path)


#Display Initial Rows
df.head()


# %%
#dataset overview
df.info()

#initial statistics
df.describe()

#Checking for missing values
missing_values = df.isnull().sum()
print('Missing values:\n',missing_values)

# %%
#correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True,cmap= 'coolwarm')
plt.title('Correlation Matrix')

# %%
#target distribution
sns.countplot(df['target'])
plt.title('Target Distribution')
plt.show

#pairplot
sns.pairplot(df,hue='target')

# %%
#boxplot for numerical features
numeric_features = ['age', 'trestbps','chol','thalach','oldpeak']
df[numeric_features].plot(kind='box',subplots=True, layout=(1,5),figsize=(20,5))
plt.show()

# %%
#feature engineering- age categories
df['age_category']= pd.cut(df['age'], bins=[29,39,49,59,69,79],labels=['30-39','40-49','50-59','60-69','70+'])
#checking distribution
sns.countplot(x='age_category',hue='target',data=df)
plt.title('Age category vs target')
plt.show

# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Ensure All Feature Names Exist in the DataFrame
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['cp', 'slope', 'thal', 'ca']

# Remove 'age_category' from the categorical features if not present yet
if 'age_category' not in df.columns:
    df['age_category'] = pd.cut(df['age'], bins=[29, 39, 49, 59, 69, 79], labels=['30-39', '40-49', '50-59', '60-69', '70+'])
    categorical_features.append('age_category')

# Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first'))
])

# Combined Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Apply Preprocessing Pipeline
df_preprocessed = preprocessor.fit_transform(df)

# Check the shape of the preprocessed data
print("Preprocessed Data Shape:", df_preprocessed.shape)


# %%
print("DataFrame Columns:", df.columns)


# %%
# Extract Target Variable
y = df['target']

# Ensure Index Alignment
df_preprocessed = pd.DataFrame(df_preprocessed)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_preprocessed, y, test_size=0.3, random_state=42)


# %%
from sklearn.linear_model import LogisticRegression

# Initialize Model
model = LogisticRegression()

# Train Model
model.fit(X_train, y_train)


# %%
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Predict on Test Data
y_pred = model.predict(X_test)

# Accuracy Score
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion, annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix")
plt.show()


# %%
from sklearn.model_selection import GridSearchCV

# Parameter Grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

# Initialize Grid Search
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best Parameters and Model
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_


# %%
# Feature Importance
importance = best_model.coef_[0]
feature_names = preprocessor.get_feature_names_out()
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()



