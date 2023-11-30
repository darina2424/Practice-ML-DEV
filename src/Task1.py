import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


data = pd.read_csv(r'C:\Users\Дарина\Desktop\MLP\dataset4\NHANES_age_prediction.csv')


print(data.head())


print(data.info())


print(data.describe())


print(data.isnull().sum())


data.dropna(inplace=True)


for column in data.columns:
    plt.figure()
    sns.histplot(data[column], kde=True)
    plt.title(f'Histogram for {column}')
    plt.savefig(f'{column}_histogram.png', bbox_inches='tight') 


numerical_data = data.select_dtypes(include=[np.number])
corr_matrix = numerical_data.corr()

plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png', bbox_inches='tight')  

sns.pairplot(data, hue='age_group') 
plt.savefig('pairplot.png', bbox_inches='tight') 


plt.figure()
sns.countplot(x='age_group', data=data, palette='Set3', hue='age_group', legend=False)
plt.title('Count Plot for age_group')
plt.xticks(rotation=45)  
plt.savefig('age_group_countplot.png', bbox_inches='tight')  


X = data.drop('age_group', axis=1)
y = data['age_group']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


rf_predictions = rf_model.predict(X_test)


rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions, average='weighted')
rf_precision = precision_score(y_test, rf_predictions, average='weighted')


print("RandomForestClassifier:")
print(f'F1 Score: {rf_f1}')
print(f'Accuracy: {rf_accuracy}')
print(f'Recall: {rf_recall}')
print(f'Precision: {rf_precision}')
print()


dummy_model = DummyClassifier(strategy='stratified') 
dummy_model.fit(X_train, y_train)


dummy_predictions = dummy_model.predict(X_test)


dummy_f1 = f1_score(y_test, dummy_predictions, average='weighted')
dummy_accuracy = accuracy_score(y_test, dummy_predictions)
dummy_recall = recall_score(y_test, dummy_predictions, average='weighted')
dummy_precision = precision_score(y_test, dummy_predictions, average='weighted')


print("DummyClassifier:")
print(f'F1 Score: {dummy_f1}')
print(f'Accuracy: {dummy_accuracy}')
print(f'Recall: {dummy_recall}')
print(f'Precision: {dummy_precision}')
print()


svm_model = SVC()
svm_model.fit(X_train, y_train)


svm_predictions = svm_model.predict(X_test)

svm_f1 = f1_score(y_test, svm_predictions, average='weighted')
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_precision = precision_score(y_test, svm_predictions, average='weighted')

print("Support Vector Machine:")
print(f'F1 Score: {svm_f1}')
print(f'Accuracy: {svm_accuracy}')
print(f'Recall: {svm_recall}')
print(f'Precision: {svm_precision}')

