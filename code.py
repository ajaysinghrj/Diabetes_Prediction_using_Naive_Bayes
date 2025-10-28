import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
data = pd.read_csv(url, names=cols)

# Step 2: Train-Test Split
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = model.predict(X_test)
acc = round(accuracy_score(y_test, y_pred)*100, 2)
cm = confusion_matrix(y_test, y_pred)

print("=============================================")
print("   DIABETES PREDICTION USING NAIVE BAYES")
print("=============================================")
print(f"Accuracy : {acc}%")
print("Confusion Matrix :")
print(cm)
print("=============================================")

# Step 5: Predict for New Input
print("Enter Patient Details Below 👇")
try:
    vals = []
    features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    for f in features:
        vals.append(float(input(f"{f}: ")))

    pred = model.predict([vals])[0]
    print("---------------------------------------------")
    print("Result:", "🩸 Diabetic" if pred==1 else "✅ Non-Diabetic")
    print("=============================================")
except:
    print("Invalid input! Please enter numbers only.")
