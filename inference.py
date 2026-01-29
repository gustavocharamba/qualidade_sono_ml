import joblib
import pandas as pd

model = joblib.load('melhor_modelo_sono.pkl')

person = pd.DataFrame(columns=model.feature_names_in_)
person.loc[0] = 0

person.at[0, "Age"] = 20
person.at[0, "Sleep Duration"] = 8
person.at[0, "Activity"] = 60
person.at[0, "Stress Level"] = 2
person.at[0, "Heart Rate"] = 78
person.at[0, "Steps"] = 1000
person.at[0, "Pressure_Systolic"] = 120
person.at[0, "Pressure_Diastolic"] = 75

# One-Hot
person.at[0, "Gender_Male"] = 1
person.at[0, "BMI_Obese"] = 0
person.at[0, "BMI_Overweight"] = 0
person.at[0, "Occupation_Doctor"] = 1

pred = model.predict(person)
print(f'Qualdiade de sono estimada: {pred[0]:.2f}')