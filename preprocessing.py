import pandas as pd

def preprocessing(df):
    #excluir colunas; person ID não diz nada importante; sleep disorder entrega resultado
    df.drop(columns=['Person ID', 'Sleep Disorder'], inplace=True)

    #renomeando colunas
    df.rename(columns={
        'BMI Category': 'BMI',
        'Quality of Sleep': 'Quality',
        'Physical Activity Level': 'Activity',
        'Daily Steps': 'Steps',
    }, inplace=True)

    #padronizando rótulos
    df['BMI'] = df['BMI'].replace({'Normal Weight': 'Normal'})

    #separação da pressão arterial em sistólica e diastólica
    bp = df['Blood Pressure'].str.split('/', expand=True)
    df['Pressure_Systolic'] = pd.to_numeric(bp[0])
    df['Pressure_Diastolic'] = pd.to_numeric(bp[1])
    df.drop(columns=['Blood Pressure'], inplace=True)

    return df