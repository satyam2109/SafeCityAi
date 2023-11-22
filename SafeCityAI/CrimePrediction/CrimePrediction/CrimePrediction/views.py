from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    dataset = pd.read_csv(r'C:\Users\satyam\Desktop\SafeCityAI\data.csv')
    data = pd.read_csv(r'C:\Users\satyam\Desktop\SafeCityAI\data.csv')
    for col in data:
        print(type(data[col][1]))
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%Y %H:%M:%S')
    column_1 = data.iloc[:, 0]
    db = pd.DataFrame({"year": column_1.dt.year,
                       "month": column_1.dt.month,
                       "day": column_1.dt.day,
                       "hour": column_1.dt.hour,
                       "dayofyear": column_1.dt.dayofyear,
                       "week": column_1.dt.week,
                       "weekofyear": column_1.dt.weekofyear,
                       "dayofweek": column_1.dt.dayofweek,
                       "weekday": column_1.dt.weekday,
                       "quarter": column_1.dt.quarter,
                       })
    dataset1 = dataset.drop('timestamp', axis=1)
    data1 = pd.concat([db, dataset1], axis=1)
    data1.dropna(inplace=True)
    X = data1.iloc[:, [0, 1, 2, 3,  16, 17]].values
    y = data1.iloc[:, [10, 11, 12, 13, 14, 15]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train)
    val1=float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3= float(request.GET['n3'])
    val4=float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    pred=rfc.predict([[ val3, val4,val5,val6,val1,val2]])
    result1=""
    if pred.any()==1:
        result1=" Crime is likely to happen."
    else:
        result1=" Crime is not likely to happen"
    return render(request, 'predict.html', {"result2":result1})

