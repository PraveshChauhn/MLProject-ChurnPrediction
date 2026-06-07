import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

df_1 = pd.read_csv(r"C:\Users\Dell\Downloads\first_telc.csv")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']
    inputQuery15 = request.form['query15']
    inputQuery16 = request.form['query16']
    inputQuery17 = request.form['query17']
    inputQuery18 = request.form['query18']
    inputQuery19 = request.form['query19']

    model = pickle.load(open(r"C:\Users\Dell\Downloads\model.sav", "rb"))

    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5,
             inputQuery6, inputQuery7, inputQuery8, inputQuery9, inputQuery10,
             inputQuery11, inputQuery12, inputQuery13, inputQuery14, inputQuery15,
             inputQuery16, inputQuery17, inputQuery18, inputQuery19]]

    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges',
                                         'gender', 'Partner', 'Dependents', 'PhoneService',
                                         'MultipleLines', 'InternetService', 'OnlineSecurity',
                                         'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract',
                                         'PaperlessBilling', 'PaymentMethod', 'tenure'])

    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2['tenure'].astype(float).astype(int),
                                  range(1, 80, 12), right=False, labels=labels)
    df_2.drop(columns=['tenure'], inplace=True)

    new_df__dummies = pd.get_dummies(df_2[['gender', 'Partner', 'Dependents',
                                       'PhoneService', 'MultipleLines', 'InternetService',
                                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                                       'Contract', 'PaperlessBilling', 'PaymentMethod',
                                       'tenure_group']])

    new_df__dummies['MonthlyCharges'] = pd.to_numeric(df_2['MonthlyCharges'], errors='coerce').fillna(0)
    new_df__dummies['TotalCharges'] = pd.to_numeric(df_2['TotalCharges'], errors='coerce').fillna(0)
    new_df__dummies['SeniorCitizen'] = pd.to_numeric(df_2['SeniorCitizen'], errors='coerce').fillna(0).astype(int)

    single = model.predict(new_df__dummies.tail(1))
    probability = model.predict_proba(new_df__dummies.tail(1))[:, 1]

    if single[0] == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)

    return render_template('home.html', output1=o1, output2=o2,
                           query1=inputQuery1, query2=inputQuery2,
                           query3=inputQuery3, query4=inputQuery4,
                           query5=inputQuery5, query6=inputQuery6,
                           query7=inputQuery7, query8=inputQuery8,
                           query9=inputQuery9, query10=inputQuery10,
                           query11=inputQuery11, query12=inputQuery12,
                           query13=inputQuery13, query14=inputQuery14,
                           query15=inputQuery15, query16=inputQuery16,
                           query17=inputQuery17, query18=inputQuery18,
                           query19=inputQuery19)

app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)