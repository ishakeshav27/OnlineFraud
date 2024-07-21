#using flask linking webpage and py prog
from flask import Flask,render_template,request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check_fraud", methods=["POST"])
def check_fraud():
    txn_id = request.form.get("txn_id")
    amount = request.form.get("amount")
    if txn_id and amount:
        if is_fraud(txn_id, amount):
            return "Fraud Transaction Detected!"
        else:
            return "Transaction is not Fraudulent."
    else:
        return "Please enter valid Transaction type, ID and Amount."

def is_fraud(txn_id, amount):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.tree import DecisionTreeClassifier
    #reading data
    df = pd.read_csv("C:/Users/neeha_ocab0af/OneDrive/Desktop/codher/dataset.csv")
    #print(df.head(10))

    type=df['type'].value_counts()
    transaction=type.index
    quantity=type.values
    import plotly.express as px
    px.pie(df,values=quantity,names=transaction,hole=0.4,title="Distribution of transaction type:")
    df=df.dropna()
    df.replace(to_replace=['PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN'],value=[2,4,1,5,3],inplace=True)
    df['isFraud']=df['isFraud'].map({0:'No fraud',1:'fraud'})
    x=df[['type','amount','oldbalanceOrg','newbalanceOrig']]
    y=df.iloc[:,-2]
    model=DecisionTreeClassifier()
    #splitting data
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
    #training data
    model.fit(xtrain,ytrain)
    #accuracy& prediction
    acc=model.score(xtest,ytest)
    ypred = model.predict(xtest)
    conf_mat=classification_report(ytest, ypred)

    #print("\nConfusion Matrix:\n\n",conf_mat)
    #print("Accuracy: ",acc,"\n")

    # This is just an example code, replace it with your actual code
    if acc > 0.5:
        return render_template ('index.html',pred='This is a Fraud Transaction'\nProbability of Fraud transaction occuring is {}'.format(acc))
    else:
        return render_template('index.html',pred='Not a Fraud Transaction'\nProbability of Fraud transaction occuring is{}'.format(acc))


if __name__ == "__main__":
    app.run(debug=True)