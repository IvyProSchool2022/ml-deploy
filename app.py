from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST","GET"])
def predict():
    feature1=float(request.form['feature1'])
    feature2=float(request.form['feature2'])
    feature3=float(request.form['feature3'])
    feature4=float(request.form['feature4'])
    feature5=float(request.form['feature5'])

    user_input=np.array([[feature1,feature2,feature3,feature4,feature5]])
    # scaled_input=scaler.transform(user_input)
    prediction=model.predict(user_input)

    return render_template("home.html",prediction_text="Predicted Sales is $ {:.0f}".format(prediction[0]))

if __name__== '__main__':
    app.run(debug=True)
