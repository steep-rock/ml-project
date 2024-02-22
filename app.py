
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData, PredictPipeine

application = Flask(__name__)

app = application

#Route for homepage 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data',methods=["GET","POST"])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        predict_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeine()
        result = predict_pipeline.predict(predict_df)
        return render_template('home.html',results= result[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",port=4999,debug=True)  
