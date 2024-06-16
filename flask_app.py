"""
@auther : Abhishek Pawar
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from flasgger import Swagger
import joblib
from sklearn.preprocessing import FunctionTransformer

# Define the custom preprocessing function
def custom_preprocessing(df):
    if 'NoDocbcCost' in df.columns:
        df.drop(['NoDocbcCost'], axis=1, inplace=True)
    if 'AnyHealthcare' in df.columns:
        df.drop(['AnyHealthcare'], axis=1, inplace=True)
    df['BMI'] = np.log(df['BMI'])
    df['MentHlth'] = np.log1p(df['MentHlth'])
    df['PhysHlth'] = np.log1p(df['PhysHlth'])
    return df

# Wrap the function in a FunctionTransformer
preprocessing_transformer = FunctionTransformer(custom_preprocessing)

app = Flask(__name__)
Swagger(app)

# Load the model
classifier = joblib.load('classifier_model1.pkl')
classifier1 = joblib.load('classifier_model.pkl')

@app.route('/')
def welcome():
    return "Welcome Friends..."

@app.route('/predict', methods=["GET"])
def predict_diabetes():
    """Let's Predict the Diabetes
    ---
    parameters:
        - name: HighBP
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: HighChol
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: CholCheck
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: BMI
          in: query
          type: number
          required: True
        - name: Smoker
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: Stroke
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: HeartDiseaseorAttack
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: PhysActivity
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: Fruits
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: Veggies
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: HvyAlcoholConsump
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: GenHlth
          in: query
          type: number
          required: True
          enum: [1, 2, 3, 4, 5]
          default: 1
        - name: MentHlth
          in: query
          type: number
          required: True
        - name: PhysHlth
          in: query
          type: number
          required: True
        - name: DiffWalk
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: Sex
          in: query
          type: number
          required: True
          enum: [0, 1]
          default: 0
        - name: Age
          in: query
          type: number
          required: True
        - name: Education
          in: query
          type: number
          required: True
          enum: [1, 2, 3, 4, 5, 6]
          default: 1
        - name: Income
          in: query
          type: number
          required: True
          enum: [1, 2, 3, 4, 5, 6, 7, 8]
          default: 1
    responses:
        200:
            description: The Output value
    """
    try:
        # Get the data from the request
        data = {
            'HighBP': [int(request.args.get('HighBP'))],
            'HighChol': [int(request.args.get('HighChol'))],
            'CholCheck': [int(request.args.get('CholCheck'))],
            'BMI': [float(request.args.get('BMI'))],
            'Smoker': [int(request.args.get('Smoker'))],
            'Stroke': [int(request.args.get('Stroke'))],
            'HeartDiseaseorAttack': [int(request.args.get('HeartDiseaseorAttack'))],
            'PhysActivity': [int(request.args.get('PhysActivity'))],
            'Fruits': [int(request.args.get('Fruits'))],
            'Veggies': [int(request.args.get('Veggies'))],
            'HvyAlcoholConsump': [int(request.args.get('HvyAlcoholConsump'))],
            'GenHlth': [int(request.args.get('GenHlth'))],
            'MentHlth': [int(request.args.get('MentHlth'))],
            'PhysHlth': [int(request.args.get('PhysHlth'))],
            'DiffWalk': [int(request.args.get('DiffWalk'))],
            'Sex': [int(request.args.get('Sex'))],
            'Age': [int(request.args.get('Age'))],
            'Education': [int(request.args.get('Education'))],
            'Income': [int(request.args.get('Income'))]
        }

        df = pd.DataFrame(data)
        prediction = classifier.predict(df)
        
        return jsonify({'prediction': str(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    
@app.route('/predict_file', methods=['POST'])
def data_file():
    """Let's check whether the person has diabetes or not
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: True
    responses:
        200:
            description: The output values
    """
    try:
        df = pd.read_csv(request.files.get("file"))
        df = preprocessing_transformer.transform(df)
        prediction = classifier1.predict(df)
        return jsonify({"predictions": prediction.tolist()})
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)