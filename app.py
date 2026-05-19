# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
import uuid 
import os
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
app.secret_key = "hantavirus_tamamen_bagimsiz_o_dev_key" 

print("🦠 Hantavirüs Bağımsız Yapay Zeka Modeli Eğitiliyor...")

# VERİ SETİNİ YÜKLE VE EĞİT
try:
    df = pd.read_csv('hantavirus_dataset.csv')
    feature_cols = [
        'age', 'sex', 'state_region', 'exposure_setting',
        'fever', 'myalgia', 'fatigue', 'headache', 'nausea', 'dyspnea', 'cough',
        'diabetes', 'immunocompromised', 'days_to_hospitalisation'
    ]
    df_encoded = pd.get_dummies(df[feature_cols], columns=['sex', 'state_region', 'exposure_setting'])
    X = df_encoded.fillna(0)
    y = df['severity_score'].fillna(df['severity_score'].mean())
    
    model = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=42)
    model.fit(X, y)
    model_features = X.columns.tolist()
    print("✅ Bağımsız Yapay Zeka Hazır. Diğer projelerle bağ koparıldı!")
except Exception as e:
    print(f"❌ Hata: {e}")
    df = pd.DataFrame()
    model_features = []

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_score = None
    
    # Grafik verilerini tamamen bu tarayıcı oturumunda (session) saklıyoruz
    if 'local_scores' not in session:
        session['local_scores'] = []
        
    if request.method == 'GET':
        session['form_id'] = str(uuid.uuid4())[:8]
        
    f_id = session.get('form_id', 'hanta')
    
    if request.method == 'POST' and not df.empty:
        age = float(request.form.get(f'age_{f_id}', 30))
        sex = request.form.get(f'sex_{f_id}')
        region = request.form.get(f'region_{f_id}')
        exposure = request.form.get(f'exposure_{f_id}')
        days = float(request.form.get(f'days_{f_id}', 1))
        
        symptoms = {
            'fever': 1 if request.form.get(f'fever_{f_id}') else 0,
            'myalgia': 1 if request.form.get(f'myalgia_{f_id}') else 0,
            'fatigue': 1 if request.form.get(f'fatigue_{f_id}') else 0,
            'headache': 1 if request.form.get(f'headache_{f_id}') else 0,
            'nausea': 1 if request.form.get(f'nausea_{f_id}') else 0,
            'dyspnea': 1 if request.form.get(f'dyspnea_{f_id}') else 0,
            'cough': 1 if request.form.get(f'cough_{f_id}') else 0,
            'diabetes': 1 if request.form.get(f'diabetes_{f_id}') else 0,
            'immunocompromised': 1 if request.form.get(f'immunocompromised_{f_id}') else 0
        }
        
        input_dict = {col: 0 for col in model_features}
        if 'age' in input_dict: input_dict['age'] = age
        if 'days_to_hospitalisation' in input_dict: input_dict['days_to_hospitalisation'] = days
        
        for k, v in symptoms.items():
            if k in input_dict: input_dict[k] = v
            
        if f"sex_{sex}" in input_dict: input_dict[f"sex_{sex}"] = 1
        if f"state_region_{region}" in input_dict: input_dict[f"state_region_{region}"] = 1
        if f"exposure_setting_{exposure}" in input_dict: input_dict[f"exposure_setting_{exposure}"] = 1
        
        input_vector = [input_dict[col] for col in model_features]
        raw_score = float(model.predict([input_vector])[0])
        
        if symptoms['dyspnea'] == 1:
            raw_score += 1.5
            
        predicted_score = round(max(1.0, min(raw_score, 10.0)), 1)
        
        # Yeni skoru Firebase yerine lokal listeye ekle (Son 10 veri)
        scores_list = session['local_scores']
        scores_list.append(predicted_score)
        session['local_scores'] = scores_list[-10:] # Son 10 vaka tut
        
        session['form_id'] = str(uuid.uuid4())[:8]

    return render_template('index.html', 
                           predicted_score=predicted_score,
                           graph_scores=session['local_scores'],
                           f_id=session.get('form_id', 'hanta'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)