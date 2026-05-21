# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.ensemble import RandomForestRegressor
import os

# 1. Firebase Bağlantısını başlat (JSON dosyanın ana dizinde olmalı)
cred = credentials.Certificate("hantavirus-klinik-firebase-adminsdk-fbsvc-9f14482855.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)
app.secret_key = "hantavirus_tamamen_bagimsiz_o_dev_key"

# Veri seti ve model hazırlığı
df = pd.read_csv('hantavirus_dataset.csv')
feature_cols = ['age', 'sex', 'state_region', 'exposure_setting', 'fever', 'myalgia', 
                'fatigue', 'headache', 'nausea', 'dyspnea', 'cough', 'diabetes', 
                'immunocompromised', 'days_to_hospitalisation']

X = df[feature_cols]
y = df['days_to_hospitalisation']
model = RandomForestRegressor()
model.fit(X, y)

# 2. Firebase'e kayıt fonksiyonu
def veriyi_kaydet(isim, sonuc):
    db.collection('klinik_veriler').add({
        'kullanici_adi': isim,
        'tahmin_sonucu': str(sonuc),
        'tarih': firestore.SERVER_TIMESTAMP
    })

@app.route('/', methods=['GET', 'POST'])
def index():
    tahmin_sonucu = None
    if request.method == 'POST':
        # Kullanıcıdan form verilerini al
        isim = request.form.get('isim', 'İsimsiz')
        
        # Model tahminini yap
        input_data = [float(request.form.get(col, 0)) for col in feature_cols if col != 'days_to_hospitalisation']
        # Modelin beklediği sütun sayısına göre tahmin (hata almamak için)
        tahmin_sonucu = model.predict([input_data[:len(feature_cols)-1]])[0]
        
        # 3. Veriyi Firebase'e gönder
        veriyi_kaydet(isim, tahmin_sonucu)
        
    return render_template('index.html', sonuc=tahmin_sonucu)

if __name__ == '__main__':
    app.run(debug=True)
