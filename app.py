from flask_cors import CORS
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import openai
import json
import os
import ast
import re
import requests
from prophet import Prophet
app = Flask(__name__)
CORS(app)
# Definizione delle colonne da usare



@app.route('/cluster-clienti', methods=['POST'])
def cluster_clienti():
    try:
        # 1. Leggi i dati dal JSON inviato
        features = [
            "fatture_interamente_pagate",
            "num_pets",
            "numero_preventivi",
            "note_di_credito",
            "debito_attuale",
            "credito_attuale",
            "spesa_totale",
            "numero_appuntamenti",
            "quantita_visite"
        ]
        data = request.get_json()
        df = pd.DataFrame(data)

        # Verifica colonne mancanti
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            return jsonify({"errore": f"Colonne mancanti nel dataset: {missing_cols}"}), 400

        # 2. Clustering
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = min(8, len(df))
        if n_clusters < 2:
            return jsonify({"errore": "Sono necessari almeno 2 clienti per eseguire il clustering."}), 400

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        # 3. Output JSON clienti + cluster
        df_export = df[["id_cliente", "cluster"]]
        json_output = df_export.to_dict(orient="records")

        # 4. Medie per cluster per il prompt
        cluster_stats = df.groupby("cluster")[features].mean().round(2)
        cluster_stats_str = cluster_stats.to_string()

        # 5. Chiamata a GPT con richiesta output JSON
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        prompt = f"""
        You are a marketing expert. Analyze the averages of the following {n_clusters} customer clusters from a veterinary clinic.

        For each cluster, return a JSON object using the following keys in Italian:
        {{
        "cluster": <number>,
        "nome": "<cluster name>",
        "descrizione": "<behavior description>",
        "strategia": "<marketing strategy>"
        }}
        Guidelines:
        - Provide a detailed and professional analysis.
        - Use clear, business-oriented language.
        - Make sure the response is a valid, parsable JSON array without any additional text.
        - The cluster name ('nome') should be concise and memorable.
        - The description ('descrizione') should summarize key behavioral traits.
        - The strategy ('strategia') should be practical and tailored to the cluster profile.

        Respond only with a valid JSON array (no additional text).

        Here are the average data for each cluster:

        {cluster_stats_str}
        """

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a segmented marketing expert specializing in customer behavior analysis for veterinary clinics. "
                "Your task is to generate detailed, practical marketing insights. "
                "Always respond using clear, professional business language. "
                "Unless explicitly asked otherwise, provide output strictly in valid JSON format, using only the specified Italian keys ('cluster', 'nome', 'descrizione', 'strategia'). "
                "Focus on realistic and actionable marketing strategies tailored to each customer cluster."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )

        gpt_output = response.choices[0].message.content

        # 6. Pulizia e parsing JSON da GPT
        json_match = re.search(r"\[\s*{.*?}\s*\]", gpt_output, re.DOTALL)
        if json_match:
            gpt_output_clean = json_match.group(0)
        else:
            gpt_output_clean = "[]"

        try:
            cluster_parsed = json.loads(gpt_output_clean)
        except json.JSONDecodeError:
            try:
                cluster_parsed = ast.literal_eval(gpt_output_clean)
            except Exception:
                cluster_parsed = []

        # 7. Risposta API
        return jsonify({
            "cluster_clienti": json_output,
            "analisi_marketing": cluster_parsed
        })

    except Exception as e:
        return jsonify({"errore": str(e)}), 500

@app.route('/cluster-da-php', methods=['GET'])
def cluster_da_php():
    try:
        f_idazienda = request.args.get('f_idazienda')
        if not f_idazienda:
            return jsonify({"errore": "Parametro 'f_idazienda' mancante."}), 400

        url_php = f"https://evolutionext.it/ai-api/Insight_cliente.php?f_idazienda={f_idazienda}"
        response = requests.get(url_php)
        clienti = response.json()

        # Usa direttamente la funzione cluster_clienti simulando una POST
        with app.test_request_context(method='POST', json=clienti):
            return cluster_clienti()

    except Exception as e:
        return jsonify({"errore": str(e)}), 500
    
@app.route('/forecast-appuntamenti', methods=['POST'])
def forecast_appuntamenti():
    try:
        print("Metodo:", request.method)
        print("Payload ricevuto:", request.get_json())

        payload = request.get_json()
        f_idazienda = payload.get("f_idazienda")
        frequenza = payload.get("frequenza", "D")  # "D", "W", "M"
        periodi = int(payload.get("periodi", 60))

        # Recupero i dati dalla  API PHP
        url = f"https://evolutionext.it/ai-api/lavoro-clinica.php?f_idazienda={f_idazienda}"
        response = requests.get(url)
        dati = response.json()

        df = pd.DataFrame(dati)
        df["ds"] = pd.to_datetime(df["data"], format="%Y%m%d")
        df["y"] = df["num_appuntamenti"]
        df = df.sort_values("ds")


        model_args = {
            "growth": "linear",
            "yearly_seasonality": False,
            "weekly_seasonality": True,
            "changepoint_prior_scale": 2
        }

        modello = Prophet(**model_args)
        modello.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        modello.fit(df)

        # Future con frequenza giornaliera
        futuro = modello.make_future_dataframe(periods=periodi, freq="D")
        previsione = modello.predict(futuro)

        # Aggrego il forecast secondo la frequenza scelta dall'utente
        previsione['data_aggregata'] = previsione['ds'].dt.to_period(frequenza).apply(lambda r: r.start_time)
        forecast_aggregato = previsione.groupby('data_aggregata')[['yhat', 'yhat_lower', 'yhat_upper']].sum().reset_index()
        forecast_aggregato.columns = ['data', 'previsione', 'min', 'max']
        forecast_aggregato['data'] = forecast_aggregato['data'].dt.strftime("%Y-%m-%d")

        # Stagionalità (rimane giornaliera)
        stagionalita = previsione[['ds']].copy()
        stagionalita['data'] = stagionalita['ds'].dt.strftime("%Y-%m-%d")
        if 'weekly' in previsione.columns:
            stagionalita['stagionalita_settimanale'] = previsione['weekly']
        if 'monthly' in previsione.columns:
            stagionalita['stagionalita_mensile'] = previsione['monthly']

        return jsonify({
            "success": True,
            "forecast": forecast_aggregato.to_dict(orient='records'),
            "stagionalita": stagionalita.drop(columns=['ds']).to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({"errore": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)




