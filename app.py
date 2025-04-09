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


@app.route('/cluster-clienti', methods=['POST'])
def cluster_clienti():
    try:
        # 1. Leggi i dati dal JSON inviato
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
        Sei un esperto di marketing. Analizza le medie dei seguenti {n_clusters} cluster clienti di una clinica veterinaria.

        Per ciascun cluster restituisci un oggetto JSON nel seguente formato:
        {{
          "cluster": <numero>,
          "nome": "<nome cluster>",
          "descrizione": "<descrizione comportamento>",
          "strategia": "<strategia marketing>"
        }}

        Rispondi solo con un array JSON valido, senza testo aggiuntivo.

        Ecco i dati medi per cluster:

        {cluster_stats_str}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sei un esperto di marketing segmentato."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
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

        url_php = f"https://www.demoevolution.it/clinic/Insight_cliente.php?f_idazienda={f_idazienda}"
        response = requests.get(url_php)
        clienti = response.json()

        # Usa direttamente la funzione cluster_clienti simulando una POST
        with app.test_request_context(method='POST', json=clienti):
            return cluster_clienti()

    except Exception as e:
        return jsonify({"errore": str(e)}), 500
    
@app.route('/forecast-appuntamenti', methods=['GET'])
def forecast_appuntamenti():
    try:
        if request.method == 'POST':
            payload = request.get_json()
            dati = payload.get("dati")
            frequenza = payload.get("frequenza", "D")
            periodi = int(payload.get("periodi", 60))
            params = payload.get("params", {})

            df = pd.DataFrame(dati)
            df['ds'] = pd.to_datetime(df['ds'])
            df['y'] = pd.to_numeric(df['y'])

            if frequenza != "D":
                df = df.set_index('ds').resample(frequenza).mean().reset_index()
        else:
            f_idazienda = request.args.get('f_idazienda', default=7, type=int)
            url = f"https://www.mychartjourney.com/api/lavoro.php?f_idazienda={f_idazienda}"
            response = requests.get(url)
            dati = response.json()

            df = pd.DataFrame(dati)
            df["ds"] = pd.to_datetime(df["data"], format="%Y%m%d")
            df["y"] = df["num_appuntamenti"]
            df = df.sort_values("ds")
            frequenza = "D"
            periodi = 60
            params = {}

        model_args = {
            "growth": params.get("growth", "linear"),
            "yearly_seasonality": params.get("yearly_seasonality", False),
            "weekly_seasonality": params.get("weekly_seasonality", True),
            "changepoint_prior_scale": params.get("changepoint_prior_scale", 0.1)
        }

        modello = Prophet(**model_args)
        modello.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        modello.fit(df)

        futuro = modello.make_future_dataframe(periods=periodi, freq=frequenza)
        previsione = modello.predict(futuro)

        forecast_export = previsione[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast_export.columns = ['data', 'previsione', 'min', 'max']
        forecast_export['data'] = forecast_export['data'].dt.strftime("%Y-%m-%d")

        stagionalita = previsione[['ds']].copy()
        stagionalita['data'] = stagionalita['ds'].dt.strftime("%Y-%m-%d")
        if 'weekly' in previsione.columns:
            stagionalita['stagionalita_settimanale'] = previsione['weekly']
        if 'monthly' in previsione.columns:
            stagionalita['stagionalita_mensile'] = previsione['monthly']

        return jsonify({
            "success": True,
            "forecast": forecast_export.to_dict(orient='records'),
            "stagionalita": stagionalita.drop(columns=['ds']).to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({"errore": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
