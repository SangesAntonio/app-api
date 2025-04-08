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

from darts import TimeSeries
from darts.models import XGBModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries






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
@app.route('/forecast-lavoro', methods=['GET'])
def forecast_lavoro():

  

    f_idazienda = request.args.get('f_idazienda')
    if not f_idazienda:
        return jsonify({"errore": "Parametro f_idazienda mancante"}), 400

    try:
        # 1. Richiama PHP per dati eventi giornalieri
        res = requests.get(f"https://www.demoevolution.it/clinic/lavoro-clinica.php?f_idazienda={f_idazienda}")
        dati = res.json()
        df = pd.DataFrame(dati)
        df['data'] = pd.to_datetime(df['data'])
        df = df.set_index('data')

        # 2. Prepara dati per forecast
        colonne_target = df.columns.drop("staff_presente") if "staff_presente" in df.columns else df.columns
        scaler = Scaler()
        target_scaled = scaler.fit_transform(TimeSeries.from_series(df[colonne_target].sum(axis=1)))

        oggi = df.index.max()
        past_covariates = None
        future_covariates = None

        for col in df.columns:
            serie = TimeSeries.from_series(df[col])
            serie_scaled = scaler.transform(serie)
            if df.loc[df.index > oggi, col].notna().sum() > 0:
                future_covariates = serie_scaled if future_covariates is None else future_covariates.stack(serie_scaled)
            else:
                past_covariates = serie_scaled if past_covariates is None else past_covariates.stack(serie_scaled)

        # Aggiungi attributi temporali
        past_covariates = past_covariates.stack(datetime_attribute_timeseries(past_covariates, "month", one_hot=True))
        past_covariates = past_covariates.stack(datetime_attribute_timeseries(past_covariates, "day_of_week", one_hot=True))
        if future_covariates:
            future_covariates = future_covariates.stack(datetime_attribute_timeseries(future_covariates, "month", one_hot=True))
            future_covariates = future_covariates.stack(datetime_attribute_timeseries(future_covariates, "day_of_week", one_hot=True))

        # 3. Forecast
        model = XGBModel(lags=30, lags_past_covariates=30, output_chunk_length=60, random_state=42)
        model.fit(series=target_scaled, past_covariates=past_covariates, future_covariates=future_covariates)
        forecast_scaled = model.predict(n=60, past_covariates=past_covariates, future_covariates=future_covariates)
        forecast = scaler.inverse_transform(forecast_scaled)

        # 4. Prepara JSON
        forecast_df = forecast.pd_dataframe().reset_index()
        forecast_df.columns = ["data", "carico_lavoro_previsto"]

        if "staff_presente" in df.columns:
            staff_futuro = df["staff_presente"].reindex(forecast_df["data"], method="nearest")
            forecast_df["staff_presente"] = staff_futuro.values

        forecast_json = forecast_df.to_dict(orient="records")

        # 5. Analisi GPT
        prompt = f"""
        Ti invio i dati previsionali del carico di lavoro di una clinica veterinaria per i prossimi 60 giorni.

        Analizzali e dimmi:
        - Quai sono i giorni più critici?
        - Esistono trend in crescita o calo?
        - Suggerisci eventuali strategie di pianificazione per il personale.
        Non restituire i valori numerici esatti.

        Dati:
        {forecast_json}
        """

        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        gpt_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sei un esperto analista di dati di cliniche veterinarie."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return jsonify({
            "forecast": forecast_json,
            "analisi_gpt": gpt_response.choices[0].message.content
        })

    except Exception as e:
        return jsonify({"errore": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
