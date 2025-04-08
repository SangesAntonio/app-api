# app.py
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)