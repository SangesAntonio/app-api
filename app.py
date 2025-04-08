# app.py
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import openai
import json
import os

app = Flask(__name__)

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

        kmeans = KMeans(n_clusters=8, random_state=42)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        # 3. Output JSON clienti + cluster
        df_export = df[["id_cliente", "cluster"]]
        json_output = df_export.to_dict(orient="records")

        # 4. Medie per cluster per il prompt
        cluster_stats = df.groupby("cluster")[features].mean().round(2)
        cluster_stats_str = cluster_stats.to_string()

        # 5. Chiamata a GPT
        prompt = f"""
        Sei un esperto di marketing. Analizza le medie dei seguenti 8 cluster clienti di una clinica veterinaria.
        Per ciascun cluster, scrivi:
        - un nome semplice (es. Cliente VIP)
        - una breve descrizione del comportamento del cliente
        - una strategia marketing suggerita

        Ecco i dati medi per cluster:

        {cluster_stats_str}
        """

        openai.api_key = os.environ.get("OPENAI_API_KEY")

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sei un esperto di marketing segmentato."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        gpt_output = response.choices[0].message.content

        # 6. Risposta API
        return jsonify({
            "cluster_clienti": json_output,
            "analisi_marketing": gpt_output
        })

    except Exception as e:
        return jsonify({"errore": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)

