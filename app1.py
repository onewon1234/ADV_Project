from flask import Flask, render_template, request, url_for
from utils2 import recommend_similar_listings, create_map
import os
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

df_tags = pd.read_csv("data/clipfinaldf.csv")

# 해시태그 리스트 
hashtags = [
    "Modern",
    "Nordic",
    "Natural",
    "Vintage Retro",
    "Lovely Romantic",
    "Industrial",
    "Unique",
    "French Provence",
    "Minimal Simple",
    "Classic Antique",
    "Korean Asian"
]

def df_to_records_with_tag_dict(df):
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": row["id"],
            "name": row.get("name", "N/A"),
            "picture_url": row["picture_url"],
            "price": row.get("price", "N/A"),
            "number_of_reviews": row.get("number_of_reviews", "N/A"),
            "review_scores_rating": row.get("review_scores_rating", "N/A"),
            "latitude": row.get("latitude", None),
            "longitude": row.get("longitude", None),
            "listing_url": row.get("listing_url", "#"),
            "tag1": {"tag": row.get("tag1", "N/A")},
            "tag7": {"tag": row.get("tag7", "N/A")},
            "tag8": {"tag": row.get("tag8", "N/A")}
        })
    return records

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    selected_tag = request.args.get('selected_tag')
    if selected_tag:
        filtered = df_tags[df_tags['tag7'] == selected_tag].copy()
        if not filtered.empty:
            recommendations = filtered.sample(n=min(6, len(filtered)), random_state=None)
        else:
            recommendations = filtered.head(0)
        create_map(recommendations)
        return render_template(
            "result.html",
            tag=selected_tag,
            recommendations=df_to_records_with_tag_dict(recommendations)
        )

    if request.method == 'POST':
        selected_tag = request.form.get('tag')
        if not selected_tag:
            return "❗ 해시태그를 선택해 주세요."
        filtered = df_tags[df_tags['tag7'] == selected_tag].copy()
        if not filtered.empty:
            recommendations = filtered.sample(n=min(6, len(filtered)), random_state=None)
        else:
            recommendations = filtered.head(0)
        create_map(recommendations)
        return render_template("result.html", recommendations=df_to_records_with_tag_dict(recommendations), tag=selected_tag)

if __name__ == "__main__":
    app.run(debug=True)
