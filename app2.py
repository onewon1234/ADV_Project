from flask import Flask, render_template, request
from utils2 import recommend_similar_listings, create_map
import os
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

df_tags = pd.read_csv("data/clip최종df_이미지검색.csv")

hashtags = ["a family-friendly place",
    "a honeymoon getaway",
    "a space for solo travel",
    "a pet-friendly home",
    "a room for workation",
    "a house with a BBQ area",
    "a camping-themed room",
    "a room with a hot tub",
    "a cozy fireplace room",
    "a home theater with projector",
    "a bunk bed setup",
    "a room with large windows",
    "a home with wood floors",
    "a loft-style apartment",
    "a bright and airy room",
    "a room with warm lighting",
    "a room with high ceilings",
    "a space filled with natural light",
    "an open and spacious layout",
    "a clean and neat interior",
    "a spacious home",
    "a white-toned interior",
    "a room with dark wood",
    "a pastel-colored space",
    "an artistic interior",
    "a Scandinavian-style home",
    "a Japanese-style room"
]


device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text_inputs = processor(text=hashtags, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    hashtag_embs = model.get_text_features(**text_inputs)
    hashtag_embs = hashtag_embs / hashtag_embs.norm(dim=1, keepdim=True)

@app.route('/')
def index():
    return render_template("index2.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return "❗이미지를 업로드해 주세요."

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(image_path)

    top_tags, recommendations = recommend_similar_listings(
        image_path, df_tags, hashtags, hashtag_embs
    )

    create_map(recommendations)  # 🗺 지도 생성 추가

    return render_template(
        "result2.html",
        tags=top_tags,
        recommendations=recommendations.to_dict(orient='records')
    )

if __name__ == '__main__':
    app.run(debug=True)
