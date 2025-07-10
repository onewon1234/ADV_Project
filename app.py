from flask import Flask, render_template, request
import os
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from utils import get_top_images_by_tag, create_map as create_tag_map
from utils2 import recommend_similar_listings, create_map as create_image_map

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 🔹 기획전용 데이터
df_clustered = pd.read_csv("clustered_marketing_texts.csv")
df_data = pd.read_csv("기획전_최종선택클러스터_숙소.csv")

cluster_title_map = {
    5:  "🏙️ 도심 한복판, 현지 감성 그대로 머물기",
    6:  "📍 어디든 가까워요! 입지 끝판왕 숙소 추천",
    18: "✨ 감성 톡톡! 넓고 럭셔리한 객실에서 호캉스",
    33: "🌿 가든뷰에서 피톤치드 한가득! 자연 속 힐링",
    40: "👨‍👩‍👧‍👦 가족 모두를 위한 평화로운 휴식처",
    46: "🌊 시선을 빼앗는 뷰맛집, 오션뷰 특가 모음",
    50: "⭐ 믿고 가는 후기 맛집! 신축 감성스테이 추천"
}

def suggest_campaign_title(cluster_id):
    return cluster_title_map.get(cluster_id, f"🏙️ 기획전 #{cluster_id}")

# ✅ 메인 페이지 (기획전)
@app.route('/')
def index():
    valid_ids = set(df_data["cluster_id"].unique())
    campaign_items = [
        {"cluster_id": cid, "title": suggest_campaign_title(cid)}
        for cid in cluster_title_map if cid in valid_ids
    ]
    campaign_items.append({
        "title": "💸 부담없이 떠나세요! 5만원 이하의 갓성비 숙소",
        "url": "/cheap"
    })
    return render_template("index.html", items=campaign_items)

@app.route("/cluster/<int:cluster_id>")
def show_cluster(cluster_id):
    title = suggest_campaign_title(cluster_id)
    filtered = df_data[df_data["cluster_id"] == cluster_id]
    items = [
        {
            "description": row.get("description", ""),
            "picture_url": row.get("picture_url", "").strip() if row.get("picture_url") else ""
        }
        for _, row in filtered.iterrows()
    ]
    return render_template("cluster.html", title=title, items=items)

@app.route("/cheap")
def cheap_special():
    df_cheap = pd.read_csv("airbnb_cheap_under_35.csv")
    title = "💸 5만원 이하 갓성비 숙소 모음"
    items = [
        {
            "description": row.get("description", ""),
            "picture_url": row.get("picture_url", "").strip() if row.get("picture_url") else ""
        }
        for _, row in df_cheap.iterrows()
    ]
    return render_template("cluster.html", title=title, items=items)

# ✅ 해시태그 기반 추천 (app1.py)
hashtags = [
    "Modern", "Nordic", "Natural", "Vintage Retro", "Lovely Romantic",
    "Industrial", "Unique", "French Provence", "Minimal Simple",
    "Classic Antique", "Korean Asian"
]

@app.route('/tag-recommend', methods=['GET', 'POST'])
def tag_recommend():
    if request.method == 'GET':
        return render_template("tag_index.html", hashtags=hashtags)
    selected_tag = request.form.get('tag')
    if not selected_tag:
        return "❗ 해시태그를 선택해 주세요."
    recommendations = get_top_images_by_tag(selected_tag, top_n=6)
    create_tag_map(recommendations, output_path="static/map.html")
    return render_template("result.html", recommendations=recommendations, tag=selected_tag)

# ✅ 이미지 업로드 기반 추천 (app2.py)
df_tags = pd.read_csv("data/clip최종df_이미지검색.csv")
clip_hashtags = [
    "a family-friendly place", "a honeymoon getaway", "a space for solo travel", "a pet-friendly home",
    "a room for workation", "a house with a BBQ area", "a camping-themed room", "a room with a hot tub",
    "a cozy fireplace room", "a home theater with projector", "a bunk bed setup", "a room with large windows",
    "a home with wood floors", "a loft-style apartment", "a bright and airy room", "a room with warm lighting",
    "a room with high ceilings", "a space filled with natural light", "an open and spacious layout",
    "a clean and neat interior", "a spacious home", "a white-toned interior", "a room with dark wood",
    "a pastel-colored space", "an artistic interior", "a Scandinavian-style home", "a Japanese-style room"
]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_inputs = processor(text=clip_hashtags, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    hashtag_embs = model.get_text_features(**text_inputs)
    hashtag_embs = hashtag_embs / hashtag_embs.norm(dim=1, keepdim=True)

@app.route('/image-recommend', methods=['GET', 'POST'])
def image_recommend():
    if request.method == 'GET':
        return render_template("image_index.html")
    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return "❗이미지를 업로드해 주세요."
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(image_path)
    top_tags, recommendations = recommend_similar_listings(
        image_path, df_tags, clip_hashtags, hashtag_embs
    )
    create_image_map(recommendations)  # 지도 생성
    return render_template(
        "result2.html",
        tags=top_tags,
        recommendations=recommendations.to_dict(orient='records')
    )

if __name__ == '__main__':
    app.run(debug=True)
