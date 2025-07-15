from flask import Flask, render_template, request, session
import os
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from utils import get_top_images_by_tag, create_map as create_tag_map
from utils2 import recommend_similar_listings, create_map as create_image_map
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key-here'  # ✅ 세션을 위한 시크릿 키 추가
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 🔹 통합 기획전 데이터
# 컬럼: cluster_id, marketing_text, emotional_summary, name, ratings, price, picture_url 등
main_csv = "기획전_emotional_summaries_cleaned.csv"
df = pd.read_csv(main_csv)

# cluster_id와 marketing_text로 기획전 타이틀 매핑
cluster_title_map = {
    row['cluster_id']: row['marketing_text']
    for _, row in df[['cluster_id', 'marketing_text']].drop_duplicates().iterrows()
}

def suggest_campaign_title(cluster_id):
    return cluster_title_map.get(cluster_id, f"🏙️ 기획전 #{cluster_id}")

# ✅ 메인 페이지 (기획전 리스트)
@app.route('/')
def index():
    campaign_items = [
        {"cluster_id": cid, "title": suggest_campaign_title(cid)}
        for cid in sorted(df['cluster_id'].unique())
    ]
    return render_template("index.html", items=campaign_items)

# ✅ 기획전 상세 페이지
@app.route("/cluster/<int:cluster_id>")
def show_cluster(cluster_id):
    title = suggest_campaign_title(cluster_id)
    filtered = df[df["cluster_id"] == cluster_id]
    
    # ✅ URL 파라미터로 랜덤 시드 관리 (새로고침할 때마다 바뀜)
    import time
    current_time = int(time.time())
    random_seed = current_time % 10000  # 현재 시간을 기반으로 시드 생성
    
    # ✅ 최대 30개로 제한 (깜빡임 방지)
    if len(filtered) > 30:
        # 현재 시간 기반으로 랜덤 섞기 후 상위 30개 선택
        filtered = filtered.sample(frac=1, random_state=random_seed).head(30).reset_index(drop=True)
    else:
        # 30개 이하면 전체 사용
        filtered = filtered.reset_index(drop=True)
    
    # ✅ 이미지 비율에 따라 분류
    tall_images = []  # 세로가 긴 이미지들
    normal_images = []  # 일반 이미지들
    
    for _, row in filtered.iterrows():
        item = {
            "name": row.get("name", ""),
            "ratings": row.get("review_scores_rating", ""),  
            "price": row.get("price", ""),
            "emotional_summary": row.get("emotional_summary", ""),
            "picture_url": row.get("picture_url", "").strip() if row.get("picture_url") else ""
        }
        
        # 이미지 URL이 있으면 tall_images에 추가 (세로가 긴 것으로 가정)
        if item["picture_url"]:
            tall_images.append(item)
        else:
            normal_images.append(item)
    
    # tall_images를 먼저, 그 다음 normal_images 순서로 배치
    items = tall_images + normal_images
    
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
df_tags = pd.read_csv("clipfinaldf.csv")
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
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                  use_safetensors=True).to(device)
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
    app.run(host="0.0.0.0", port=8000, debug=True)  # ✅ 디버그 모드 활성화
