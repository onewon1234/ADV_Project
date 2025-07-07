from flask import Flask, render_template, request
from utils import recommend_similar_listings, create_map
import os
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel

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

# 🔹 CLIP 모델 준비
df_tags = pd.read_csv(os.path.join(os.path.dirname(__file__), "clip최종df.csv"), encoding="utf-8")
hashtags = [
    "Modern", "Nordic", "Natural", "Vintage Retro", "Lovely Romantic",
    "Industrial", "Unique", "French Provence", "Minimal Simple",
    "Classic Antique", "Korean Asian"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_inputs = processor(text=hashtags, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    hashtag_embs = model.get_text_features(**text_inputs)
    hashtag_embs = hashtag_embs / hashtag_embs.norm(dim=1, keepdim=True)

# ✅ 메인 페이지
@app.route('/')
def index():
    valid_ids = set(df_data["cluster_id"].unique())

    # 기존 감성 기획전 (기존 구조 그대로 유지)
    campaign_items = [
        {"cluster_id": cid, "title": suggest_campaign_title(cid)}
        for cid in cluster_title_map if cid in valid_ids
    ]

    # ✅ 특가 기획전만 추가 (url 필드 포함)
    campaign_items.append({
        "title": "💸 부담없이 떠나세요! 5만원 이하의 갓성비 숙소",
        "url": "/cheap"
    })

    return render_template("index.html", items=campaign_items)


# ✅ 기획전 상세 페이지
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
    # CSV 파일에서 특가 숙소 데이터 불러오기
    df_cheap = pd.read_csv("airbnb_cheap_under_35.csv")

    # 기획전 제목
    title = "💸 5만원 이하 갓성비 숙소 모음"

    # template에 넘겨줄 숙소 리스트
    items = [
        {
            "description": row.get("description", ""),
            "picture_url": row.get("picture_url", "").strip() if row.get("picture_url") else ""
        }
        for _, row in df_cheap.iterrows()
    ]

    # cluster.html 템플릿 재사용
    return render_template("cluster.html", title=title, items=items)


# ✅ 이미지 업로드 기반 추천
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

    create_map(recommendations)  # 지도 생성 (map.html)
    return render_template("result.html", tags=top_tags, recommendations=recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
