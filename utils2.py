import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import folium
import ast  

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 및 프로세서 불러오기
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 해시태그 임베딩 계산 함수
def compute_hashtag_embeddings(hashtags):
    text_inputs = processor(text=hashtags, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        hashtag_embs = model.get_text_features(**text_inputs)
        hashtag_embs = hashtag_embs / hashtag_embs.norm(dim=1, keepdim=True)
    return hashtag_embs

# 추천 함수
def recommend_similar_listings(uploaded_image_path, df_tags, hashtags, hashtag_embs, top_k=6):
    # 1. 이미지 임베딩 추출
    image = Image.open(uploaded_image_path).convert("RGB")
    image_input = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(**image_input)
        image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)

    # 2. 해시태그 예측
    similarities = torch.matmul(image_emb, hashtag_embs.T).squeeze()
    top_indices = similarities.topk(top_k).indices.cpu().numpy()
    top_tags = [hashtags[i] for i in top_indices]

    # 3. 각 숙소와의 유사도 점수 계산
    df_candidates = df_tags.copy()
    scores = []
    hashtags_list = []
    for _, row in df_candidates.iterrows():
        # tag1, tag7, tag8 사용
        row_tags = [row.get("tag1", ""), row.get("tag7", ""), row.get("tag8", "")]
        row_scores = [row.get("score1", 0.0), row.get("score7", 0.0), row.get("score8", 0.0)]
        matched_scores = [row_scores[i] for i in range(len(row_tags)) if row_tags[i] in top_tags]

        raw_hashtags = row.get("hashtags", "[]")
        try:
            parsed_hashtags = ast.literal_eval(raw_hashtags)
        except:
            parsed_hashtags = []

        scores.append(np.mean(matched_scores) if matched_scores else 0)
        hashtags_list.append(parsed_hashtags) 

    df_candidates["match_score"] = scores
    df_candidates["hashtags"] = hashtags_list

    # 4. 상위 숙소 추천 (top_k개)
    top_matches = df_candidates.sort_values(by="match_score", ascending=False).head(top_k)

    return top_tags, top_matches[
        ["id", "name", "picture_url", "price", "number_of_reviews", "review_scores_rating", 
         "hashtags", "latitude", "longitude", "listing_url", "tag1", "tag7", "tag8"]
    ]

# 지도 생성 함수
def create_map(recommendations):
    # 첫 번째 숙소의 위치를 지도 중심으로 사용
    if not recommendations.empty:
        center_lat = recommendations.iloc[0]["latitude"]
        center_lon = recommendations.iloc[0]["longitude"]
    else:
        center_lat = 37.5665  # 서울 기본값
        center_lon = 126.9780
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    for _, row in recommendations.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            tooltip=f"{row['name']} | {row['price']}원 | 평점: {row['review_scores_rating']}"
        ).add_to(m)

    m.save("static/map.html")
