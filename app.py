<<<<<<< HEAD
from flask import Flask, render_template, request, url_for, jsonify, redirect
=======
from flask import Flask, render_template, request, session
>>>>>>> 예림
import os
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
<<<<<<< HEAD
import random
import re
from utils2 import recommend_similar_listings, create_map

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 데이터 로딩
main_csv = "기획전_emotional_summaries_cleaned.csv"
df_campaign = pd.read_csv(main_csv)
df_tags = pd.read_csv("clipfinaldf.csv")
try:
    df_hosts = pd.read_csv('merged_host_최최최종.csv')
except:
    df_hosts = None

# 해시태그 리스트 (tag 기반 추천)
hashtags = [
    "Modern", "Nordic", "Natural", "Vintage Retro", "Lovely Romantic",
    "Industrial", "Unique", "French Provence", "Minimal Simple",
    "Classic Antique", "Korean Asian"
]
# 이미지 기반 추천 해시태그
clip_hashtags = [
    "a family-friendly place", "a honeymoon getaway", "a space for solo travel",
    "a pet-friendly home", "a room for workation", "a house with a BBQ area",
    "a camping-themed room", "a room with a hot tub", "a cozy fireplace room",
    "a home theater with projector", "a bunk bed setup", "a room with large windows",
    "a home with wood floors", "a loft-style apartment", "a bright and airy room",
    "a room with warm lighting", "a room with high ceilings", "a space filled with natural light",
    "an open and spacious layout", "a clean and neat interior", "a spacious home",
    "a white-toned interior", "a room with dark wood", "a pastel-colored space",
    "an artistic interior", "a Scandinavian-style home", "a Japanese-style room"
]

# CLIP 모델 준비 (이미지 기반 추천)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_inputs = processor(text=clip_hashtags, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    hashtag_embs = model.get_text_features(
        input_ids=text_inputs["input_ids"].to(device),
        attention_mask=text_inputs["attention_mask"].to(device)
    )
    hashtag_embs = hashtag_embs / hashtag_embs.norm(dim=1, keepdim=True)

def df_to_records_with_tag_dict(df):
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": row.get("id", "N/A"),
            "name": row.get("name", "N/A"),
            "picture_url": row.get("picture_url", ""),
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

def suggest_campaign_title(cluster_id):
    cluster_title_map = {
        row['cluster_id']: row['marketing_text']
        for _, row in df_campaign[['cluster_id', 'marketing_text']].drop_duplicates().iterrows()
    }
    return cluster_title_map.get(cluster_id, f"🏙️ 기획전 #{cluster_id}")

def get_banner_image(cluster_id):
    banner_images = {
        1: "/static/image/가성비숙소.jpg",
        5: "https://readdy.ai/api/search-image?query=a%20modern%20minimalist%20apartment%20with%20large%20windows%2C%20city%20skyline%20view%2C%20sleek%20furniture%2C%20neutral%20color%20palette%2C%20perfect%20for%20urban%20professionals%2C%20high-rise%20building&width=600&height=400&seq=2&orientation=landscape",
        6: "/static/image/입지.ong",
        18: "/static/image/럭셔리.png",
        33: "https://readdy.ai/api/search-image?query=a%20unique%20treehouse%20accommodation%20in%20a%20forest%2C%20wooden%20structure%2C%20large%20windows%2C%20cozy%20interior%20with%20fairy%20lights%2C%20hammock%2C%20surrounded%20by%20tall%20trees%2C%20magical%20atmosphere&width=600&height=400&seq=5&orientation=landscape",
        40: "/static/image/가족.jpg",
        46: "https://readdy.ai/api/search-image?query=a%20beachfront%20villa%20with%20infinity%20pool%2C%20ocean%20view%2C%20palm%20trees%2C%20white%20sandy%20beach%2C%20luxury%20outdoor%20furniture%2C%20sunset%20lighting%2C%20tropical%20paradise&width=600&height=400&seq=3&orientation=landscape",
        50: "/static/image/신축.jpg"
    }
    return banner_images.get(cluster_id, "")

# 메인 페이지
@app.route('/')
def index():
    campaign_items = [
        {"cluster_id": cid, "title": suggest_campaign_title(cid), "banner_image": get_banner_image(cid)}
        for cid in sorted(df_campaign['cluster_id'].unique())
    ]
    # 호스트 카드용 데이터 미리 뽑기
    if df_hosts is not None:
        grouped_sample = (
            df_hosts.groupby('cluster_name', group_keys=False)
              .apply(lambda x: x.sample(1))
        )
        if len(grouped_sample) > 6:
            selected_hosts = grouped_sample.sample(n=6)
        else:
            selected_hosts = grouped_sample
        hosts = []
        for _, row in selected_hosts.iterrows():
            host = row.to_dict()
            hashtags = host.get('hashtags', '')
            if isinstance(hashtags, str):
                hashtags = [tag.strip() for tag in hashtags.split(',') if tag.strip()]
            elif not isinstance(hashtags, list):
                hashtags = []
            host['hashtags'] = hashtags
            hosts.append(host)
    else:
        hosts = []
    hashtags = [
        "Nordic", "Natural", "Vintage Retro", "Lovely Romantic", "Industrial",
        "Unique", "French Provence", "Minimal Simple", "Classic Antique", "Korean"
    ]
    return render_template("index.html", items=campaign_items, hosts=hosts, hashtags=hashtags)

# /index.html로 접근해도 index.html이 렌더링되도록 라우트 추가
@app.route('/index.html')
def index_html():
    return render_template("index.html")

# 기획전 상세 페이지
@app.route('/cluster/<int:cluster_id>')
def show_cluster(cluster_id):
    subtitle_map = {
        1: "가성비와 감성을 모두 잡은 최고의 선택!",
        5: "도심 속에서 만나는 이국적인 감성, 특별한 하루를 경험하세요",
        6: "교통이 편리한 최고의 위치, 여행이 더 쉬워집니다",
        18: "럭셔리와 감성의 완벽한 조화, 프라이빗한 휴식",
        33: "자연과 함께하는 힐링 타임, 몸과 마음이 쉬어갑니다",
        40: "아이부터 어른까지 모두가 행복한 공간",
        46: "탁 트인 오션뷰와 함께하는 특별한 하루",
        50: "후기로 검증된 감성 신축 숙소만 모았어요"
    }
    subtitle = subtitle_map.get(cluster_id, "이 기획전만의 특별한 감성을 느껴보세요!")
    title = suggest_campaign_title(cluster_id)
    filtered = df_campaign[df_campaign["cluster_id"] == cluster_id]
    if len(filtered) > 36:
        filtered = filtered.sample(n=36, random_state=random.randint(0, 10000)).reset_index(drop=True)
    else:
        filtered = filtered.reset_index(drop=True)
    items = []
    for _, row in filtered.iterrows():
        items.append({
            "name": row.get("name", ""),
            "ratings": row.get("review_scores_rating", ""),  
            "price": row.get("price", ""),
            "emotional_summary": row.get("emotional_summary", ""),
            "picture_url": row.get("picture_url", "").strip() if row.get("picture_url") else "",
            "listing_url": row.get("listing_url", "#")
        })
    marketing_text = title
    banner_image = get_banner_image(cluster_id)
    return render_template("cluster.html", title=title, marketing_text=marketing_text, subtitle=subtitle, banner_image=banner_image, items=items)

# 해시태그 기반 추천
@app.route('/tag-recommend', methods=['GET', 'POST'])
def tag_recommend():
    if request.method == 'POST':
        selected_tag = request.form.get('tag')
    else:  # GET 방식
        selected_tag = request.args.get('selected_tag')
    if not selected_tag:
        return "❗ 해시태그를 선택해 주세요."
=======
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
    return render_template("index.html", items=campaign_items, hashtags=hashtags)

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
df_tags = pd.read_csv("clipfinaldf.csv")

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

@app.route('/tag-recommend', methods=['GET', 'POST'])
def tag_recommend():
    if request.method == 'GET':
        return render_template("tag_index.html", hashtags=hashtags)
    
    selected_tag = request.form.get('tag')
    if not selected_tag:
        return "❗ 해시태그를 선택해 주세요."
    
>>>>>>> 예림
    filtered = df_tags[df_tags['tag7'] == selected_tag].copy()
    if not filtered.empty:
        recommendations = filtered.sample(n=min(6, len(filtered)), random_state=None)
    else:
        recommendations = filtered.head(0)
<<<<<<< HEAD
    create_map(recommendations)
    return render_template("result.html", recommendations=df_to_records_with_tag_dict(recommendations), tag=selected_tag)

# 이미지 업로드 기반 추천
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
    create_map(recommendations)
    user_image_url = url_for('static', filename='uploads/' + uploaded_file.filename)
    return render_template(
        "result2.html",
        tags=top_tags,
        recommendations=recommendations.to_dict(orient='records'),
        user_image_url=user_image_url
    )

# 호스트 스와이퍼
@app.route('/host-swiper')
def host_swiper():
    if df_hosts is None:
        return "호스트 데이터가 없습니다."
    grouped_sample = (
        df_hosts.groupby('cluster_name', group_keys=False)
          .apply(lambda x: x.sample(1))
    )
    if len(grouped_sample) > 6:
        selected_hosts = grouped_sample.sample(n=6)
    else:
        selected_hosts = grouped_sample
    def clean_summary(text):
        if not isinstance(text, str):
            return ''
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        text = re.sub(r'\s+([\.,?!])', r'\1', text)
        text = re.sub(r'([\.,?!])([\.,?!]+)$', r'\1', text)
        def capitalize_after_dot(match):
            return match.group(1) + match.group(2).upper()
        text = re.sub(r'(\.\s*)([a-zA-Z가-힣])', capitalize_after_dot, text)
        return text
    selected_hosts['summary'] = selected_hosts['summary'].apply(clean_summary)
    hosts = selected_hosts.to_dict(orient='records')
    return render_template('host_swiper.html', hosts=hosts)

@app.route('/refresh')
def refresh():
    if df_hosts is None:
        return jsonify({'hosts': []})
    grouped_sample = (
        df_hosts.groupby('cluster_name', group_keys=False)
          .apply(lambda x: x.sample(1))
    )
    if len(grouped_sample) > 6:
        selected_hosts = grouped_sample.sample(n=6)
    else:
        selected_hosts = grouped_sample
    def clean_summary(text):
        if not isinstance(text, str):
            return ''
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        text = re.sub(r'\s+([\.,?!])', r'\1', text)
        text = re.sub(r'([\.,?!])([\.,?!]+)$', r'\1', text)
        def capitalize_after_dot(match):
            return match.group(1) + match.group(2).upper()
        text = re.sub(r'(\.\s*)([a-zA-Z가-힣])', capitalize_after_dot, text)
        return text
    selected_hosts['summary'] = selected_hosts['summary'].apply(clean_summary)
    hosts = selected_hosts.to_dict(orient='records')
    return jsonify({'hosts': hosts})

@app.route('/host_swiper_partial', methods=['POST'])
def host_swiper_partial():
    hosts = request.get_json().get('hosts', [])
    return render_template('host_swiper.html', hosts=hosts)

# result2 페이지 라우트 추가 (이미 있으면 생략)
@app.route('/result2')
def result2():
    tags = []
    return render_template("result2.html", tags=tags)

# result 페이지 라우트 추가 (이미 있으면 생략)
@app.route('/result')
def result():
    return render_template("result.html")
=======
    
    create_tag_map(recommendations, output_path="static/map.html")
    return render_template("result.html", recommendations=df_to_records_with_tag_dict(recommendations), tag=selected_tag)

# ✅ 이미지 업로드 기반 추천 (app2.py)
df_tags = pd.read_csv("clipfinaldf.csv")

clip_hashtags = ["a family-friendly place",
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
>>>>>>> 예림

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

    create_image_map(recommendations)  # 🗺 지도 생성 추가

    # 업로드한 이미지의 static 경로 생성
    user_image_url = url_for('static', filename='uploads/' + uploaded_file.filename)

    return render_template(
        "result2.html",
        tags=top_tags,
        recommendations=recommendations.to_dict(orient='records'),
        user_image_url=user_image_url
    )

# ✅ 호스트 스와이퍼 섹션 (app3.py)
import re

@app.route('/host-swiper')
def host_swiper():
    df = pd.read_csv('merged_host_최최최종.csv')
    print(df.columns.tolist())

    # 각 cluster_name(호스트 유형)별로 1명씩 랜덤하게 뽑기
    grouped_sample = (
        df.groupby('cluster_name', group_keys=False)
          .apply(lambda x: x.sample(1))
    )

    # 유형이 6개 이상인 경우만 6명 선택
    if len(grouped_sample) > 6:
        selected_hosts = grouped_sample.sample(n=6)
    else:
        selected_hosts = grouped_sample

    # summary 컬럼 가공 함수
    def clean_summary(text):
        if not isinstance(text, str):
            return ''
        # 첫글자 대문자
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        # 문장부호(.,?!) 앞 공백 제거
        text = re.sub(r'\s+([\.,?!])', r'\1', text)
        # 문장 끝에 문장부호 여러 개면 첫 번째만 남김
        text = re.sub(r'([\.,?!])([\.,?!]+)$', r'\1', text)
        # 온점(.) 뒤에 나오는 첫 글자 대문자
        def capitalize_after_dot(match):
            return match.group(1) + match.group(2).upper()
        text = re.sub(r'(\.\s*)([a-zA-Z가-힣])', capitalize_after_dot, text)
        return text

    # summary 컬럼 가공 적용
    selected_hosts['summary'] = selected_hosts['summary'].apply(clean_summary)

    # 리스트 형태로 전달
    hosts = selected_hosts.to_dict(orient='records')
    return render_template('host_swiper.html', hosts=hosts)

@app.route('/refresh')
def refresh():
    df = pd.read_csv('merged_host_최최최종.csv')
    # 각 cluster_name(호스트 유형)별로 1명씩 랜덤하게 뽑기
    grouped_sample = (
        df.groupby('cluster_name', group_keys=False)
          .apply(lambda x: x.sample(1))
    )
    if len(grouped_sample) > 6:
        selected_hosts = grouped_sample.sample(n=6)
    else:
        selected_hosts = grouped_sample
    # summary 컬럼 가공 함수 (중복 정의 방지)
    def clean_summary(text):
        if not isinstance(text, str):
            return ''
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        text = re.sub(r'\s+([\.,?!])', r'\1', text)
        text = re.sub(r'([\.,?!])([\.,?!]+)$', r'\1', text)
        def capitalize_after_dot(match):
            return match.group(1) + match.group(2).upper()
        text = re.sub(r'(\.\s*)([a-zA-Z가-힣])', capitalize_after_dot, text)
        return text
    selected_hosts['summary'] = selected_hosts['summary'].apply(clean_summary)
    hosts = selected_hosts.to_dict(orient='records')
    return jsonify({'hosts': hosts})

@app.route('/host_swiper_partial', methods=['POST'])
def host_swiper_partial():
    hosts = request.get_json().get('hosts', [])
    return render_template('host_swiper.html', hosts=hosts)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)  # ✅ 디버그 모드 활성화
