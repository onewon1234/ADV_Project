from flask import Flask, render_template, request
import pandas as pd
from utils import get_top_images_by_tag

app = Flask(__name__)

# 1. 숙소 기획전용 데이터 로딩
df_clustered = pd.read_csv("clustered_marketing_texts.csv")  # cluster_id, marketing_text
df_data = pd.read_csv("기획전_최종선택클러스터_숙소.csv")         # cluster_id, description, picture_url 등 포함

cluster_title_map = {
    5:  "🏙️ 도심 한복판, 현지 감성 그대로 머물기",
    6:  "📍 어디든 가까워요! 입지 끝판왕 숙소 추천",
    18: "✨ 감성 톡톡! 넓고 럭셔리한 객실에서 호캉스",
    33: "🌿 가든뷰에서 피톤치드 한가득! 자연 속 힐링",
    40: "👨‍👩‍👧‍👦 가족 모두를 위한 평화로운 휴식처",
    46: "🌊 시선을 빼앗는 뷰맛집, 오션뷰 특가 모음",
    50: "⭐ 믿고 가는 후기 맛집! 신축 감성스테이 추천"
}
marketing_map = dict(zip(df_clustered["cluster_id"], df_clustered["marketing_text"]))

def suggest_campaign_title(cluster_id):
    return cluster_title_map.get(cluster_id, f"🏙️ 기획전 #{cluster_id}")

# 2. 해시태그 리스트
hashtags = [
    "a cozy bedroom", "a stylish living room", "a modern kitchen", "a clean bathroom",
    "a balcony with a view", "a relaxing rooftop", "a small studio apartment",
    "a modern interior", "a minimal design", "a cozy atmosphere", "a luxurious space",
    "a rustic cabin style", "a romantic room for two", "an industrial-style room",
    "a vintage-inspired room", "a room by the beach", "a room with a mountain view",
    "a room with a city view", "a room with a lake view", "a peaceful countryside home",
    "an urban apartment", "a family-friendly place", "a honeymoon getaway",
    "a space for solo travel", "a pet-friendly home", "a room for workation",
    "a house with a BBQ area", "a camping-themed room", "a room with a hot tub",
    "a cozy fireplace room", "a home theater with projector", "a bunk bed setup",
    "a room with large windows", "a home with wood floors", "a loft-style apartment",
    "a bright and airy room", "a room with warm lighting", "a room with high ceilings",
    "a space filled with natural light", "an open and spacious layout",
    "a clean and neat interior", "a spacious home", "a white-toned interior",
    "a room with dark wood", "a pastel-colored space", "an artistic interior",
    "a Scandinavian-style home", "a Japanese-style room"
]

# 3. 기존 기획전 홈 (루트 페이지는 이걸 유지)
@app.route("/")
def campaign_index():
    items = []
    valid_ids = set(df_data["cluster_id"].unique())

    for cid in cluster_title_map:
        if cid not in valid_ids:
            continue

        title = suggest_campaign_title(cid)
        items.append({
            "cluster_id": cid,
            "title": title
        })

    return render_template("index.html", items=items)

# 4. 클러스터별 상세 보기
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

# 5. 해시태그 추천 기능은 /hashtags 경로로 분리
@app.route("/hashtags")
def show_hashtag_input():
    return render_template("hashtag_input.html", tags=hashtags)

@app.route("/recommend", methods=["POST"])
def recommend():
    selected_tag = request.form["tag"]
    images = get_top_images_by_tag(selected_tag)  # utils.py의 함수
    return render_template("result.html", tag=selected_tag, images=images)

# 6. 실행
if __name__ == "__main__":
    app.run(debug=True)
