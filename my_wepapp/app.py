from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# 1. 데이터 로딩
df_clustered = pd.read_csv("clustered_marketing_texts.csv")  # cluster_id, marketing_text
df_data = pd.read_csv("기획전_최종선택클러스터_숙소.csv")         # cluster_id, description, picture_url 등 포함

# 2. cluster_id → 고정된 기획전 제목 매핑
cluster_title_map = {
    5:  "🏙️ 도심 한복판, 현지 감성 그대로 머물기",
    6:  "📍 어디든 가까워요! 입지 끝판왕 숙소 추천",
    18: "✨ 감성 톡톡! 넓고 럭셔리한 객실에서 호캉스",
    33: "🌿 가든뷰에서 피톤치드 한가득! 자연 속 힐링",
    40: "👨‍👩‍👧‍👦 가족 모두를 위한 평화로운 휴식처",
    46: "🌊 시선을 빼앗는 뷰맛집, 오션뷰 특가 모음",
    50: "⭐ 믿고 가는 후기 맛집! 신축 감성스테이 추천"
}

# 3. cluster_id → marketing_text 매핑 (사용은 안 하지만 유지)
marketing_map = dict(zip(df_clustered["cluster_id"], df_clustered["marketing_text"]))

# 4. 기획전 제목 반환 함수
def suggest_campaign_title(cluster_id):
    return cluster_title_map.get(cluster_id, f"🏙️ 기획전 #{cluster_id}")

# 5. 인덱스 페이지: 기획전 목록
@app.route("/")
def index():
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

# 6. 상세 페이지: 클러스터별 숙소 보기
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

# 7. 실행
if __name__ == "__main__":
    app.run(debug=True)
