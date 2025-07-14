from flask import Flask, render_template
import pandas as pd
import os
import random

app = Flask(__name__)

# 🔹 통합 기획전 데이터
main_csv = "기획전_emotional_summaries_cleaned.csv"
df = pd.read_csv(main_csv)

# cluster_id와 marketing_text로 기획전 타이틀 매핑
def suggest_campaign_title(cluster_id):
    cluster_title_map = {
        row['cluster_id']: row['marketing_text']
        for _, row in df[['cluster_id', 'marketing_text']].drop_duplicates().iterrows()
    }
    return cluster_title_map.get(cluster_id, f"🏙️ 기획전 #{cluster_id}")

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
    filtered = df[df["cluster_id"] == cluster_id]
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
            "picture_url": row.get("picture_url", "").strip() if row.get("picture_url") else ""
        })
    # 타이틀(마케팅 문구)
    cluster_title_map = {
        row['cluster_id']: row['marketing_text']
        for _, row in df[['cluster_id', 'marketing_text']].drop_duplicates().iterrows()
    }
    marketing_text = cluster_title_map.get(cluster_id, f"기획전 #{cluster_id}")
    image_url = ""  # 필요시 이미지 URL 지정
    return render_template("cluster.html", title=title, marketing_text=marketing_text, subtitle=subtitle, image_url=image_url, items=items)

if __name__ == "__main__":
    app.run(debug=True) 