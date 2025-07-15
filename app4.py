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

# cluster_id별 배너 이미지 매핑
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
    banner_image = get_banner_image(cluster_id)
    return render_template("cluster.html", title=title, marketing_text=marketing_text, subtitle=subtitle, banner_image=banner_image, items=items)

if __name__ == "__main__":
    app.run(debug=True) 