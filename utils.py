import pandas as pd
import folium
import ast
import random  # ✅ 추가

def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0


def get_top_images_by_tag(selected_tag, top_n=6):
    df = pd.read_csv("data/clipfinaldf.csv")

    # tag7이 선택한 태그와 같은 행만 필터링
    filtered = df[df['tag7'] == selected_tag].copy()

    # 랜덤하게 top_n개 추출
    if not filtered.empty:
        filtered = filtered.sample(n=min(top_n, len(filtered)), random_state=None)

    records = []
    for _, row in filtered.iterrows():
        hashtags = row.get("hashtags", "[]")
        if hashtags is None:
            hashtags = "[]"
        try:
            hashtags_list = ast.literal_eval(hashtags)
        except:
            hashtags_list = []
        records.append({
            "id": row["id"],
            "name": row.get("name", "N/A"),
            "picture_url": row["picture_url"],
            "price": row.get("price", "N/A"),
            "number_of_reviews": row.get("number_of_reviews", "N/A"),
            "review_scores_rating": row.get("review_scores_rating", "N/A"),
            "hashtags": hashtags_list,
            "latitude": row.get("latitude", None),
            "longitude": row.get("longitude", None),
            "listing_url": row.get("listing_url", "#"),
            "tag1": {"tag": row.get("tag1", "N/A"), "score": safe_float(row.get("score1", 0.0))},
            "tag7": {"tag": row.get("tag7", "N/A"), "score": safe_float(row.get("score7", 0.0))},
            "tag8": {"tag": row.get("tag8", "N/A"), "score": safe_float(row.get("score8", 0.0))}
        })
    return records


def create_map(recommendations, output_path="static/map.html"):
    if not recommendations:
        # 빈 지도라도 생성 (서울 기준)
        m = folium.Map(location=[37.5665, 126.9780], zoom_start=12)
        m.save(output_path)
        return

    start_lat = recommendations[0]["latitude"]
    start_lon = recommendations[0]["longitude"]
    m = folium.Map(location=[start_lat, start_lon], zoom_start=12)

    for rec in recommendations:
        if pd.notna(rec["latitude"]) and pd.notna(rec["longitude"]):
            popup_text = f"Price: {rec['price']}<br>Rating: {rec['review_scores_rating']}"
            folium.Marker(
                location=[rec["latitude"], rec["longitude"]],
                popup=popup_text,
                tooltip="Click for info"
            ).add_to(m)

    m.save(output_path)



