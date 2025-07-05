import pandas as pd
import folium
import ast

def get_top_images_by_tag(selected_tag, top_n=6):
    df = pd.read_csv("data/clip최종df.csv")

    records = []
    for _, row in df.iterrows():
        for i in range(1, 3):
            tag_col = f"tag{i}"
            if row[tag_col] == selected_tag:
                hashtags = row.get("hashtags", "[]")
                try:
                    hashtags_list = ast.literal_eval(hashtags)
                except:
                    hashtags_list = []

                records.append({
                    "id": row["id"],
                    "picture_url": row["picture_url"],
                    "price": row.get("price", "N/A"),
                    "number_of_reviews": row.get("number_of_reviews", "N/A"),
                    "review_scores_rating": row.get("review_scores_rating", "N/A"),
                    "hashtags": hashtags_list,
                    "latitude": row.get("latitude", None),
                    "longitude": row.get("longitude", None)
                })
    return records[:top_n]

def create_map(recommendations, output_path="static/map.html"):
    # 중심 위치: 첫 번째 숙소로
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



