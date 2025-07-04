import pandas as pd

def get_top_images_by_tag(selected_tag, top_n=5):
    try:
        df = pd.read_csv("data/airbnb_tag_scores_top5.csv")
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다: data/airbnb_tag_scores_top5.csv")
        return []

    records = []
    for _, row in df.iterrows():
        for i in range(1, 6):
            tag_col = f"tag{i}"
            score_col = f"score{i}"
            if tag_col in row and row[tag_col] == selected_tag:
                score = row.get(score_col, 0)
                if pd.notna(score):  # NaN이 아닌 경우만 추가
                    records.append({
                        "picture_url": row.get("picture_url", "").strip(),
                        "score": score,
                        "id": row.get("id", "")
                    })

    sorted_records = sorted(records, key=lambda x: x["score"], reverse=True)
    return sorted_records[:top_n]
