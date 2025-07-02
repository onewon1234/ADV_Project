# utils.py

import pandas as pd

def get_top_images_by_tag(selected_tag, top_n=5):
    # CSV 파일을 함수 내부에서 불러오기
    df = pd.read_csv("data/airbnb_tag_scores_top5.csv")  # 경로는 실제 위치에 맞게 수정!

    records = []
    for _, row in df.iterrows():
        for i in range(1, 6):  # tag1 ~ tag5
            tag_col = f"tag{i}"
            score_col = f"score{i}"
            if row[tag_col] == selected_tag:
                records.append({
                    "picture_url": row["picture_url"],
                    "score": row[score_col],
                    "id": row["id"]
                })
    
    sorted_records = sorted(records, key=lambda x: x["score"], reverse=True)[:top_n]
    return sorted_records


