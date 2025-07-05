from flask import Flask, render_template, request
from utils import get_top_images_by_tag, create_map
import os

app = Flask(__name__)

# 해시태그 리스트 
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

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_tag = request.form.get('tag')
    if not selected_tag:
        return "❗ 해시태그를 선택해 주세요."

    recommendations = get_top_images_by_tag(selected_tag, top_n=6)
    create_map(recommendations, output_path="static/map.html")  # 지도 생성

    return render_template("result.html", recommendations=recommendations, tag=selected_tag)

if __name__ == "__main__":
    app.run(debug=True)

