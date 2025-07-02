from flask import Flask, render_template, request
from utils import get_top_images_by_tag

app = Flask(__name__)

# 해시태그 리스트 (너가 준 리스트)
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

@app.route("/")
def index():
    return render_template("index.html", tags=hashtags)

@app.route("/recommend", methods=["POST"])
def recommend():
    selected_tag = request.form["tag"]
    images = get_top_images_by_tag(selected_tag)  # utils.py의 함수
    return render_template("result.html", tag=selected_tag, images=images)

if __name__ == "__main__":
    app.run(debug=True)

