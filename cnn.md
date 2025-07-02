# ADV_Project
from google.colab import drive
drive.mount('/content/drive')
import os
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
from torchvision import models, transforms
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
dpath = '/content/drive/MyDrive/data/'
df = pd.read_csv(dpath + 'listings.csv')
sample_urls = df['picture_url'].dropna().tolist()
print(sample_urls[:5])
# 2. 이미지 다운로드
image_dir = "./airbnb_images"
os.makedirs(image_dir, exist_ok=True)
image_paths = []

for i, url in enumerate(sample_urls):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        path = os.path.join(image_dir, f"img_{i}.jpg")
        img.save(path)
        image_paths.append(path)
    except:
        continue
# 3. 이미지 전처리 및 모델 준비
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
# 4. 이미지 임베딩 생성
def extract_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor).squeeze().numpy()
    return embedding

embeddings = np.array([extract_embedding(p) for p in image_paths])
# 5. 기준 이미지와 유사한 이미지 찾기
knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn.fit(embeddings)

reference_index = 0  # 기준 이미지 인덱스
distances, indices = knn.kneighbors([embeddings[reference_index]])

# 6. 유사 이미지 출력
print("기준 이미지:", image_paths[reference_index])
print("유사한 이미지 5개:")
for i in indices[0][1:]:
    print(f"- {image_paths[i]}")
