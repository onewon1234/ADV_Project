# map.html에 result_image.html 추가해서 렌더링하는 구조
[ index.html ]
   ↓ 사용자가 이미지 업로드
POST → /recommend
   ↓ Flask가 추천 계산 + map.html 생성
렌더링 → result_image.html (추천 리스트 + 지도 iframe 포함)
   ↓
iframe 내부에서 static/map.html 자동 로딩됨
