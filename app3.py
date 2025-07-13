from flask import Flask, render_template, jsonify, request
import pandas as pd
import re

app = Flask(__name__)

@app.route('/')
def index():
    df = pd.read_csv('data/merged_host_최최최종.csv')
    print(df.columns.tolist())


    # 각 cluster_name(호스트 유형)별로 1명씩 랜덤하게 뽑기
    grouped_sample = (
        df.groupby('cluster_name', group_keys=False)
          .apply(lambda x: x.sample(1))
    )

    # 유형이 6개 이상인 경우만 6명 선택
    if len(grouped_sample) > 6:
        selected_hosts = grouped_sample.sample(n=6)
    else:
        selected_hosts = grouped_sample

    # summary 컬럼 가공 함수
    def clean_summary(text):
        if not isinstance(text, str):
            return ''
        # 첫글자 대문자
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        # 문장부호(.,?!) 앞 공백 제거
        text = re.sub(r'\s+([\.,?!])', r'\1', text)
        # 문장 끝에 문장부호 여러 개면 첫 번째만 남김
        text = re.sub(r'([\.,?!])([\.,?!]+)$', r'\1', text)
        # 온점(.) 뒤에 나오는 첫 글자 대문자
        def capitalize_after_dot(match):
            return match.group(1) + match.group(2).upper()
        text = re.sub(r'(\.\s*)([a-zA-Z가-힣])', capitalize_after_dot, text)
        return text

    # summary 컬럼 가공 적용
    selected_hosts['summary'] = selected_hosts['summary'].apply(clean_summary)

    # 리스트 형태로 전달
    hosts = selected_hosts.to_dict(orient='records')
    return render_template ('index.html', hosts=hosts)

@app.route('/refresh')
def refresh():
    df = pd.read_csv('data/merged_host_최최최종.csv')
    # 각 cluster_name(호스트 유형)별로 1명씩 랜덤하게 뽑기
    grouped_sample = (
        df.groupby('cluster_name', group_keys=False)
          .apply(lambda x: x.sample(1))
    )
    if len(grouped_sample) > 6:
        selected_hosts = grouped_sample.sample(n=6)
    else:
        selected_hosts = grouped_sample
    # summary 컬럼 가공 함수 (중복 정의 방지)
    def clean_summary(text):
        if not isinstance(text, str):
            return ''
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        text = re.sub(r'\s+([\.,?!])', r'\1', text)
        text = re.sub(r'([\.,?!])([\.,?!]+)$', r'\1', text)
        def capitalize_after_dot(match):
            return match.group(1) + match.group(2).upper()
        text = re.sub(r'(\.\s*)([a-zA-Z가-힣])', capitalize_after_dot, text)
        return text
    selected_hosts['summary'] = selected_hosts['summary'].apply(clean_summary)
    hosts = selected_hosts.to_dict(orient='records')
    return jsonify({'hosts': hosts})

@app.route('/host_swiper_partial', methods=['POST'])
def host_swiper_partial():
    hosts = request.get_json().get('hosts', [])
    return render_template('host_swiper.html', hosts=hosts)

if __name__ == '__main__':
    app.run(debug=True)



