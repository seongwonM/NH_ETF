

# MS의 azure open ai를 이용하여 구현한 큐레이션 서비스 코드입니다.
# 오류명 : " (pyodbc.OperationalError) ('08001', '[08001] [Microsoft][ODBC Driver 18 for SQL Server]TCP 공급자: 시간 초과 오류입니다. " 
# 해당 오류는 랜덤하게 나타나는 단순 시간 초과오류로, 해결할 방법이 없습니다...그냥 다시 이 파일을 실행해주시면 됩니다.  

import pandas as pd
import yfinance as yf
import openai
import pyodbc
import io
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import numpy as np
import requests
import faiss
import json
import time
import tempfile


# Azure SQL Database 연결
from sqlalchemy import create_engine

conn_str = (
    'mssql+pyodbc://jyun22:1emdrkwk%21@jy-nh-server01.database.windows.net:1433/NH-SQLDB-001?'
    'driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes&ConnectionTimeout=300'
)
engine = create_engine(conn_str)

# 전역적으로 conn 정의
conn = engine.connect() 

# Azure OpenAI API 설정
openai.api_type = "azure"
openai.api_base = "https://commend01.openai.azure.com/"  # 엔드포인트 URL
openai.api_version = "2023-03-15-preview"  # API 버전
openai.api_key = ""  # API 키

# 23-1. index,( 유사도 모델의 index호출 )
# Azure Blob Storage URL에서 임시 파일로 다운로드하는 함수
def download_to_tempfile(url):
    response = requests.get(url)
    response.raise_for_status()
    # 임시 파일 생성
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(response.content)
    temp_file.flush()  # 모든 내용을 작성한 후 저장
    return temp_file

# 리소스를 로드하는 함수
def load_resources(index_url):
    # FAISS 인덱스를 임시 파일로 다운로드하고 로드
    with download_to_tempfile(index_url) as index_file:
        index = faiss.read_index(index_file.name)

    return index

# 사용 예시 - 각 URL을 입력
index_url = "https://nh02.blob.core.windows.net/index/faiss_index.bin"

index = load_resources(index_url)

# 23-2 model
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 23-3. documents.json 불러오기
# Azure Blob Storage URL에서 JSON 파일 다운로드 및 로드
def load_documents_from_json_url(url):
    response = requests.get(url)
    response.raise_for_status()
    
    # JSON 파일을 메모리 상에서 로드하여 파싱
    documents = response.json()  # JSON 파일 내용을 파싱하여 Python 객체로 변환
    return documents

# 사용 예시 - documents URL을 Azure Blob의 JSON 파일 URL로 지정
documents_url = "https://nh02.blob.core.windows.net/documents/documents.json"
documents = load_documents_from_json_url(documents_url)

# 23-4. 어투 데이터 불러오기 및 열 이름 변경 함수 ( 앞에 고객분류 결과를 바탕으로 제작 )
def load_and_rename_data():
    # SQL 쿼리로 데이터 불러오기
    query = """
    SELECT king, gentle, enfp, azae
    FROM 어투변환_로우데이터
    """
    
    # pandas를 사용하여 SQL 쿼리 실행 후 데이터프레임으로 저장
    ment = pd.read_sql(query, engine)

    # 열 이름 변경
    ment = ment.rename(columns={
        'king': '왕',
        'gentle': '젠틀맨',
        'enfp': 'ENFP',
        'azae': '아저씨'
    })
    
    return ment

# 함수 호출하여 데이터 로드
ment = load_and_rename_data()

# 1. 종가 불러오기 함수
def get_stock_prices(tickers):
    stock_prices = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        stock_prices[ticker] = stock.history(period="1d")['Close'].iloc[-1]  # 가장 최근 종가
    return stock_prices

# 2. GPT 불러오기
def gpt_response(prompt_type, user_input):
    if prompt_type == "포트폴리오 입력":
        prompt = f"""
        사용자가 다음과 같은 포트폴리오를 입력했습니다: '{user_input}'.
        종목명을 티커종목코드로 변환해주고, 티커종목코드와 종목수량을 반환해주세요.
        반환 예시: AAPL 10, TSLA 5, NVDA 3    
        """
    elif prompt_type == "포트폴리오 업데이트":
        prompt = f"""
        사용자가 다음과 같은 포트폴리오를 입력했습니다: '{user_input}'.
        종목명을 티커종목코드로 변환해주고, 매수/매도를 판단하여 티커종목코드와 변화량을 +, -로 반환해주세요.
        반환 예시: AAPL 10, TSLA -5, NVDA 3
        """
    else:
        prompt = user_input


    # OpenAI API 요청
    response = openai.ChatCompletion.create(
        deployment_id="gpt-3.5-turbo",  # 정확한 배포 ID를 사용하세요
        messages=[
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response['choices'][0]['message']['content'].strip()



# 3. GPT 응답 파싱 함수
def parse_portfolio_response(response):
    stock_counts = {}
    # 정규 표현식을 이용해 매수는 +, 매도는 -로 추출 (예: AAPL +10, NVDA -5)
    matches = re.findall(r'([A-Z]+)\s([+-]?[\d.]+)', response)
    for match in matches:
        ticker = match[0]
        change = float(match[1])
        stock_counts[ticker] = change
    return stock_counts

# 4. 비중 계산 함수
def calculate_portfolio(portfolio):
    stock_prices = get_stock_prices(portfolio.keys())
    portfolio_data = []
    total_value = 0

    for ticker, count in portfolio.items():
        price = stock_prices[ticker]
        value = price * count
        portfolio_data.append({'티커종목코드': ticker, '보유수량': count, '종가': price, '총가치': value})
        total_value += value

    for data in portfolio_data:
        data['비중'] = data['총가치'] / total_value

    return pd.DataFrame(portfolio_data)

# 5. CSV 입력을 처리하는 함수
def read_csv_data(csv_content):
    # CSV 내용을 pandas DataFrame으로 변환
    decoded_content = csv_content.decode('utf-8')
    csv_data = pd.read_csv(io.StringIO(decoded_content))
    return csv_data

# 6. 고객 정보 확인 함수
def check_customer_id_and_password(customer_id, password):
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT 비밀번호 FROM 고객DB WHERE 고객ID = :customer_id"),
            {"customer_id": customer_id}
        ).fetchone()
    return result and result[0] == password

# 7. 포트폴리오 저장 함수 (DB에 저장)
def save_portfolio_to_db(customer_id, portfolio_df):
    with engine.connect() as connection:
        transaction = connection.begin()  # 트랜잭션 시작
        try:
            for _, row in portfolio_df.iterrows():
                connection.execute(
                    text('''
                        INSERT INTO 포트폴리오DB (고객ID, 티커종목코드, 보유수량, 비중)
                        VALUES (:customer_id, :ticker, :count, :weight)
                    '''),
                    {
                        "customer_id": customer_id,
                        "ticker": row['티커종목코드'],
                        "count": row['보유수량'],
                        "weight": row['비중']
                    }
                )
            transaction.commit()  # 성공적으로 실행된 경우 커밋
        except Exception as e:
            transaction.rollback()  # 실패 시 롤백
            print(f"포트폴리오 저장 오류 발생: {e}")

# 8. 포트폴리오 삭제 후 새로 입력 (DB에서 기존 포트폴리오 삭제)
def delete_existing_portfolio(customer_id):
    with engine.connect() as connection:
        transaction = connection.begin()  # 트랜잭션 시작
        try:
            connection.execute(
                text("DELETE FROM 포트폴리오DB WHERE 고객ID = :customer_id"),
                {"customer_id": customer_id}
            )
            transaction.commit()  # 성공적으로 실행된 경우 커밋
        except Exception as e:
            transaction.rollback()  # 실패 시 롤백
            print(f"오류 발생: {e}")

# 9. 기존 포트폴리오에서 종목 업데이트 함수
def update_existing_portfolio(customer_id, parsed_changes):
    # 기존 포트폴리오 조회
    with engine.connect() as connection:
        existing_portfolio_query = "SELECT 티커종목코드, 보유수량 FROM 포트폴리오DB WHERE 고객ID = :customer_id"
        existing_portfolio_df = pd.read_sql(existing_portfolio_query, connection, params={"customer_id": customer_id})

        # 데이터프레임을 딕셔너리로 변환 (주식명 -> 보유수량)
        existing_portfolio = dict(zip(existing_portfolio_df['티커종목코드'], existing_portfolio_df['보유수량']))

        # 종목 추가/삭제 업데이트
        for ticker, change in parsed_changes.items():
            if ticker in existing_portfolio:
                # 기존 종목은 수량을 업데이트 (매수는 더하고, 매도는 뺌)
                existing_portfolio[ticker] += change
                # 매도로 인해 보유수량이 0 이하가 되면 포트폴리오에서 제거
                if existing_portfolio[ticker] <= 0:
                    connection.execute(
                        text("DELETE FROM 포트폴리오DB WHERE 고객ID = :customer_id AND 티커종목코드 = :ticker"),
                        {"customer_id": customer_id, "ticker": ticker}
                    )
            else:
                # 새로운 종목은 추가
                existing_portfolio[ticker] = change

    # 업데이트된 포트폴리오를 DB에 저장 (0 이하인 종목은 제거)
    for ticker, count in existing_portfolio.items():
        if count > 0:
            connection.execute(
                    text('''
                        UPDATE 포트폴리오DB
                        SET 보유수량 = :count
                        WHERE 고객ID = :customer_id AND 티커종목코드 = :ticker
                    '''),
                    {"count": count, "customer_id": customer_id, "ticker": ticker}
                )
        connection.commit()

# 10. 신규 고객 등록 함수 (고객ID 반환)
def register_new_customer(customer_id, name, age, balance, skill_level, password):
    with engine.connect() as connection:
        connection.execute(
            text('''
                INSERT INTO 고객DB (고객ID, 이름, 나이, 보유자금, 투자실력, 비밀번호)
                VALUES (:customer_id, :name, :age, :balance, :skill_level, :password)
            '''),
            {
                "customer_id": customer_id,
                "name": name,
                "age": age,
                "balance": balance,
                "skill_level": skill_level,
                "password": password
            }
        )
        connection.commit()

# 11. 고객 지표 저장 함수
def save_customer_metrics_to_db(customer_id, metrics):
    with engine.connect() as connection:
        connection.execute(
            text('''
            INSERT INTO 고객지표DB (고객ID, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수, 최대보유섹터)
            VALUES (:customer_id, :beta, :std_dev, :trend, :sentiment, :sector)
            '''),
            {
                "customer_id": customer_id,
                "beta": metrics['베타계수'],
                "std_dev": metrics['수익률표준편차'],
                "trend": metrics['트렌드지수'],
                "sentiment": metrics['투자심리지수'],
                "sector": metrics['최대보유섹터']
            }
        )
        connection.commit()

# 주식별지표 데이터 불러오기 ( 12번 함수 때문 )
def load_stock_info():
    query = "SELECT * FROM 주식별지표"
    stock_info = pd.read_sql_query(query, engine)
    return stock_info

# 12. 지표 계산 함수
def calculate_customer_metrics(customer_id, portfolio_df, stock_info):
    
    merged_df = pd.merge(stock_info, portfolio_df, on='티커종목코드')

    # 병합 결과가 비어 있는 경우 예외 처리
    if merged_df.empty:
        raise ValueError(f"Customer {customer_id}의 포트폴리오와 주식 정보가 매칭되지 않습니다.")

    # 결과를 저장할 딕셔너리 초기화
    group_result = {'고객ID': customer_id}
    
    # 해당 그룹의 계좌수 기준으로 가중치를 계산
    weights = merged_df['비중']

    # 지표에 가중치를 곱하여 가중평균 계산
    weighted_beta = (merged_df['베타계수'] * weights).sum()
    weighted_std = np.sqrt((merged_df['수익률표준편차']**2 * weights).sum())
    weighted_invest = (merged_df['투자심리지수'] * weights).sum()
    weighted_trend = (merged_df['화제성'] * weights).sum()
    weighted_sector = merged_df.groupby('섹터분류명')['비중'].sum().sort_values(ascending=False).index[0]
    
    # 가중평균 값 저장
    group_result['베타계수'] = weighted_beta
    group_result['수익률표준편차'] = weighted_std
    group_result['트렌드지수'] = weighted_trend
    group_result['투자심리지수'] = weighted_invest
    group_result['최대보유섹터'] = weighted_sector

    return group_result

# 13. 포트폴리오 저장 함수
def process_portfolio_and_metrics(customer_id, portfolio_df, stock_info):
    # 고객의 지표 계산
    customer_metrics = calculate_customer_metrics(customer_id, portfolio_df, stock_info)
    save_customer_metrics_to_db(customer_id, customer_metrics)

    return pd.DataFrame([customer_metrics])

# 14. 고객 포트폴리오 불러오기 함수
def load_portfolio_from_db(customer_id):
    query = '''
    SELECT 티커종목코드, 보유수량, 비중
    FROM 포트폴리오DB
    WHERE 고객ID = :customer_id
    '''
    # SQLAlchemy의 text() 함수를 사용하여 매개변수를 전달
    portfolio_df = pd.read_sql_query(text(query), engine, params={"customer_id": customer_id})
    
    return portfolio_df

# 15. 모든 고객의 최신 지표 불러오기 함수 (지표ID가 가장 큰, 최신 지표만 선택)
def fetch_latest_customer_metrics():
    query = '''
    SELECT 고객ID, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수
    FROM 고객지표DB
    WHERE 지표ID IN (
        SELECT MAX(지표ID) 
        FROM 고객지표DB
        GROUP BY 고객ID
    )
    '''
    customer_metrics_df = pd.read_sql_query(query, engine)
    return customer_metrics_df

# 16. 고객 ID 중복 여부 확인 함수
def is_customer_id_exist(customer_id):
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT 1 FROM 고객DB WHERE 고객ID = :customer_id"),
            {"customer_id": customer_id}
        ).fetchone()
    return result is not None

# 큐레이션지표 데이터 불러오기 ( 17번 함수때문 )
def load_curation_metrics():
    query = "SELECT * FROM 큐레이션지표"
    curation_metrics = pd.read_sql_query(query, engine)
    return curation_metrics

# 17. 고객 지표와 ETF 지표를 사용하여 코사인 유사도 계산 함수
def recommend_etfs(customer_metrics, curation_metrics, top_n=3):
    # 고객 지표와 ETF 지표 중 비교할 열들 (트렌드지수, 베타계수, 수익률표준편차, 투자심리지수)
    metrics_columns = ['트렌드지수', '베타계수', '수익률표준편차', '투자심리지수']
    
    # curation_metrics가 필요한 열을 포함하고 있는지 확인
    if not all(column in curation_metrics.columns for column in metrics_columns):
        raise ValueError("Curation metrics 데이터가 필수 열을 포함하지 않습니다.")
    
    # 데이터가 비어있을 경우 예외 처리
    if customer_metrics.empty or curation_metrics.empty:
        raise ValueError("Customer metrics 또는 curation metrics가 비어있습니다.")
    
    # 결측값 처리 (결측값이 있다면 대체)
    customer_metrics = customer_metrics.fillna(0)
    curation_metrics = curation_metrics.fillna(0)

    # 고객 지표 벡터 추출 (여기서는 고객 1명의 데이터만 있다고 가정)
    customer_vector = customer_metrics[metrics_columns].values

    # ETF 지표 벡터 추출 (모든 ETF에 대해)
    etf_vectors = curation_metrics[metrics_columns].values

    # 코사인 유사도 계산
    similarities = cosine_similarity(customer_vector, etf_vectors)
    
    # 유사도 상위 N개의 ETF 인덱스 선택
    top_n_indices = similarities[0].argsort()[::-1][:top_n]

    # 상위 N개의 ETF 추천
    recommended_etfs = curation_metrics.iloc[top_n_indices]
    
    return recommended_etfs

# 18. 문장 2개의 코사인 유사도 계산 함수
from yake import KeywordExtractor

def keyphrase_similarity(query, long_document, model=model):
    # 긴 문장에서 핵심어 추출
    kw_extractor = KeywordExtractor(lan="ko", n=1, top=5)  # 한글에 최적화되어 있으며, n=1은 단일 단어 핵심어 추출
    keywords = kw_extractor.extract_keywords(long_document)
    keyphrases = [kw[0] for kw in keywords]
    
    # 핵심어와 짧은 문장 사이의 유사도 계산
    similarities = []
    for keyphrase in keyphrases:
        query_embedding = model.encode([query])
        keyphrase_embedding = model.encode([keyphrase])
        cosine_sim = cosine_similarity(query_embedding, keyphrase_embedding)
        similarities.append(cosine_sim[0][0])
    
    return max(similarities)

# 문장 2개의 코사인 유사도 계산 함수
def calculate_cosine_similarity(sentence1, sentence2, model=model):
    # 입력 문장 임베딩
    sentence1_embedding = model.encode([sentence1])
    sentence2_embedding = model.encode([sentence2])

    # 벡터 정규화
    sentence1_embedding = sentence1_embedding / np.linalg.norm(sentence1_embedding, axis=1, keepdims=True)
    sentence2_embedding = sentence2_embedding / np.linalg.norm(sentence2_embedding, axis=1, keepdims=True)

    # 코사인 유사도 계산
    cosine_similarity = np.dot(sentence1_embedding, sentence2_embedding.T)[0][0]
    
    return cosine_similarity

# 19. 유사 문장 검색 함수
def find_similar_document(query, index, chunks, model, top_k=1):
    # 질문을 임베딩
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # 유사한 문장 검색 (top_k개의 유사 문장)
    _, indices = index.search(np.array(query_embedding), top_k)
    
    # 가장 유사한 문장 반환
    similar_documents = [chunks[idx] for idx in indices[0]]
    
    return similar_documents, keyphrase_similarity(query, similar_documents[0])

# 20. 고객 분류 및 어투 변환 함수
# 그룹 지표와 어투 매칭 딕셔너리 생성 (그룹명과 어투의 매칭)
tone_dict = {
    '부기': '왕',
    '랑이': '젠틀맨',
    '아웅이': 'ENFP',
    '숭이': '아저씨'
}

# SQL 쿼리 작성: group_metrics 테이블에서 필요한 데이터를 가져오기
query = """
    SELECT 그룹명, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수, 최대보유섹터
    FROM group_metrics
"""
group_metrics_df = pd.read_sql(query, engine)


def classify_customer_tone(customer_metrics_df, group_metrics_df, tone_dict=tone_dict):
    """
    고객의 지표를 기반으로 그룹과 유사도를 비교하여 고객을 분류하고, 그에 맞는 어투를 반환하는 함수.
    
    Args:
        customer_metrics_df (DataFrame): 고객의 지표 데이터프레임
        group_metrics_df (DataFrame): 그룹 지표 데이터프레임
        tone_dict (dict): 그룹별로 어투를 정의한 딕셔너리
    
    Returns:
        dict: 고객 ID와 해당 고객의 어투 매칭 결과
    """
    metrics_columns = ['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']

    # 데이터 확인
    if customer_metrics_df.empty:
        raise ValueError("customer_metrics_df is empty.")
    if not all(col in customer_metrics_df.columns for col in metrics_columns):
        raise ValueError(f"Missing columns in customer_metrics_df. Expected: {metrics_columns}")

    # 고객 지표 표준화
    scaler_customer = StandardScaler()
    try:
        scaled_customer_metrics = scaler_customer.fit_transform(customer_metrics_df[metrics_columns])
    except ValueError as e:
        print("Error while scaling customer metrics:", e)
        print("Check the input data format or missing values.")
        raise

    # 그룹 지표 표준화
    scaler_group = StandardScaler()
    scaled_group_metrics = scaler_group.fit_transform(group_metrics_df[metrics_columns])

    # 코사인 유사도를 계산하고 각 고객에 대해 가장 유사한 그룹 찾기
    tone_result = {}
    for idx, customer_vector in enumerate(scaled_customer_metrics):
        customer_vector = customer_vector.reshape(1, -1)  # 고객 벡터를 2차원으로 변환
        similarities = cosine_similarity(customer_vector, scaled_group_metrics)  # 유사도 계산
        most_similar_group_idx = np.argmax(similarities)  # 가장 유사한 그룹의 인덱스 찾기
        most_similar_group = group_metrics_df.iloc[most_similar_group_idx]['그룹명']
        
        # 고객ID와 매칭된 어투를 저장
        customer_id = customer_metrics_df.iloc[idx]['고객ID']
        matched_tone = tone_dict[most_similar_group]
        tone_result[customer_id] = matched_tone
    
    return tone_result, most_similar_group


# 21. 답변 양식 일치 평가 함수
metrics_list = [
    '트렌드 지수',
    '베타 계수',
    '수익률 표준 편차',
    '투자 심리 지수',
    '섹터분류명',
    '고객님의 트렌드 지수',
    '고객님의 베타 계수',
    '고객님의 수익률 표준 편차',
    '고객님의 투자 심리 지수',
    '가장 적은 차이를 보이는 지표'
]

def evaluate_clarity(response, metrics=metrics_list):
    # 설명 명확성 평가: 응답이 얼마나 많은 지표 정보를 포함하고 있는지
    metric_count = sum(1 for metric in metrics if metric in response)  # 포함된 지표 수

    return round(metric_count/len(metrics), 3) * 100


# 22. GPT 모델을 통해 ETF 추천 이유 생성
def get_etf_recommendation_with_json(etfs, user_metrics, index, documents, model):
    explanations = []
    db_scores = []
    answer_scores = []
    answer_db_scores = []
    for _, etf_info in etfs.iterrows():
        etf_name = etf_info['티커종목코드']
        trend = etf_info['트렌드지수']
        beta = etf_info['베타계수']
        volatility = etf_info['수익률표준편차']
        invest = etf_info['투자심리지수']
        sector = etf_info['섹터분류명']

        # 사용자 지표와 비교
        user_trend = user_metrics['트렌드지수'].values[0]
        user_beta = user_metrics['베타계수'].values[0]
        user_volatility = user_metrics['수익률표준편차'].values[0]
        user_invest = user_metrics['투자심리지수'].values[0]

        description, desc_dist = find_similar_document(f"{etf_name}에 대한 설명", index, documents, model, top_k=3)

        metrics = {
            '트렌드지수': abs(trend-user_trend),
            '베타계수': abs(beta-user_beta),
            '수익률표준편차': abs(volatility-user_volatility),
            '투자심리지수': abs(invest-user_invest)
        }

        min_metric = min(metrics, key=metrics.get)

        metric_description, metric_dist = find_similar_document(f"{min_metric}에 대한 설명", index, documents, model, top_k=1)

        # GPT에게 전달할 프롬프트
        prompt = f"""
        인사를 할때, 자신이 속한 그룹에 대해 함께 소개하기.
        그리고 해당 그룹에 대한 설명을 documents애서 찾아서 출력하기

        ETF {etf_name}는 다음과 같은 특징을 가지고 있습니다:
        - 트렌드 지수: {trend}
        - 베타 계수: {beta}
        - 수익률 표준 편차: {volatility}
        - 투자 심리 지수: {invest}
        - 섹터분류명: {sector}
        설명: {description}

        고객님의 포트폴리오와 비교:
        - 고객님의 트렌드 지수: {user_trend}
        - 고객님의 베타 계수: {user_beta}
        - 고객님의 수익률 표준 편차: {user_volatility}
        - 고객님의 수익률 표준 편차: {user_invest}

        가장 적은 차이를 보이는 지표:
        - 지표명: {min_metric}
        설명: {metric_description}

        주의 사항:
        - 각 지표는 표준화된 지표로 양수이면 전체 평균보다 높다는 것이고 음수이면 전체 평균보다 낮다는 것입니다.

        메인 요청 사항:
        - {etf_name}에 대한 전체적인 분석
        - 트렌드지수, 베타계수, 수익률표준편차, 투자심리지수에 대한 설명과 차이 설명
        - {min_metric}에 대해 자세한 설명

        찾아야 하는 정보를 찾지 못했을 경우, '-'로 표시해주세요.
        '설명: ' 외에 다른 지표나 지수 설명에는 문장이 아닌 단어 또는 숫자 형식으로 넣어주세요.

        """
        # 호출 후 간격 추가
        time.sleep(10)

        response = openai.ChatCompletion.create(
            deployment_id="dev1-gpt-35-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt}
            ],
            # max_tokens=1000  # 필요에 따라 조정 가능
        )
        

        explanation = response.choices[0].message.content
        explanations.append(f"{etf_name} 추천 이유:\n{explanation}\n\n")
        db_scores.append(round((desc_dist.mean()+metric_dist.mean())/2, 3)*100)
        answer_scores.append(evaluate_clarity(explanation))
        answer_db_scores.append(round(calculate_cosine_similarity(explanation, prompt), 3)*100)

    return explanations, db_scores, answer_scores, answer_db_scores

# 24. 메인 입력 루프
def portfolio_input_loop(index, documents, model, ment=ment):
    print("환영합니다.\n")
    while True:
        customer_type=input("기존 고객이신가요? 아니면 신규 고객이신가요?\n답변 (기존/신규): ").strip()
        if customer_type=="기존":
            # 고객 ID 및 비밀번호 입력 받기
            customer_id = input("고객 ID를 입력하세요: ").strip()
            password = input("비밀번호를 입력하세요: ").strip()

            # 고객 정보 확인
            if check_customer_id_and_password(customer_id, password):
                print("기존 고객님, 반갑습니다!\n")

                # 기존 포트폴리오 조회
                portfolio_query = "SELECT 티커종목코드, 보유수량, 비중 FROM 포트폴리오DB WHERE 고객ID = ?"
                portfolio_df = pd.read_sql(portfolio_query, conn, params=(customer_id,))
                
                if portfolio_df.empty:
                    print("현재 등록된 포트폴리오가 없습니다.\n")
                else:
                    print("현재 포트폴리오:\n")
                    print(portfolio_df, '\n\n')

                
                portfolio_choice=input("포트폴리오를 수정하시겠습니까?\n답변 (네/아니오): ")

                while portfolio_choice=="네":
                    print("1. 새로 입력")
                    print("2. 포트폴리오 변경")
                    print("3. GPT로 포트폴리오 변경")
                    print("0. 처음으로")

                    choice = input("선택 (1/2/3/0): ").strip()

                    if choice == "1":
                        # 기존 포트폴리오 삭제
                        delete_existing_portfolio(customer_id)
                        break

                    elif choice == "2":
                        while True:
                            print("처음으로 돌아가시려면 '종료'를 입력하세요.")

                            # 기존 포트폴리오 일부 수정
                            ticker = input("변경할 주식의 티커종목코드를 입력하세요 (예: AAPL): ").strip()
                            change_type = input("매수/매도 여부를 입력하세요 (예: 매수/매도): ").strip()
                            change_amount = float(input(f"{change_type}할 수량을 입력하세요: "))

                            # 기존 포트폴리오에서 주식 수정
                            if change_type == "매수":
                                with engine.connect() as connection:
                                    connection.execute(
                                        text('''
                                        UPDATE 포트폴리오DB
                                        SET 보유수량 = 보유수량 + :change_amount
                                        WHERE 고객ID = :customer_id AND 티커종목코드 = :ticker
                                        '''),
                                        {"change_amount": change_amount, "customer_id": customer_id, "ticker": ticker}
                                    )

                            elif change_type == "매도":
                                with engine.connect() as connection:
                                    connection.execute(
                                        text('''
                                        UPDATE 포트폴리오DB
                                        SET 보유수량 = 보유수량 - :change_amount
                                        WHERE 고객ID = :customer_id AND 티커종목코드 = :ticker
                                        '''),
                                        {"change_amount": change_amount, "customer_id": customer_id, "ticker": ticker}
                                    )

                            if ticker == "종료" or change_type == "종료" or change_amount == "종료":
                                break

                        print(f"\n{ticker} 주식의 포트폴리오가 성공적으로 업데이트되었습니다.\n")

                    elif choice == "3":
                        while True:
                            # GPT를 통해 포트폴리오 수정
                            user_input = input("챗봇에게 보낼 포트폴리오 입력 (예: 애플 10주 추가 매수, 테슬라 5주 매도, 엔비디아 3주 신규 매수했습니다.): ")
                            gpt_response_text = gpt_response("포트폴리오 업데이트", user_input)
                            print("\n챗봇 응답:\n", gpt_response_text, '\n\n')

                            parsed_portfolio = parse_portfolio_response(gpt_response_text)

                            print(parsed_portfolio)

                            portfolio_check = input("변경하시려는 수량이 맞나요?\n 답변 (네/아니오): ")

                            if portfolio_check == "네":
                                # 기존 포트폴리오에 업데이트
                                update_existing_portfolio(customer_id, parsed_portfolio)
                            elif portfolio_check == "아니오":
                                print("다시 입력해주세요.\n")
                                break

                    elif choice == "0":
                        print("프로그램을 종료합니다.\n\n")
                        break
            else:
                print("아이디와 비밀번호를 확인하세요!\n\n")

        elif (customer_type=="신규")|(choice=="3"):
            if customer_type=="신규":
                portfolio_choice=input("신규 고객이시군요. 포트폴리오를 데이터 베이스에 저장하시겠습니까?\n답변 (네/아니오): ")
                if portfolio_choice=="네":
                    # 고객 ID를 입력받고 중복 확인
                    while True:
                        customer_id = input("고객 ID를 입력하세요: ").strip()
                        
                        if is_customer_id_exist(customer_id):
                            print(f"입력하신 고객 ID '{customer_id}'는 이미 존재합니다. 다른 ID를 입력해주세요.\n")
                        else:
                            print(f"사용할 수 있는 고객 ID '{customer_id}'입니다.\n")
                            break
                    password = input("비밀번호를 입력하세요: ").strip()
                    customer_name = input("성명을 입력하세요: ").strip()
                    customer_age = int(input("나이를 입력하세요 (예: 25): ").strip())
                    customer_balance = int(input("보유자산을 입력하세요 (단위: 만원, 예: 10000): ").strip())
                    customer_skill = input("투자실력을 입력하세요 (고수, 일반 중 택 1): ").strip()
                    register_new_customer(customer_id, customer_name, customer_age, customer_balance, customer_skill, password)

            elif customer_type=="종료": 
                break

            else:
                portfolio_choice="네"

            # 포트폴리오 입력 방식 선택
            while portfolio_choice=="네":
                print("포트폴리오 입력 방식을 선택하세요:")
                print("1. 직접 입력")
                print("2. CSV 파일 업로드")
                print("3. GPT-3.5로 입력")
                print("0. 처음으로")

                choice = input("선택 (1/2/3/0): ").strip()

                save_response=""

                if choice == "1":
                    stock_names = input("보유 주식명을 쉼표로 구분하여 입력하세요 (예: AAPL,TSLA): ").split(",")
                    stock_counts = list(map(float, input("보유 주식 개수를 쉼표로 구분하여 입력하세요 (예: 10,5): ").split(",")))
                    portfolio = dict(zip(stock_names, stock_counts))
                    portfolio_df = calculate_portfolio(portfolio)
                    print("\n포트폴리오 계산 결과:\n", portfolio_df, '\n\n')
                    save_response=input("포트폴리오를 고객 데이터베이스에 저장할까요?\n답변 (네/아니오): ")

                elif choice == "2":
                    file_path = input("CSV 파일 경로를 입력하세요(파일형태: utf-8, 컬럼명: 티커종목코드, 보유수량): ").strip()
                    try:
                        with open(file_path, mode='r') as file:
                            csv_content = file.read().encode('utf-8')
                            portfolio_df = read_csv_data(csv_content)
                            portfolio_dict = dict(zip(portfolio_df['티커종목코드'], portfolio_df['보유수량']))
                            portfolio_df = calculate_portfolio(portfolio_dict)
                            print("\n포트폴리오 계산 결과:\n", portfolio_df, '\n\n')
                            save_response=input("포트폴리오를 고객 데이터베이스에 저장할까요?\n답변 (네/아니오): ")
                    except FileNotFoundError:
                        print("CSV 파일을 찾을 수 없습니다. 다시 시도하세요.\n\n")

                elif choice == "3":
                    user_input = input("챗봇에게 보낼 포트폴리오 입력 (예: 애플 10주, 테슬라 5주 보유 중입니다): ")
                    gpt_response = gpt_response("포트폴리오 입력", user_input)
                    print("\nGPT 응답:\n", gpt_response)
                    parsed_portfolio = parse_portfolio_response(gpt_response)
                    portfolio_df = calculate_portfolio(parsed_portfolio)
                    print("\n포트폴리오 계산 결과:\n", portfolio_df)
                    save_response=input("포트폴리오를 고객 데이터베이스에 저장할까요?\n답변 (네/아니오): ")

                elif choice == "0":
                    print("처음으로 돌아갑니다.\n\n")
                    break

                else:
                    print("잘못된 입력입니다. 다시 시도하세요.\n\n")
                
            if save_response=="네":
                save_portfolio_to_db(customer_id, portfolio_df)
            elif save_response=="아니오":
                print("다시 선택해주세요.\n\n")
        else:
            print("다시 입력해주세요.\n\n")

        chat=input("챗봇을 사용하시겠습니까? 답변 (네/아니오): ")
        while chat=="네":
            data_base=input("ETF추천을 위한 포트폴리오를 데이터베이스에서 불러오시겠습니까? 아니면 입력하여 진행하시겠습니까?\n답변 (불러오기/입력): ")
            if data_base =="불러오기":
                try:
                    portfolio_df=load_portfolio_from_db(customer_id)
                    print("고객님의 포트폴리오:")
                    print(portfolio_df, '\n\n')
                except:
                    print("고객님의 포트폴리오가 존재하지 않습니다.\n")

            elif data_base=="입력":
                user_input = input("챗봇에게 보낼 포트폴리오 입력 (예: 애플 10주, 테슬라 5주 보유 중입니다): ")
                gpt_response = gpt_response("포트폴리오 입력", user_input)
                parsed_portfolio = parse_portfolio_response(gpt_response)
                portfolio_df = calculate_portfolio(parsed_portfolio)
                print("고객님의 포트폴리오:")
                print(portfolio_df, '\n\n')
            else:
                break

            customer_metrics = process_portfolio_and_metrics(customer_id, portfolio_df, stock_info)

            # 고객 지표와 ETF 지표에 대한 표준화 작업
            scaler = StandardScaler()

            # 고객 지표 불러오기 (최신 지표만)
            customer_metrics_df = fetch_latest_customer_metrics()

            # 고객 지표 표준화
            scaler.fit_transform(customer_metrics_df[['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']])

            # 고객 지표 표준화
            scaled_customer_metrics = pd.DataFrame(
                scaler.transform(customer_metrics[['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']]),
                columns=['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']
            )

            # ETF 추천
            recommended_etfs = recommend_etfs(scaled_customer_metrics)
            print(recommended_etfs, '\n\n')

            # 고객을 그룹과 매칭하여 어투 변환 진행
            tone_result, group_result = classify_customer_tone(customer_metrics_df)

            # 고객 어투 변환 결과 안내
            print(f" {customer_id}님은 '{group_result}' 투자자 입니다.\n")

            # 유사한 문장 검색
            similar_documents, similar_score = find_similar_document(f"{group_result} 그룹에 대한 설명", index, documents, model, top_k=3)

            prompt = f'''
                **말투 데이터의 내용은 절대 사용하지 말고, 말투만 참고하세요.**
                **출력할 내용은 아래의 그룹 설명에서 가져와야 하며, 절대 변경하지 마세요.**

                ### 1. 말투 데이터 (말투 학습용 데이터로, 내용은 무시하세요):
                - 사용할 말투: {ment[tone_result[customer_id]].dropna()[:5]}
                (이 말투는 학습용이며, 내용은 무시하세요.)

                ### 2. 출력할 실제 내용 (이 내용만 사용하세요):
                - 고객이 속한 그룹: {group_result}
                - 그룹 설명: {similar_documents[0]}

                **중요 지시사항**:
                1. "말투 데이터"는 **문체만 학습**하고, **내용은 절대 사용하지 마세요**.
                2. "출력할 내용"은 "그룹 설명"에서만 가져오세요.
                3. "그룹 설명"의 **내용은 변경하지 말고**, 오직 말투만 변형하세요.
                4. 고객은 위 그룹에 속한 고객입니다. 고객이 충분히 이해하도록 "고객이 {group_result} 그룹에 속했다는 것"을 언급하고 "위에서 학습한 말투"로 인사를 하고 "학습한 말투"로 "고객의 그룹"을 설명하세요.

                답변 구성: 학습한 말투로 인사, 학습한 말투로 고객이 어떤 그룹에 속해있는지, 학습한 말투로 그룹 설명
            '''

            gpt_reply = gpt_response(prompt)
            print(gpt_reply, '\n\n', '그룹 분류 신뢰도: ',round(similar_score, 3)*100, '%')

            # RAG 기반으로 ETF 추천 이유 생성 및 지표 설명 추가
            etf_explanations, db_scores, answer_scores, answer_db_scores = get_etf_recommendation_with_json(recommended_etfs, scaled_customer_metrics, index, documents, model)
            
            i=0
            # 결과 출력
            for explanation, db_score, answer_score, answer_db_score in zip(etf_explanations, db_scores, answer_scores, answer_db_scores):
                i+=1
                print(f"{i}번째 ETF\n")
                if (db_score+answer_score+answer_db_score)/3>=50:
                    print(explanation)
                    print('\n')
                else:
                    print('생성된 문장이 완벽하지 않아서 출력하지 않습니다. 정보를 알고 싶으시면 ETF 이름을 물어봐주세요.\n')
                print(f"\n데이터베이스에서 추출된 자료의 신뢰도: {db_score}%")
                print(f"\n문장 완성도: {answer_score}%")
                print(f"\n문장과 데이터베이스의 유사도: {answer_db_score}%\n\n")

            while True:
                print("처음으로 돌아가시려면 '종료'를 입력하세요.\n")
                customer_response = input("자유롭게 질문하세요.")
                
                if customer_response == '종료':
                    break
                
                # 유사한 문장 검색
                similar_documents, similar_response = find_similar_document(customer_response, index, documents, model, top_k=3)
                
                if keyphrase_similarity(customer_response, similar_documents)>0.5:
                    print('이 문장은 데이터 베이스에서 유사한 정보를 가져온 후 작성된 문장입니다.\n')
                else:
                    print('이 문장은 데이터 베이스에서 유사한 정보를 찾지 못하고 작성되 문장입니다.\n')
                # GPT에 유사한 문장을 프롬프트로 전달
                gpt_prompt = f"사용자의 질문: {customer_response}\n\n유사한 문장: {similar_documents[0]}\n\nGPT의 답변:"
                gpt_reply = gpt_response(gpt_prompt)
                
                # GPT 답변 출력
                print(f"사용자: {customer_response}\n")
                print("챗봇 답변: ", gpt_reply, '\n\n')


# 고객 질의 예시
# 기존 pyodbc 연결 문자열을 사용하여 SQLAlchemy 엔진 생성
conn_str = (
    'mssql+pyodbc://jyun22:1emdrkwk!@jy-nh-server01.database.windows.net:1433/NH-SQLDB-001?'
    'driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no'
)
engine = create_engine(conn_str)

# 고객 포트폴리오 및 지표 처리 함수
def AI_test(customer_id, stock_info, curation_metrics, index=index, documents=documents, model=model ):
    portfolio_df=load_portfolio_from_db(customer_id)
    print("고객님의 포트폴리오:")
    print(portfolio_df, '\n\n')

    customer_metrics = process_portfolio_and_metrics(customer_id, portfolio_df,stock_info)

    # 고객 지표와 ETF 지표에 대한 표준화 작업
    scaler = StandardScaler()

    # 고객 지표 불러오기 (최신 지표만)
    customer_metrics_df = fetch_latest_customer_metrics()

    # 고객 지표 표준화
    scaler.fit_transform(customer_metrics_df[['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']])

    # 고객 지표 표준화
    scaled_customer_metrics = pd.DataFrame(
        scaler.transform(customer_metrics[['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']]),
        columns=['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']
    )

    etfs = []

    # ETF 추천
    recommended_etfs = recommend_etfs(scaled_customer_metrics, curation_metrics)
    print('추천 ETF: ', end='')
    for etf in recommended_etfs['티커종목코드']:
        print(etf, end=' ')
        etfs.append(etf)

    print('\n\n')

    # 고객을 그룹과 매칭하여 어투 변환 진행
    tone_result, group_result = classify_customer_tone(customer_metrics_df,group_metrics_df)

    # 고객 어투 변환 결과 안내
    print(f" {customer_id}님은 '{group_result}' 투자자 입니다.\n")

    # 검색할 기준 문장 생성 (예: 그룹 이름을 기준으로 설명 검색)
    query = group_result + "에 대한 설명"  # 그룹 이름 사용

    # 유사한 문장 검색
    similar_documents, _ = find_similar_document(query, index, documents, model, top_k=1)


    # RAG 기반으로 ETF 추천 이유 생성 및 지표 설명 추가
    etf_explanations, db_scores, answer_scores, answer_db_scores = get_etf_recommendation_with_json(recommended_etfs, scaled_customer_metrics, index, documents, model)
    
    i=0
    # 결과 출력
    for explanation, db_score, answer_score, answer_db_score in zip(etf_explanations, db_scores, answer_scores, answer_db_scores):
        i+=1
        print(f"{i}번째 ETF\n")
        if (db_score+answer_score+answer_db_score)/3>=50:
            print(explanation)
            print('\n')
        else:
            print('생성된 문장이 완벽하지 않아서 출력하지 않습니다. 정보를 알고 싶으시면 ETF 이름을 물어봐주세요.\n')
        print(f"\n데이터베이스에서 추출된 자료의 유사도: {db_score}%")
        print(f"\n문장 완성도: {answer_score}%")
        print(f"\n문장과 데이터베이스의 유사도: {answer_db_score}%\n\n")
    
    return etfs

# stock_info를 로드하여 각 고객에 대해 AI_test 함수 호출
stock_info = load_stock_info()
curation_metrics = load_curation_metrics() # 큐레이션 지표 불러오기
etfs = []


for cus in ['0번고객', '1번고객', '2번고객', '3번고객']:
    etfs.append(AI_test(cus, stock_info=stock_info, curation_metrics=curation_metrics, model=model, index=index, documents=documents))



# 함수 작동 
# 실제 서비스 작동 코드입니다. 
# 실행하면, 현재 DB에 저장되어있는 ID와 비밀번호를 입력해서 서비스가 진행되기에 주석처리하였습니다.
# portfolio_input_loop(index, documents, model)
# conn.close()