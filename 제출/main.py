'''
각 파트별 요약

### 1. 고객 분류
투자자들의 투자 성향을 파악하기 위해 다섯 개 그룹으로 고객을 세분화했습니다.  
여러 테이블에 나뉘어진 주식 정보를 분석하여, 주식별로 ‘변동성’, ‘시장 민감도’, ‘화제성’, 그리고 ‘투자 심리’를 반영하는 파생 변수를 생성했습니다.   
이를 바탕으로 고객의 보유 주식 비중에 따라 고객 분류를 진행하고, 이를 ETF 큐레이션과 연결하여 각 투자 스타일에 맞는 맞춤형 ETF 큐레이션을 시도했습니다.

### 2. 큐레이션
고객 군집별 특성에 맞춘 ETF 큐레이션 서비스를 제공하기 위해 다음의 4가지 지표를 생성했습니다:

- 트렌드 지수: '화제성' + '종목 조회 건수 & 종가 변화율'  
- 수익률 표준편차  
- 베타 계수  
- 투자 심리지수  

초기에는 적은 양의 ETF를 기반으로 서비스를 제공했으나, 더 폭넓은 선택지를 제공하고 싶다는 아쉬움이 남았습니다.    
따라서 해외증시뉴스 데이터를 통해 트렌드 지수를, ETF 점수정보 테이블을 활용하여 나머지 지표를 검증했습니다.  
결과적으로 총 441개의 ETF에 대한 큐레이션을 완성했으며 이를 통해 고객에게 폭넓은 선택지를 제공하는 맞춤형 추천 서비스를 도입하였습니다.


### 3. 생성형 AI
위에서 진행한 각 종목에 대한 분석을 바탕으로 코사인 유사도를 이용하고 생성형 AI를 활용하여 고객 맞춤형 ETF 추천을 진행했습니다.  
각 고객 군집의 성향에 맞춘 4가지 어투를 학습시켜 투자자에게 제공했으며, 이를 통해 투자자는 본인의 투자 성향을 파악하고 더욱 정교한 개인 맞춤형 큐레이션을 진행할 수 있었습니다.

또한, 생성형 AI의 주요 문제점인 할루시네이션을 방지하기 위해 RAG(랜덤 접근 그래프)와 평가지표를 이용했습니다. 고객 데이터를 담고 있는 데이터베이스를 생성하여, 신규 고객과 기존 고객의 포트폴리오를 구분하고 보다 심층적인 추천을 진행했습니다. 또한 사용자 정보를 저장할 수 있어 실제 배포 시 편리한 이용이 가능하도록 설계하였습니다.

마지막으로 고객의 입력과 데이터베이스 정보를 바탕으로 생성형 AI 출력을 평가하는 3단계 정확도 평가 프로세스를 통해 할루시네이션을 방지했습니다. 
이러한 체계적인 평가 과정을 통해 생성형 AI는 신뢰할 수 있는 결과를 제공할 수 있습니다.
'''

import koreanize_matplotlib
import matplotlib.pyplot as plt
import warnings

# 그림 선명도 향상을 위해 DPI 설정
plt.rcParams['figure.dpi'] = 150  # Retina 디스플레이와 유사한 선명도

# 모든 경고 메시지를 무시
warnings.filterwarnings('ignore')

# 라이브러리 로드
import pandas as pd
import numpy as np
import matplotlib
import selenium
import statsmodels
import yfinance as yf
import sklearn
import seaborn as sns
import sqlite3
import faiss
import openai
import yake
import koreanize_matplotlib
from sentence_transformers import SentenceTransformer
import sys  # 파이썬 버전 확인을 위해 추가
import openpyxl

# 패키지 버전 확인
def get_package_version(package_name):
    try:
        import pkg_resources
        return pkg_resources.get_distribution(package_name).version
    except:
        return "Version not found"

# 라이브러리 버전 출력
print(f"Python version: {sys.version}")  # 파이썬 버전 출력
print(f"pandas version: {pd.__version__}")
print(f"openpyxl version: {openpyxl.__version__}")
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")
print(f"selenium version: {selenium.__version__}")
print(f"statsmodels version: {statsmodels.__version__}")
print(f"yfinance version: {yf.__version__}")
print(f"sklearn version: {sklearn.__version__}")
print(f"seaborn version: {sns.__version__}")
print(f"sqlite3 version: {sqlite3.sqlite_version}")
print(f"faiss version: {faiss.__version__}")
print(f"sentence-transformers version: {get_package_version('sentence-transformers')}")
print(f"openai version: {openai.__version__}")
print(f"yake version: {yake.__version__}")
print(f"koreanize-matplotlib version: {get_package_version('koreanize_matplotlib')}")

# 외의 라이브러리 로드
import os
from sklearn.preprocessing import StandardScaler
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import statsmodels.api as sm
import math
from functools import reduce
import itertools
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import io
import re
from yake import KeywordExtractor
from sklearn.ensemble import RandomForestRegressor
import json

# 아래의 주석은 openai 1.51.1에서만 실행이 가능합니다. azure가 호환되는 0.28 에서는 불가능합니다. 1.51.1로 아래의 주석을 풀고 실행하시면 어투변환 결과를 확인하실 수 있습니다.
# from openai import OpenAI


# 출력설정
pd.options.display.float_format = '{:,.6f}'.format

# 분석 및 EDA

# 파일 한글명으로 불러오기
def load_and_map_data(schema_file_path, folder_path):
    """
    스키마 파일을 기반으로 테이블명과 컬럼명을 매핑하여 CSV 파일을 불러오고,
    한글 테이블명에 맞춰 데이터프레임을 객체로 저장하는 함수.

    Parameters:
    - schema_file_path (str): 스키마 파일의 경로 (엑셀 파일)
    - folder_path (str): CSV 파일들이 저장된 폴더 경로
    """

    # 스키마 파일 읽어오기
    schema_df = pd.read_excel(schema_file_path)

    # 스키마에 따라 테이블명과 컬럼명을 매핑하기 위한 딕셔너리 생성
    table_column_mapping = {}
    table_name_mapping = {}

    # 스키마를 순회하여 테이블별 컬럼명 매핑을 생성
    for index, row in schema_df.iterrows():
        # 영문 테이블명과 한글 테이블명 매핑
        table_english_name = 'NH' + row['테이블영문명'][3:]
        table_korean_name = row['테이블한글명'].split('_')[-1]  # 마지막 _ 뒤에 있는 단어를 사용

        # 영문 테이블명을 기준으로 한글 테이블명 매핑 딕셔너리 생성
        table_name_mapping[table_english_name] = table_korean_name

        # 영어 컬럼명과 한글 컬럼명을 매핑
        column_name_eng = row['컬럼'].lower().strip()
        column_name_kor = row['컬럼명'].split('(')[0].strip()

        # 영문 테이블명에 해당하는 컬럼 매핑 딕셔너리 생성
        if table_english_name not in table_column_mapping:
            table_column_mapping[table_english_name] = {}

        # 컬럼명 매핑 추가
        table_column_mapping[table_english_name][column_name_eng] = column_name_kor

    # data 폴더 안에 있는 CSV 파일을 불러오고 컬럼명을 한글로 변경하여 객체로 저장
    for csv_file in os.listdir(folder_path):
        if csv_file.endswith('.csv'):
            # 파일명에서 테이블명을 추출
            table_english_name = csv_file.split('.')[0]  # 확장자를 제거한 파일명

            # 영문 테이블명에 대응하는 한글 테이블명 확인
            if table_english_name in table_name_mapping:
                table_korean_name = table_name_mapping[table_english_name]
                file_path = os.path.join(folder_path, csv_file)

                # CSV 파일을 읽어오기
                df = pd.read_csv(file_path, encoding='cp949')

                # 각 컬럼의 값에서 앞뒤 공백을 제거하고, 연속된 공백을 단일 공백으로 변경
                df = df.map(lambda x: x.strip().replace('  ', ' ') if isinstance(x, str) else x)

                # 해당 테이블에 대한 컬럼 매핑을 가져옴
                column_mapping = table_column_mapping.get(table_english_name, {})

                # 컬럼명을 한글로 변경
                df.rename(columns=column_mapping, inplace=True)

                # 테이블명을 한글 테이블명에서 마지막 "_" 뒤에 나온 부분으로 설정하고 데이터프레임을 저장
                globals()[table_korean_name] = df
                print(f"'{table_korean_name}' 객체에 데이터프레임이 저장되었습니다.")
            
    return list(table_name_mapping.values())

# 함수 사용 예시
schema_file_path = './data/2024NH투자증권빅데이터경진대회_데이터_스키마.xlsx'
folder_path = './data'
table_names = load_and_map_data(schema_file_path, folder_path)

# 겹치는 종목코드 확인하는 함수
def compare_stock_codes(df1, df2, code_col1, code_col2, df1_name, df2_name):
    """
    두 데이터프레임에서 종목코드가 겹치는 개수를 확인하는 함수
    
    Parameters:
    df1 (DataFrame): 첫 번째 데이터프레임 (예: 주식일별정보)
    df2 (DataFrame): 두 번째 데이터프레임 (예: 고객보유정보)
    code_col1 (str): 첫 번째 데이터프레임의 종목코드 컬럼명 (예: '티커종목코드')
    code_col2 (str): 두 번째 데이터프레임의 종목코드 컬럼명 (예: '티커종목코드')
    df1_name (str): 첫 번째 데이터프레임의 이름
    df2_name (str): 두 번째 데이터프레임의 이름
    
    Returns:
    None: 출력 결과는 겹치는 종목 코드 개수와 각 데이터프레임에만 있는 종목 코드 개수
    """
    # 첫 번째 데이터프레임의 종목코드 추출 (고유 값)
    codes1 = df1[code_col1].unique()
    
    # 두 번째 데이터프레임의 종목코드 추출 (고유 값)
    codes2 = df2[code_col2].unique()
    
    # 1. 겹치는 종목코드 (교집합)
    common_codes = set(codes1).intersection(set(codes2))
    print(f"겹치는 종목코드 개수: {len(common_codes)}")
    
    # 2. 첫 번째 데이터프레임에만 있는 종목코드 (차집합)
    only_in_df1 = set(codes1) - set(codes2)
    print(f"{df1_name}에만 있는 종목코드 개수: {len(only_in_df1)}")
    
    # 3. 두 번째 데이터프레임에만 있는 종목코드 (차집합)
    only_in_df2 = set(codes2) - set(codes1)
    print(f"{df2_name}에만 있는 종목코드 개수: {len(only_in_df2)}")

    return only_in_df1, common_codes

# 함수 호출 시 데이터프레임과 함께 이름을 전달
test_list1, test_list2 = compare_stock_codes(종목일자별시세, 주식일별정보, '티커종목코드', '티커종목코드', '종목일자별시세', '주식일별정보')

# 액면분할 등 큰 이벤트가 있었던 주식 찾기
def detect_and_adjust_based_on_price(df, stock_col, date_col, quantity_col, price_col):
    """
    기준종가의 급격한 변화를 기반으로 액면분할을 감지하고 자동으로 데이터를 조정하는 함수.
    가격 변동이 음수일 때는 감소(액면분할), 양수일 때는 증가(주식 병합)를 처리합니다.

    Parameters:
    df (pd.DataFrame): 데이터프레임 (주식 정보 포함)
    stock_col (str): 종목 코드가 포함된 열 이름
    date_col (str): 날짜가 포함된 열 이름
    quantity_col (str): 총 보유 수량이 포함된 열 이름
    price_col (str): 기준 가격이 포함된 열 이름
    threshold (float): 기준 가격의 변화율 임계값 (예: 0.4는 40% 이상의 변화를 감지)

    Returns:
    pd.DataFrame: 변화가 감지된 데이터를 수정한 새로운 데이터프레임
    """
    # 결과를 저장할 빈 DataFrame 생성
    adjusted_df = pd.DataFrame()

    # 각 종목별로 데이터를 처리
    for stock in df[stock_col].unique():
        stock_df = df[df[stock_col] == stock].copy()
        stock_df = stock_df.sort_values(by=date_col)

        # 기준종가의 전날 대비 변화율 계산
        stock_df['가격변화율'] = stock_df[price_col].pct_change()

        # 기준종가 변화율이 임계값을 초과하는 경우 액면분할 또는 병합이 발생한 것으로 추정
        stock_df['액면분할감지'] = (stock_df['가격변화율'] > 0.9)|(stock_df['가격변화율']<-0.45)

        # 급격한 변화가 발생한 첫 시점 탐색
        change_points = stock_df[stock_df['액면분할감지']][date_col].values
        if len(change_points) > 0:
            split_date = change_points[0]  # 첫 번째 변화 지점
            price_change = stock_df.loc[stock_df[date_col] == split_date, '가격변화율'].values[0]

            # 가격 변화율이 음수일 때 (액면분할), 양수일 때 (주식 병합)
            if price_change < 0:
                split_ratio = 1 / (1 + price_change)
                split_type = '액면분할'
            else:
                split_ratio = 1 + price_change
                split_type = '주식병합'

            split_ratio = round(split_ratio)  # 항상 정수로 처리
            print(f"{stock}의 감지 시점: {split_date}, 추정 분할 비율: {split_ratio}")

            # 분할 시점을 기준으로 중간 이후(또는 이전) 처리
            middle_date = stock_df[date_col].median()
            if pd.to_datetime(split_date) > pd.to_datetime(middle_date):
                # 중간 이후면 이후 데이터를 조정
                after_split = stock_df[stock_df[date_col] >= split_date]
                if split_type == '액면분할':
                    stock_df.loc[after_split.index, price_col] *= split_ratio
                    stock_df.loc[after_split.index, quantity_col] = (stock_df.loc[after_split.index, quantity_col] / split_ratio).round().astype(int)
                else:  # 주식병합일 경우
                    stock_df.loc[after_split.index, price_col] /= split_ratio
                    stock_df.loc[after_split.index, quantity_col] = (stock_df.loc[after_split.index, quantity_col] * split_ratio).round().astype(int)
            else:
                # 중간 이하면 이전 데이터를 조정
                before_split = stock_df[stock_df[date_col] < split_date]
                if split_type == '액면분할':
                    stock_df.loc[before_split.index, price_col] /= split_ratio
                    stock_df.loc[before_split.index, quantity_col] = (stock_df.loc[before_split.index, quantity_col] * split_ratio).round().astype(int)
                else:  # 주식병합일 경우
                    stock_df.loc[before_split.index, price_col] *= split_ratio
                    stock_df.loc[before_split.index, quantity_col] = (stock_df.loc[before_split.index, quantity_col] / split_ratio).round().astype(int)

        # 처리된 종목의 데이터프레임을 결과에 추가
        adjusted_df = pd.concat([adjusted_df, stock_df])

    # 불필요한 컬럼 제거
    adjusted_df = adjusted_df.drop(columns=['가격변화율', '액면분할감지'])

    return adjusted_df

# 함수 호출 예시
주식일별정보_보완 = detect_and_adjust_based_on_price(주식일별정보, '티커종목코드', 'BSE_DT', '총보유수량', '기준종가')

# 결과 확인
print(주식일별정보_보완)

# 예시 (NVDA - 6.10)
print(주식일별정보[주식일별정보['티커종목코드']=='NVDA'].sort_values('BSE_DT').head(20))

# 각 주식 당 고객 군집의 보유 계좌, 금액 구하는 함수
def process_customer_stock_data(df1, df2):
    merged_df = pd.merge(df1, df2[['BSE_DT', '티커종목코드', '총보유계좌수', '총보유수량', '기준종가', '종목조회건수', '신규매수계좌수', '전량매도계좌수']].rename(columns={'BSE_DT': '기준일자'}), how='inner', on=['기준일자', '티커종목코드'])
    merged_df['고객구성계좌수']=round(merged_df['고객구성계좌수비율']*merged_df['총보유계좌수']/100)
    merged_df['총보유금액']=merged_df['총보유수량']*merged_df['기준종가']/1000
    merged_df['고객구성금액']=merged_df['고객구성투자비율']*merged_df['총보유금액']/100
    merged_df.drop(['고객구성계좌수비율', '고객구성투자비율', '총보유수량'], axis=1, inplace=True)

    # 고객구성중분류코드에 따른 열 이름과 설명 매핑
    code_mapping = {
        11: '고수',
        12: '일반',
        21: '20대이하',
        22: '30대',
        23: '40대',
        24: '50대',
        25: '60대이상',
        31: '3000만미만',
        32: '3000만-1억',
        33: '1억-10억',
        34: '10억이상'
    }

    # 고객구성중분류코드에 따른 계좌수 및 투자수량 열 설명 추가
    merged_df['고객구성중분류설명'] = merged_df['고객구성중분류코드'].map(code_mapping)

    # 피벗을 통해 데이터를 행에서 열로 변환 (열 이름을 계좌수와 투자수량을 각각 구분)
    df_pivoted = merged_df.pivot_table(
        index=['기준일자', '티커종목코드', '총보유계좌수', '총보유금액', '종목조회건수', '신규매수계좌수', '전량매도계좌수'],
        columns='고객구성중분류설명',
        values=['고객구성계좌수', '고객구성금액'],
        aggfunc='sum'
    ).reset_index()

    # 멀티 인덱스를 평평하게 만들어 열 이름을 보기 쉽게 변환
    df_pivoted.columns = [
        f"{name} {col[4:]}" if name else col
        for col, name in df_pivoted.columns
    ]

    # NaN 값을 0으로 대체한 후 정수형으로 변환
    df_pivoted.fillna(0, inplace=True)

    # 열을 code_mapping 순서대로 정렬
    sorted_columns = ['기준일자', '티커종목코드', '종목조회건수', '총보유계좌수', '총보유금액', '신규매수계좌수', '전량매도계좌수']

    # code_mapping에 따라 계좌수와 투자수량 열을 순서대로 정렬
    for key in code_mapping.values():
        sorted_columns += [f"{key} 계좌수", f"{key} 금액"]

    # 정렬된 열 순서로 데이터프레임 재구성
    df_pivoted = df_pivoted[sorted_columns]

    for col in df_pivoted.columns[2:-1:2]:
        df_pivoted[col]=df_pivoted[col].astype(int)

    return pd.merge(df_pivoted, 해외종목정보[['티커종목코드', '주식/ETF구분코드']], how='left', on='티커종목코드')

주식별고객정보 = process_customer_stock_data(고객보유정보, 주식일별정보_보완)
주식별고객정보 = 주식별고객정보[주식별고객정보['티커종목코드'].isin(종목일자별시세['티커종목코드'].unique())]
주식별고객정보.to_csv('주식별고객정보.csv', encoding='cp949', index=False)
print(주식별고객정보)

# 실력대/나이대/자산군 별 주식 보유 정보 시각화
# 숫자형 컬럼만 선택
numeric_cols = ['총보유계좌수', '총보유금액', '신규매수계좌수', '전량매도계좌수', '고수 계좌수', '고수 금액', '일반 계좌수', '일반 금액',
                '20대이하 계좌수', '30대 계좌수', '40대 계좌수', '50대 계좌수', '60대이상 계좌수', '3000만미만 계좌수', '3000만미만 금액',
                '3000만-1억 계좌수', '3000만-1억 금액', '1억-10억 계좌수', '1억-10억 금액', '10억이상 계좌수', '10억이상 금액']

# 필요한 컬럼만 필터링하여 groupby 수행 (여기서는 티커 종목코드별로 그룹화)
df_numeric = 주식별고객정보[주식별고객정보['주식/ETF구분코드']=='주식'][['티커종목코드'] + numeric_cols].groupby('티커종목코드').sum().reset_index()

# 상위 30개 종목코드를 '총보유계좌수' 기준으로 정렬
df_sorted = df_numeric.sort_values(by='총보유계좌수', ascending=False).head(30)

# 세트별 카테고리 리스트
categories = [
    (['고수 계좌수', '일반 계좌수'], '고수/일반 계좌수', '티커 종목코드별 고수/일반 계좌수'),
    (['20대이하 계좌수', '30대 계좌수', '40대 계좌수', '50대 계좌수', '60대이상 계좌수'], '나이대별 계좌수', '티커 종목코드별 나이대별 계좌수'),
    (['3000만미만 계좌수', '3000만-1억 계좌수', '1억-10억 계좌수', '10억이상 계좌수'], '금액대별 계좌수', '티커 종목코드별 금액대별 계좌수')
]

# 반복문을 통해 각 카테고리별 스택형 막대차트 그리기
for group, ylabel, title in categories:
    plt.figure(figsize=(10, 6))

    # 첫 번째 항목은 따로 처리
    bottom = df_sorted[group[0]]
    plt.bar(df_sorted['티커종목코드'], bottom, label=group[0], color='skyblue')

    # 나머지 항목들을 쌓아가는 방식으로 그리기
    for col in group[1:]:
        plt.bar(df_sorted['티커종목코드'], df_sorted[col], bottom=bottom, label=col)
        bottom += df_sorted[col]  # 이전 항목을 bottom에 더해 쌓음

    # 제목, 축 라벨, 범례 설정
    plt.title(title, fontsize=14)
    plt.xlabel('티커 종목코드', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 실력대/나이대/자산군 별 ETF 보유 정보 시각화
import matplotlib.pyplot as plt

# 숫자형 컬럼만 선택
numeric_cols = ['총보유계좌수', '총보유금액', '신규매수계좌수', '전량매도계좌수', '고수 계좌수', '고수 금액', '일반 계좌수', '일반 금액',
                '20대이하 계좌수', '30대 계좌수', '40대 계좌수', '50대 계좌수', '60대이상 계좌수', '3000만미만 계좌수', '3000만미만 금액',
                '3000만-1억 계좌수', '3000만-1억 금액', '1억-10억 계좌수', '1억-10억 금액', '10억이상 계좌수', '10억이상 금액']

# 필요한 컬럼만 필터링하여 groupby 수행 (여기서는 티커 종목코드별로 그룹화)
df_numeric = 주식별고객정보[주식별고객정보['주식/ETF구분코드']=='ETF'][['티커종목코드'] + numeric_cols].groupby('티커종목코드').sum().reset_index()

# 상위 30개 종목코드를 '총보유계좌수' 기준으로 정렬
df_sorted = df_numeric.sort_values(by='총보유계좌수', ascending=False).head(30)

# 세트별 카테고리 리스트
categories = [
    (['고수 계좌수', '일반 계좌수'], '고수/일반 계좌수', '티커 종목코드별 고수/일반 계좌수'),
    (['20대이하 계좌수', '30대 계좌수', '40대 계좌수', '50대 계좌수', '60대이상 계좌수'], '나이대별 계좌수', '티커 종목코드별 나이대별 계좌수'),
    (['3000만미만 계좌수', '3000만-1억 계좌수', '1억-10억 계좌수', '10억이상 계좌수'], '금액대별 계좌수', '티커 종목코드별 금액대별 계좌수')
]

# 반복문을 통해 각 카테고리별 스택형 막대차트 그리기
for group, ylabel, title in categories:
    plt.figure(figsize=(10, 6))

    # 첫 번째 항목은 따로 처리
    bottom = df_sorted[group[0]]
    plt.bar(df_sorted['티커종목코드'], bottom, label=group[0], color='skyblue')

    # 나머지 항목들을 쌓아가는 방식으로 그리기
    for col in group[1:]:
        plt.bar(df_sorted['티커종목코드'], df_sorted[col], bottom=bottom, label=col)
        bottom += df_sorted[col]  # 이전 항목을 bottom에 더해 쌓음

    # 제목, 축 라벨, 범례 설정
    plt.title(title, fontsize=14)
    plt.xlabel('티커 종목코드', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 지표생성

# 화제성

# 주식
df = 종목일자별시세.copy()

# 날짜 형식 변환
df['거래일자'] = pd.to_datetime(df['거래일자'], format='%Y%m%d')

# 월 단위 그룹화 위해 '월' 컬럼 생성
df['월'] = df['거래일자'].dt.to_period('M')

# 매도+매수 합계와 누적거래수량 비교
df['일치여부'] = (df['매도체결합계수량'] + df['매수체결합계수량']) == df['누적거래수량']

# 월별로 일치율 계산
monthly_match = df.groupby('월').agg(
    총행수=('일치여부', 'count'),
    일치행수=('일치여부', 'sum')  # True 값 개수를 합산
).reset_index()

# 일치율(%) 계산
monthly_match['일치율(%)'] = (monthly_match['일치행수'] / monthly_match['총행수']) * 100

# 결과 출력
print(monthly_match)

종목일자별시세_보완 = 종목일자별시세.copy()

종목일자별시세_보완.loc[(종목일자별시세_보완['매수체결합계수량']==0)&(종목일자별시세_보완['매도체결합계수량']==0) , ['매수체결합계수량', '매도체결합계수량']] = np.nan

print(종목일자별시세_보완['매수체결합계수량'].isna().sum()/len(종목일자별시세_보완))

# 주식 화제성

# 변화량을 계산하는 함수
def calculate_change_amount(current, previous):
    return current - previous  # 그날과 전날의 차이

# 모든 날의 변화량 절댓값을 계산하는 함수
def calculate_all_change_abs(group_data, columns):
    all_changes = {}

    for col in columns:
        # 전날 값과 현재 값의 차이를 절댓값으로 계산
        previous_values = group_data[col].shift(1)  # 전날 값
        current_values = group_data[col]  # 현재 값

        # 변화량 절댓값 계산
        change_amount = [
            abs(calculate_change_amount(current, previous))
            for current, previous in zip(current_values, previous_values)
        ]

        # 변화량 절댓값을 시리즈로 변환
        change_series = pd.Series(change_amount, index=group_data.index)  # 인덱스를 유지

        # 변화량 절댓값을 저장
        all_changes[col] = change_series

    return pd.DataFrame(all_changes)

# 종목조회건수를 기반으로 아웃라이어 탐지 및 아웃라이어 날의 변화량 절댓값을 구하는 함수
def calculate_changes_for_outliers(group_data, columns):
    changes = {}
    # 종목조회건수
    differences = group_data['종목조회건수']
                    
    # 종목조회에 대한 z-score 계산
    mean_diff = differences.mean()
    std_diff = differences.std()
    z_scores = (differences - mean_diff) / std_diff

    # z-score가 3 이상인 경우를 아웃라이어로 설정
    group_data['조회건수_아웃라이어'] = np.where(z_scores >= 0.7, 1, 0)

    # 모든 날의 변화량 절댓값 계산
    all_change_abs = calculate_all_change_abs(group_data, columns)

    # 전체 기간의 평균과 표준편차 계산 
    overall_mean = all_change_abs.mean()
    overall_std = all_change_abs.std()

    # 각 열에 대해 아웃라이어인 날의 변화량 추출 및 z-score 절댓값 계산
    for col in columns:
        outlier_days = group_data[group_data['조회건수_아웃라이어'] == 1]

        # 아웃라이어인 날의 변화량 절댓값 추출
        if not outlier_days.empty:
            outlier_changes = all_change_abs.loc[outlier_days.index, col].dropna()
            if not outlier_changes.empty:
                # 아웃라이어인 날의 z-score 절댓값 계산
                z_scores = (outlier_changes - overall_mean[col]) / overall_std[col]
                abs_z_scores_mean = z_scores.abs().mean()

                changes[col] = abs_z_scores_mean
            else:
                changes[col] = 0
        else:
            changes[col] = 0
    if type=='None':
        return changes
    else:
        return changes

화제성계산 = pd.merge(주식별고객정보, 종목일자별시세_보완, left_on=['티커종목코드', '기준일자'], right_on=['티커종목코드', '거래일자'])

# 종목별로 그룹화한 후, 아웃라이어 탐지 및 변화량 절댓값의 z-score 계산
grouped_zscore_means = 화제성계산[화제성계산['주식/ETF구분코드'] == '주식'].groupby('티커종목코드').apply(lambda group: calculate_changes_for_outliers(group.sort_values('기준일자'), ['매수체결합계수량', '매도체결합계수량']))

# 결과를 데이터프레임으로 변환
화제성_주식 = pd.DataFrame(grouped_zscore_means.tolist(), index=grouped_zscore_means.index)

화제성_주식.columns=['매수', '매도']

화제성_주식['화제성'] = 화제성_주식['매수'] + 화제성_주식['매도']

# 결과 출력
print(화제성_주식)

# 결과를 CSV로 저장 (선택사항)
화제성_주식.to_csv("화제성_주식.csv", index=True, encoding='cp949')

# ETF 화제성
# 종목별로 그룹화한 후, 아웃라이어 탐지 및 변화량 절댓값의 z-score 계산
grouped_zscore_means = 화제성계산[화제성계산['주식/ETF구분코드'] == 'ETF'].groupby('티커종목코드').apply(lambda group: calculate_changes_for_outliers(group.sort_values('기준일자'), ['매수체결합계수량', '매도체결합계수량']))

# 결과를 데이터프레임으로 변환
화제성_ETF = pd.DataFrame(grouped_zscore_means.tolist(), index=grouped_zscore_means.index)

화제성_ETF.columns=['매수', '매도']

화제성_ETF['화제성'] = 화제성_ETF['매수'] + 화제성_ETF['매도']

# 결과 출력
print(화제성_ETF)

# 결과를 CSV로 저장 (선택사항)
화제성_ETF.to_csv("화제성_ETF.csv", index=True, encoding='cp949')

뉴스 = pd.read_csv('final_news_data.csv', encoding='cp949')

뉴스['기준일자'] = pd.to_datetime(뉴스['기준일자'])

# 데이터프레임 로드
화제성검증 = 화제성계산.loc[화제성계산['티커종목코드'].isin(뉴스['티커종목코드'].unique()), ['티커종목코드', '기준일자', '종목조회건수', '매수체결합계수량', '매도체결합계수량']].copy()

화제성검증['기준일자'] = pd.to_datetime(화제성검증['기준일자'], format='%Y%m%d')

화제성검증 = pd.merge(화제성검증, 뉴스, on=['티커종목코드', '기준일자'], how='left')

화제성검증.fillna(0, inplace=True)

# 종목별로 z-score 계산
화제성검증['종목조회건수_zscore'] = 화제성검증.groupby('티커종목코드')['종목조회건수'].transform(
    lambda x: StandardScaler().fit_transform(x.values.reshape(-1, 1)).flatten())

best_f1 = 0
best_day = None
best_threshold = None

# n 값 범위 설정과 초기 threshold 범위 설정
n_values = [-2, -1, 0, 1, 2]
threshold_values = np.arange(0.5, 1.5, 0.05)  # 초기에는 넓은 간격으로 탐색

# threshold와 n 값을 이중 루프로 탐색
for threshold in threshold_values:
    for n in n_values:
        shifted_df = 화제성검증.copy()
        shifted_df['예측'] = shifted_df.groupby('티커종목코드').apply(
            lambda x: x.loc[(x.sort_values('기준일자')['종목조회건수_zscore'] >= threshold), '종목조회건수'].shift(n)
        ).reset_index(level=0, drop=True).fillna(0).astype(int)
        
        actual = 화제성검증['등장횟수']
        predictions = shifted_df['예측']
        
        f1 = pd.concat([actual, predictions], axis=1).corr().iloc[0, 1]
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_day = n

print("전체 종목에 적용할 최적의 z-score 임계값: ", best_threshold)
print("전체 종목에 적용할 최적의 일수 차이: ", best_day)
print("전체 종목에 적용할 최적의 precision score: ", best_f1)

# 해외종목정보 테이블 섹터분류명 누락값 복원
sectors=pd.merge(주식별고객정보[['티커종목코드', '주식/ETF구분코드']], 해외종목정보[['티커종목코드', '외화증권한글명', '섹터분류명', '시가총액']], on='티커종목코드', how='left').drop_duplicates()
missing_sectors=sectors[(sectors['섹터분류명']=='-')&(sectors['주식/ETF구분코드']=='주식')]
print(missing_sectors)

missing_tickers = missing_sectors['티커종목코드']

missing_sector_names = [
'Basic Materials', 'Basic Materials', 'Healthcare', 'Technology', 'Healthcare', 'Healthcare', 'Energy', 'Energy', 
'Basic Materials', 'Communication Services', 'Energy', 'Energy', 'Financial', 'Healthcare', 'Basic Materials', 'Energy', 
'Basic Materials', 'Financial', 'Healthcare', 'Energy', 'Healthcare', 'Real Estate', 'Healthcare', 'Basic Materials', 
'Basic Materials', 'Basic Materials', 'Energy', 'Basic Materials', 'Industrials', 'Energy', 'Energy', 'Basic Materials', 
'Energy', 'Basic Materials', 'Communication Services', 'Healthcare'
]

missing_caps = [
158.13, 4160, 832.06, 15020, 50.40, 201.73, 1820, 182.47, 13.18, 3520, 236.99, 1340, 52.05, 126.34, 2130, 458, 1460, 1710, 
89.72, 319.06, 21.86, 106.89, 193.45, 1470, 404.23, 962.45, 892.98, 644.11, 36.96, 2840, 431.11, 113.40, 990.09, 50.75, 42.29, 121.45
]

for i, ticker in enumerate(missing_tickers):
    해외종목정보.loc[해외종목정보['티커종목코드']==ticker, '섹터분류명'] = missing_sector_names[i]
    해외종목정보.loc[해외종목정보['티커종목코드']==ticker, '시가총액'] = missing_caps[i]

해외종목정보.to_csv('해외종목정보_보완.csv', index=False, encoding='cp949')


# ETF 섹터 추가 해당 파트는 크롤링과 핸드라벨링을 병했했습니다. 해당 과정에서 실행 시간에 따라 값이 변하므로 주석처리 합니다. 값이 달라 오류가 발생할 수 있습니다.
'''
# Data 정의 (152개의 '티커종목코드'와 '섹터')
data = {
    '티커종목코드': [
        'AAPB', 'AAPU', 'AIQ', 'AIYY', 'AMDL', 'AMDY', 'AMZU', 'AMZY', 'AOR', 'APLY', 'ARKF', 'ARKG',
        'ARKK', 'ARKQ', 'ARKW', 'ARKX', 'AWAY', 'BOTZ', 'CIBR', 'CLOU', 'CONL', 'CONY', 'COPX',
        'CWEB', 'DFEN', 'DGRO', 'DGRW', 'DIA', 'DISO', 'DIV', 'DIVO', 'DPST', 'DRIV', 'EETH',
        'EWY', 'FAS', 'FBL', 'FBY', 'FXI', 'GDX', 'GGLL', 'GLDM', 'GOOY', 'GPIQ', 'GPIX', 'GRID',
        'HDRO', 'IBB', 'ICLN', 'IEMG', 'IHI', 'IJH', 'IJR', 'ITA', 'IVV', 'IWM', 'JEPI', 'JEPQ',
        'JETS', 'JPMO', 'KBWY', 'KLIP', 'KORU', 'KRBN', 'KRE', 'KWEB', 'LABU', 'LIT', 'METV',
        'MGK', 'MOAT', 'MRNY', 'MSFO', 'MSFU', 'MSOS', 'MSOX', 'NAIL', 'NFLY', 'NOBL', 'NUSI',
        'NVD', 'NVDL', 'NVDU', 'NVDY', 'OARK', 'PAVE', 'PFF', 'PGX', 'PRNT', 'PYPY', 'QCLN',
        'QLD', 'QQQ', 'QQQM', 'QYLD', 'RETL', 'RPAR', 'RYLD', 'SCHD', 'SDIV', 'SKYY', 'SMH',
        'SOF', 'SOXL', 'SOXQ', 'SOXX', 'SPHD', 'SPLG', 'SPXL', 'SPY', 'SPYD', 'SPYG', 'SPYV',
        'SQY', 'SSO', 'SVOL', 'TAN', 'TECL', 'TLT', 'TLTW', 'TMF', 'TNA', 'TQQQ', 'TSL', 'TSLL',
        'TSLY', 'UDOW', 'UPRO', 'URA', 'URTY', 'USD', 'VEA', 'VIG', 'VNQ', 'VOO', 'VT', 'VTI',
        'VWO', 'VYM', 'WEBL', 'XBI', 'XLE', 'XLF', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XOMO',
        'XYLD', 'YINN', 'YMAX'
    ],

    '섹터분류명': [
        'Technology', 'Technology', 'Technology', 'Technology', 'Technology', '-', 'Consumer Cyclical',
        'Consumer Discretionary', '-', 'Technology', '-', 'Healthcare', '-', '-', '-', 'Industrials',
        'Consumer Cyclical', '-', 'Technology', 'Technology', 'Financial Services', 'Technology',
        'Basic Materials', '-', 'Industrials', '-', '-', '-', '-', '-', '-', 'Financial Services', '-',
        '-', '-', 'Financial Services', 'Communication Services', 'Communication Services', '-',
        'Basic Materials', 'Communication Services', 'Basic Materials', 'Communication Services',
        'Technology', '-', 'Industrials', 'Industrials', 'Healthcare', 'Utilities', '-', 'Healthcare',
        '-', '-', 'Industrials', '-', '-', '-', 'Technology', 'Industrials', 'Financial Services',
        'Real Estate', '-', '-', '-', 'Financial Services', '-', 'Healthcare', '-', 'Technology',
        'Technology', '-', 'Healthcare', 'Technology', 'Technology', 'Real Estate', '-', 'Consumer Cyclical',
        'Communication Services', '-', 'Technology', '-', 'Financial Services', 'Technology', '-',
        '-', 'Industrials', 'Utilities', 'Financial Services', 'Technology', '-', '-', 'Technology',
        'Technology', 'Technology', 'Technology', 'Consumer Cyclical', '-', '-', '-', '-', 'Technology',
        'Technology', '-', 'Technology', 'Technology', 'Technology', '-', '-', '-', '-', '-', 'Technology',
        '-', '-', '-', '-', 'Technology','Technology', '-', '-', '-', '-','Technology', 'Consumer Cyclical', 'Consumer Cyclical', 'Consumer Cyclical',
        '-', '-', 'Energy', '-', 'Technology', '-', '-', 'Real Estate', '-', '-', '-', '-', '-', '-',
        'Healthcare', 'Energy', 'Financial Services', 'Technology', 'Consumer Defensive', 'Real Estate',
        'Utilities', 'Healthcare', 'Energy', '-', '-', '-'
    ]
}

# DataFrame 생성 및 CSV 저장
섹터분류_일부 = pd.DataFrame(data)

섹터분류_ETF = pd.DataFrame(주식별고객정보[주식별고객정보['주식/ETF구분코드']=='ETF']['티커종목코드'].unique(), columns=['티커종목코드'])


섹터분류_ETF = pd.merge(섹터분류_ETF, 섹터분류_일부, how='left', on='티커종목코드')

def clean_sector_name(sector_str):
    """섹터 이름에서 불필요한 부분 제거."""
    return sector_str.split('.')[0].strip()

def get_stock_info_selenium(ticker):
    # Chrome 옵션 생성 (헤드리스 모드 활성화)
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # 백그라운드에서 실행
    # 웹 드라이버 실행
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f"https://finance.yahoo.com/quote/{ticker}/holdings/")
    
    driver.implicitly_wait(1)  # 페이지 로딩 대기
    
    try:
        # 섹터 분류명 가져오기
        sector_percent = driver.find_element(By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[2]/section[2]/div/div[2]/span[2]').text
        if int(clean_sector_name(sector_percent)) >= 50:
            sector = driver.find_element(By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[2]/section[2]/div/div[2]/a').text
        else:
            sector = 'Under 50'

    except Exception as e:
        # 오류 발생 시 None으로 처리
        print(f"Error occurred: {e}")
        sector = np.nan

    # 웹 드라이버 종료
    driver.quit()
    
    return sector

while True:
    missing_sectors = 섹터분류_ETF[섹터분류_ETF['섹터분류명'].isna()]['티커종목코드']
    for ticker in missing_sectors:
        섹터분류_ETF.loc[섹터분류_ETF['티커종목코드']==ticker, '섹터분류명'] = get_stock_info_selenium(ticker)
    break
    
sector_list = ['-', '-', 'Technology', 'Consumer Discretionary', '-',
    '-', '-', 'Technology', 'Technology', '-', '-', '-', '-', '-', '-', '-', 'Basic Materials', '-',
    '-', 'Financial Services', 'Real Estate', 'Financial Services', 'Financial Services', 'Energy', 
    '-', '-', 'Real Estate', '-', '-', '-', '-', '-', '-', '-', '-', 'Technology',
    'Real Estate', '-', '-', '-', '-', '-', 'Technology', 'Technology', '-', '-', '-', '-', 'Utilities', '-', '-', '-', 'Technology']

len(sector_list)

섹터분류_ETF.loc[[
    3, 6, 9, 17, 35, 37, 41, 46, 47, 56, 57, 73, 88, 101, 133, 150, 151, 159, 
    161, 163, 166, 182, 202, 205, 216, 217, 219, 237, 257, 258, 259, 260, 261, 
    269, 280, 290, 295, 301, 312, 318, 352, 356, 360, 363, 365, 366, 373, 377, 
    378, 382, 383, 392, 439
], '섹터분류명'] = np.nan

for i, ticker in enumerate(섹터분류_ETF[섹터분류_ETF['섹터분류명'].isna()]['티커종목코드']):
    섹터분류_ETF.loc[섹터분류_ETF['티커종목코드']==ticker, '섹터분류명'] = sector_list[i]

섹터분류_ETF.loc[섹터분류_ETF['섹터분류명']=='Under 50', '섹터분류명'] = '-'

miss_sec = {'ACES' : 'Energy', 'ACP' : '회사채', 'ACWI' : '-', 'AFK' : '아프리카', 'AGG' : '미국 채권', 'AIEQ' : '미국 주식',
 'ALTY' : 'Real Estate', 'AMDY' : 'AMD', 'AOA' : '-', 'AOK' : '-', 'AOM' : '-', 'AOR' : '-', 'ARGT' : '아르헨티나 주식',
 'ARKF' : 'Technology', 'ARKK' : 'Technology', 'ARKQ' : 'Technology', 'ARKW' : 'Technology', 'ASHR' : '중국 주식', 'BALI' : '미국 주식',
 'BATT' : 'Energy', 'BHK' : '-', 'BIL' : '미국 채권', 'BKLN' : '-', 'BLCN' : 'Technology', 'BLOK' : 'Technology',
 'BOTZ' : 'Technology', 'BRZU' : '브라질', 'BUZZ' : '미국 주식', 'CATH' : 'S&P 500', 'CDC' : '미국 주식', 'CHAU' : '중국 주식', 
 'CII' : '-', 'CIK' : '-', 'CNRG' : 'Energy', 'COWZ' : '미국 주식', 'CRBN' : 'Energy', 'CTEC' : 'Energy', 'CWEB' : '중국',
 'DDM' : '다우존스', 'DFAC' : '미국 주식', 'DGRO' : '미국 주식', 'DGRW' : '미국 채권', 'DHY' : '-', 'DIA' : '다우존스',
 'DISO' : 'DIS', 'DIV' : '미국 주식', 'DIVB' : '미국 주식', 'DIVO' : '미국 주식', 'DJD' : '다우존스', 'DRIV' : 'Technology',
 'DVY' : '미국 주식', 'EDC' : '-', 'EDV' : '미국 국채', 'EEM' : '-', 'EETH' : 'Basic Materials', 'EFA' : '-', 'EFIV' : 'Energy',
 'ESGD' : '-', 'ESGE' : '-', 'ESGU' : '미국 주식', 'ESGV' : '미국 주식', 'EVV' : '-', 'EWA' : '호주 주식', 'EWG' : '독일 주식',
 'EWJ' : '일본 주식', 'EWW' : '멕시코 주식', 'EWY' : '한국 주식', 'EWZ' : '브라질 주식', 'EZA' : '남아프리카 주식', 'EZU' : '유럽 주식',
 'FDN' : 'Technology', 'FEZ' : '유럽 주식', 'FNDE' : '-', 'FPX' : '미국 주식',  'FXI' : '중국 주식' , 'GOF' : '-',
 'GPIX' : ' 미국 주식', 'HAIL' : '미국 주식', 'HDV' : '미국 주식' , 'HIBL' : 'S&P 500', 'HNDL' : '-', 'HYT' : '미국 채권',
 'IDV': '-', 'IEF': '미국 채권', 'IEFA' : '-', 'IEI':'미국 채권', 'IEMG': '-', 'IJH' : 'S&P 500' , 'IJR' : 'S&P 500' ,
 'INDA' : '인도 주식', 'INDL' : '인도 주식', 'IOO' : '-', 'ITOT': '미국 주식', 'IVE' : 'S&P 500' , 'IVV' : 'S&P 500',
 'IWD' : '러셀2000', 'IWM' : '러셀2000', 'IWN': '러셀2000', 'IWO' : '-', 'IXUS' : '-', 'JEPI' : '미국 주식', 'KARS' : 'Technology',
 'KLIP' : 'Technology', 'KOMP' : 'Technology', 'KORU' : '한국 주식',  'KRBN' : '-', 'KROP' : 'Consumer Defensive',
 'KSA' : ' 사우디아라비아 주식', 'KWEB' : 'Technology' , 'LIT' : 'Technology', 'LQD' : '미국 채권', 'LQDW' : '미국 채권', 
 'MGC' : '미국 주식', 'MGV' : '미국 주식', 'MIDU' : '미국 주식', 'MILN' : '미국 주식', 'MOAT' : '미국 주식', 'MOO' : 'Consumer Defensive' ,
 'MSOX' : 'Consumer Discretionary' , 'MTUM' : '미국 주식', 'NCV' : '-', 'NLR' : 'Energy' , 'NOBL' : '미국 주식',
 'NVD' : 'Technology', 'NVDY' : '미국 주식', 'OARK' : 'Technology', 'OEF' : 'S&P 500', 'PAWZ' : '-', 'PBD' : 'Energy',
 'PBUS' : '미국 주식', 'PBW' : 'Energy', 'PCN' : '미국 채권', 'PDBC' : 'Consumer Defensive' , 'PDI': '-', 'PDO' : '-',
 'PDT' : '-', 'PEY' : '미국 주식', 'PHK' : '-', 'PID' : '-', 'PIN' : '인도 주식', 'PTY' : '-', 'PYPY' : 'Financial Services',
 'QCLN' : 'Energy' , 'QQQA' : '미국 주식', 'QQQE' : '미국 주식', 'QQQJ' : 'Technology', 'QUAL' : '미국 주식', 'RIV' : '-',
 'RPAR' : '-', 'RSP' : 'S&P 500', 'RYLD' : '러셀2000', 'SCHB' : '미국 주식', 'SCHD' : '미국 주식', 'SCHP' : 'TIPS',
 'SCHX' : '미국 주식', 'SCHY' : '-', 'SDIV' : '-', 'SDY' : 'S&P 500', 'SHY' : '미국 채권', 'SNPE' : 'S&P 500', 'SOF' : '-',
 'SPGP' : 'S&P 500', 'SPHD' : 'S&P 500', 'SPHQ' : 'S&P 500', 'SPLG' : 'S&P 500', 'SPLV' : 'S&P 500', 'SPXL' : 'S&P 500',
 'SPY' : 'S&P 500', 'SPYD' : 'S&P 500', 'SPYV' : 'S&P 500', 'SQY' : 'S&P 500','SRS' : 'Real Estate', 'SSO' : 'S&P 500', 
 'SUSA' : 'MSCI USA 지수', 'SUSL' : 'MSCI USA 지수', 'SVOL' : '-', 'TIP' : 'TIPS', 'TLT' : '미국 채권', 
 'TLTW' : '미국 채권', 'TMF' : '미국 채권','TMV' : '미국 채권', 'TNA' : '러셀2000', 'TUA' : '-', 'TYD' : '미국 채권', 
 'UDOW' : '다우존스', 'UPRO' : 'S&P 500','URTH' : 'MSCI World',  'URTY' : '러셀2000', 'USFR' : '미국 채권', 'USMV' : '미국 주식', 
 'UTF' : 'Utilities', 'UWM' : '러셀2000', 'VCIT' : '미국 채권', 'VCLT' : '미국 채권', 'VEA' : '-', 'VEGI' : 'Consumer Defensive',
 'VEU' : '-', 'VGK' : '유럽 주식', 'VGLT' : '미국 채권', 'VIG' : '미국 주식', 'VIGI' : '-', 'VOO' : 'S&P 500',
 'VOOV' : 'S&P 500', 'VT' : '-', 'VTI' : '미국 주식', 'VTV' : 'S&P 500', 'VTWO' : '러셀2000', 'VUG' : 'S&P 500', 'VWO' : '-',
 'VXUS' : '-', 'VYM' : '미국 주식', 'VYMI' : '-', 'WEBL' : '다우존스', 'XLG' : 'S&P 500', 'XYLD' : 'S&P 500',
 'YINN' : '중국 주식', 'YMAX' : 'S&P 500', 'YYY' : '-'}

섹터분류_ETF['섹터분류명'] = 섹터분류_ETF['티커종목코드'].map(miss_sec).fillna(섹터분류_ETF['섹터분류명'])
'''
name_dict = {'Technology': '기술 관련주',
'Healthcare': '헬스케어 관련주',
'Consumer Cyclical': '경기 소비재 관련주',
'Consumer Discretionary': '임의 소비재 관련주',
'Industrials': '산업재 관련주',
'Financial Services': '금융 서비스 관련주',
'Basic Materials': '기초 소재 관련주',
'Real Estate': '부동산 관련주',
'Energy': '에너지 관련주',
'Communication Services': '통신 서비스 관련주',
'Utilities': '공공사업 관련주',
'Consumer Defensive': '필수 소비재 관련주'}

'''
섹터분류_ETF['섹터분류명'] = 섹터분류_ETF['섹터분류명'].map(name_dict).fillna(섹터분류_ETF['섹터분류명'])

print(섹터분류_ETF['섹터분류명'].unique())

output_path = '섹터분류_ETF.csv'
섹터분류_ETF.to_csv(output_path, encoding='cp949', index=False)
print(f"CSV 파일이 '{output_path}' 경로에 저장되었습니다.")

'''

# 베타계수, 수익률표준편차
# 전날 대비 수익률 계산 함수
def calculate_returns(df, return_col, date_col, stock_col):
    # 종목별로 데이터를 정렬하고, 전날 수익률과의 차이로 계산
    df = df.sort_values([stock_col, date_col])
    df['전날종가'] = df.groupby(stock_col)[return_col].shift(1)  # 전날의 수익률
    df['진짜수익률'] = (df[return_col] - df['전날종가']) / df['전날종가']  # 수익률 차이 계산
    return df

# 섹터별 수익률 계산 함수 (기존)
def calculate_sector_return(df, date_col, sector_col, return_col, market_cap_col):
    # NaN 및 시가총액이 0인 행 제거
    df_filtered = df[df[market_cap_col] > 0].dropna(subset=[return_col, market_cap_col])
    
    # 날짜와 섹터별로 그룹화하여 매일의 섹터 수익률 계산
    daily_sector_returns = df_filtered.groupby([date_col, sector_col]).apply(
        lambda x: (x[return_col] * x[market_cap_col]).sum() / x[market_cap_col].sum()
    ).reset_index()
    
    daily_sector_returns.columns = [date_col, sector_col, '섹터수익률']
    return daily_sector_returns

# 종목별 지표 계산 함수 (기존)
def calculate_beta_per_stock(df, stock_col, stock_return_col, sector_return_col, sector_col):
    betas = {}
    stds = {}
    sectors = {}

    # 각 종목별로 지표 계산
    for stock in df[stock_col].unique():
        stock_data = df[df[stock_col] == stock].dropna(subset=stock_return_col).sort_values('BSE_DT')
        
        X = stock_data[sector_return_col]  # 섹터 수익률
        y = stock_data[stock_return_col]   # 종목 수익률
        X = sm.add_constant(X)  # 상수항 추가
        
        # 베타 계산을 위한 회귀 분석 수행
        model = sm.OLS(y, X).fit()
        
        if len(model.params) > 1:
            beta = model.params.iloc[1]  # 위치 기반 접근 수정
            betas[stock] = beta # 베타
        else:
            betas[stock] = np.nan
        
        # 수익률표준편차 계산
        stds[stock] = stock_data[stock_return_col].std() # 표준편차

        # 섹터 분류 계산
        if len(stock_data[sector_col].unique())==1:
            sectors[stock] = stock_data[sector_col].drop_duplicates().values[0]
        else:
            sectors[stock] = 'error'

    return betas, stds, sectors

지표계산=pd.merge(주식일별정보_보완[['티커종목코드', 'BSE_DT', '기준종가', '당사평균수익율']], 해외종목정보[['티커종목코드', '주식/ETF구분코드', '섹터분류명', '시가총액']], how='left', on='티커종목코드')
# 베타계산[(베타계산['BSE_DT']==20240528)&(베타계산['섹터분류명']=='-')]['티커종목코드'].to_csv('섹터결측.csv', index=False, encoding='cp949')

# 전날 대비 수익률 계산
지표계산2 = calculate_returns(지표계산, '기준종가', 'BSE_DT', '티커종목코드')

# 매일 섹터 수익률 계산
daily_sector_returns = calculate_sector_return(지표계산2, 'BSE_DT', '섹터분류명', '진짜수익률', '시가총액')

# 원래 데이터와 병합
지표계산3 = pd.merge(지표계산2, daily_sector_returns, on=['BSE_DT', '섹터분류명'])

# 종목별 베타 계산
betas_per_stock, stds_per_stock, sectors_per_stock = calculate_beta_per_stock(지표계산3, '티커종목코드', '진짜수익률', '섹터수익률', '섹터분류명')

# 베타계수 결과 저장
베타계수_주식 = pd.DataFrame(list(betas_per_stock.items()), columns=['티커종목코드', '베타계수'])
베타계수_주식.to_csv("베타계수_주식.csv", index=False, encoding='cp949')

# 표준편차 결과 저장
표준편차_주식 = pd.DataFrame(list(stds_per_stock.items()), columns=['티커종목코드', '수익률표준편차'])
표준편차_주식.to_csv("표준편차_주식.csv", index=False, encoding='cp949')

# 섹터분류 결과 저장
섹터분류_주식 = pd.DataFrame(list(sectors_per_stock.items()), columns=['티커종목코드', '섹터분류명'])
섹터분류_주식.to_csv("섹터분류_주식.csv", index=False, encoding='cp949')

# ETF 베타계수, 수익률 표준편차

daily_sector_returns['섹터분류명'] = daily_sector_returns['섹터분류명'].map(name_dict).fillna(daily_sector_returns['섹터분류명'])

# ETF별 섹터분류 데이터 로드
ETF섹터분류=pd.read_csv('섹터분류_ETF.csv', encoding='cp949')

# ETF 종가와 날짜에 섹터분류명 추가
ETF지표계산=pd.merge(주식일별정보_보완[['BSE_DT', '티커종목코드', '기준종가']], ETF섹터분류, on='티커종목코드')

# 섹터 수익률 추가
ETF지표계산2=pd.merge(ETF지표계산, daily_sector_returns, on=['BSE_DT', '섹터분류명'], how='left')

# 함수 정의: 섹터분류명이 '-'인 경우 같은 날짜의 다른 섹터분류명의 섹터수익률 평균을 채워주는 함수
def fill_missing_sector_returns(df):
    
    # 섹터분류명이 '-'인 행들을 찾음
    missing_sector_rows = df[df['섹터분류명'].isin(name_dict.values())]
    
    # 같은 날짜의 다른 섹터들의 섹터수익률 평균으로 채우기
    for idx, row in missing_sector_rows.iterrows():
        same_date_rows = df[(df['BSE_DT'] == row['BSE_DT']) & (df['섹터분류명'].isin(name_dict.values()))]
        
        if not same_date_rows.empty:
            # 섹터수익률의 평균을 계산하고 결측값 채우기
            mean_sector_return = same_date_rows['섹터수익률'].mean()
            df.at[idx, '섹터수익률'] = mean_sector_return

    return df

# 함수를 실행하여 섹터수익률을 채운 데이터프레임
ETF지표계산3 = fill_missing_sector_returns(ETF지표계산2)

# 수익률 추가
ETF지표계산4 = calculate_returns(ETF지표계산3, '기준종가', 'BSE_DT', '티커종목코드')

# 지표 계산
betas_per_etf, stds_per_etf, sectors_per_etf =calculate_beta_per_stock(ETF지표계산4.dropna(), '티커종목코드', '진짜수익률', '섹터수익률', '섹터분류명')

# 베타계수 결과 저장
베타계수_ETF = pd.DataFrame(list(betas_per_etf.items()), columns=['티커종목코드', '베타계수'])
베타계수_ETF.to_csv("베타계수_ETF.csv", index=False, encoding='cp949')

# 표준편차 결과 저장
표준편차_ETF = pd.DataFrame(list(stds_per_etf.items()), columns=['티커종목코드', '수익률표준편차'])
표준편차_ETF.to_csv("표준편차_ETF.csv", index=False, encoding='cp949')

# 투자심리지수
# 주식일별정보 데이터프레임에서 필요한 열을 가져옵니다.
# 거래일자와 종목코드별로 그룹화하여 RSI를 계산

# 기존데이터유지
rsi = 주식일별정보_보완.copy()

# '거래일자'를 datetime 형식으로 변환 (필요한 경우)
rsi['BSE_DT'] = pd.to_datetime(rsi['BSE_DT'], format='%Y%m%d')

# '거래일자' 기준으로 정렬
rsi = rsi.sort_values(by=['티커종목코드', 'BSE_DT'])

def calculate_rsi(data, window=14):
    # 종가의 변화량을 구합니다.
    delta = data['기준종가'].diff()

    # 상승폭과 하락폭을 구합니다.
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # 이동 평균 계산 (기본은 14일 기준)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()


    # RSI 계산
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# 종목별로 그룹화하여 RSI 계산
rsi['RSI'] = rsi.groupby('티커종목코드', group_keys=False).apply(calculate_rsi)

# 결과 확인
rsi_결측치존재 = rsi[['티커종목코드', 'BSE_DT', '기준종가', 'RSI']]
rsi_결측치존재

## 14일을 기준으로 이동평균을 구했기 때문에, 0528~0613까지 결측치가 존재함
## yfinance 라이브러리를 통해 2024년 5월의 티커종목코드별 종가데이터를 불러와 결측치를 채울 예정

### 결측치 채우기 : yfinance 라이브러리에서 앞의 14일 날짜 종가 데이터 불러오기

# 종목 코드와 날짜 범위 설정
start_date = '2024-05-01'
end_date = '2024-05-27'

# 티커 코드 리스트 가져오기
tickers = 주식일별정보_보완['티커종목코드'].unique()

# 데이터프레임을 저장할 빈 리스트 생성
data_frames = []

# 각 티커에 대해 yfinance로 데이터 가져오기
for ticker in tickers:
    try:
        # yfinance에서 데이터 다운로드
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # 필요한 데이터만 필터링 (종가)
        stock_data['티커종목코드'] = ticker  # 티커 코드 추가
        stock_data.reset_index(inplace=True)  # 인덱스를 초기화하여 'Date' 컬럼을 일반 컬럼으로 만듭니다.
        
        # 2024년 5월 14일부터 5월 27일까지의 데이터만 선택
        stock_data_filtered = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
        
        # 결과를 리스트에 추가
        data_frames.append(stock_data_filtered)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# 모든 데이터프레임을 하나로 합치기
yfinance_05 = pd.concat(data_frames, ignore_index=True)

# 열 이름 변경 -> 날짜,티커코드,종가만뽑기 (필요한 데이터만 가져오기)
yfinance_05.rename(columns={'Date': 'BSE_DT', 'Close': '기준종가'}, inplace=True)
yfinance_05 = yfinance_05[['BSE_DT','티커종목코드','기준종가']]

# 불러온 데이터를 csv 파일로 저장장
yfinance_05.to_csv('yfinance_data05.csv',encoding='cp949',index=False)

yfinance_data = pd.read_csv('yfinance_data05.csv',encoding='cp949')

# 거래일자 컬럼을 datetime 형식으로 변환 (시간 정보 제거)
rsi_결측치존재['BSE_DT'] = pd.to_datetime(rsi_결측치존재['BSE_DT']).dt.date
yfinance_data['BSE_DT'] = pd.to_datetime(yfinance_data['BSE_DT']).dt.date

# 결측치를 채우기 위해 두 데이터를 합치기
rsi_data = pd.concat([rsi_결측치존재, yfinance_data])
rsi_data = rsi_data.sort_values(by=['티커종목코드', 'BSE_DT'], ascending=[True, True])

# RSI 계산 함수
def calculate_rsi(data, window=14):
    # 종가의 변화량을 구합니다.
    delta = data['기준종가'].diff()

    # 상승폭과 하락폭을 구합니다.
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # 이동 평균 계산 (기본은 14일 기준)
    avg_gain = gain.rolling(window=window, min_periods=14).mean()
    avg_loss = loss.rolling(window=window, min_periods=14).mean()

    # RSI 계산
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# 종목별로 그룹화하여 RSI 계산 (날짜 정렬된 상태에서)
rsi_data['RSI'] = rsi_data.groupby('티커종목코드', group_keys=False).apply(calculate_rsi)

# '거래일자'를 datetime 형식으로 변환
rsi_data['BSE_DT'] = pd.to_datetime(rsi_data['BSE_DT'])

# 특정 날짜 범위의 데이터를 제거
rsi_data = rsi_data[~rsi_data['BSE_DT'].between('2024-05-01', '2024-05-27')]

print(rsi_data)

## 주식일별정보 데이터 프레임에서 신규매수계좌수, 전량매도계좌수를 이용하여 bsi 계산

# 기존 데이터 유지
bsi = 주식일별정보_보완.copy()

# '거래일자'를 datetime 형식으로 변환 (필요한 경우)
bsi['BSE_DT'] = pd.to_datetime(bsi['BSE_DT'], format='%Y%m%d')

# '거래일자' 기준으로 정렬
bsi = bsi.sort_values(by=['티커종목코드', 'BSE_DT'])

# BSI 계산 함수
def calculate_bsi(bsi):
    # 총 매수량을 신규매수계좌수로 설정
    total_buy_accounts = bsi['신규매수계좌수']
    
    # 순매수량: 신규매수계좌수 - 전량매도계좌수
    net_buy_accounts = bsi['신규매수계좌수'] - bsi['전량매도계좌수']
    
    # BSI 계산
    bsi = total_buy_accounts / (2 * total_buy_accounts - net_buy_accounts)
    
    return bsi

# BSI 데이터프레임에 BSI 값을 추가
bsi['BSI'] = calculate_bsi(bsi)

# 'bsi' 데이터프레임에서 '신규매수계좌수'가 0인 경우 BSI의 NaN 값을 0으로 대체
## 결측치 처리 완료
bsi.loc[bsi['신규매수계좌수'] == 0, 'BSI'] = bsi['BSI'].fillna(0)

## bsi_data 데이터 프레임 생성성
bsi_data = bsi[['BSE_DT','티커종목코드','BSI']]

# 'bsi' 데이터프레임
print(bsi_data)

# rsi와 bsi 데이터프레임을 거래일자와 티커종목코드를 기준으로 합치기
투자심리지수 = pd.merge(rsi_data, bsi_data, on=['BSE_DT', '티커종목코드'], how='inner')
투자심리지수

# 정규화 함수 정의
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())
# RSI, BSI, Turnover 정규화
투자심리지수['RSI_normalized'] = min_max_normalize(투자심리지수['RSI'])
투자심리지수['BSI_normalized'] = min_max_normalize(투자심리지수['BSI'])

# 투자심리지수 계산
투자심리지수['투자심리지수'] = (
    투자심리지수['RSI_normalized'] + 
    투자심리지수['BSI_normalized']
)

### 종목별 rsi, bsi, 투자심리지수 계산 ()
average_investment_rsi = 투자심리지수.groupby('티커종목코드')['RSI_normalized'].mean().reset_index()
average_investment_bsi = 투자심리지수.groupby('티커종목코드')['BSI_normalized'].mean().reset_index()
average_investment_sentiment = 투자심리지수.groupby('티커종목코드')['투자심리지수'].mean().reset_index()

# '티커종목코드'를 기준으로 데이터프레임 결합
종목별_투자심리지수 = average_investment_rsi.merge(average_investment_bsi, on='티커종목코드')
종목별_투자심리지수 = 종목별_투자심리지수.merge(average_investment_sentiment, on='티커종목코드')

## 주식별고객정보에서 '주식'인 티커종목코드의 유일한 값-> 주식별고객정보_주식
주식별고객정보_주식 = pd.DataFrame(주식별고객정보.loc[주식별고객정보['주식/ETF구분코드'] == '주식', '티커종목코드'].unique(), columns=['티커종목코드'])

## 주식별고객정보에서 'ETF'인 티커종목코드의 유일한 값 -> 주식별고객정보_ETF
주식별고객정보_ETF = pd.DataFrame(주식별고객정보.loc[주식별고객정보['주식/ETF구분코드'] == 'ETF', '티커종목코드'].unique(), columns=['티커종목코드'])

# ETF와 주식 티커 목록을 각각 나누기
etf_tickers = 주식별고객정보_ETF['티커종목코드']
stock_tickers = 주식별고객정보_주식['티커종목코드']

# 투자심리지수 데이터프레임에서 ETF와 주식으로 나누기
투자심리지수_ETF = 종목별_투자심리지수[종목별_투자심리지수['티커종목코드'].isin(etf_tickers)]
투자심리지수_주식 = 종목별_투자심리지수[종목별_투자심리지수['티커종목코드'].isin(stock_tickers)]

# 원하는 열만 추출하기
투자심리지수_주식 = 투자심리지수_주식[[ '티커종목코드', '투자심리지수']]
투자심리지수_ETF = 투자심리지수_ETF[[ '티커종목코드', '투자심리지수']]

# csv 파일로 저장
투자심리지수_주식.to_csv('투자심리지수_주식.csv',index=False,encoding='cp949')
투자심리지수_ETF.to_csv('투자심리지수_ETF.csv',index=False,encoding='cp949')

# 지표저장

# 데이터프레임 리스트 생성
dfs = [베타계수_주식, 표준편차_주식, 화제성_주식, 투자심리지수_주식, 섹터분류_주식]

# 공통 키로 병합하는 함수 (on='공통키'로 병합할 열 지정)
주식별지표 = reduce(lambda left, right: pd.merge(left, right, on='티커종목코드'), dfs)

주식별지표.to_csv('주식별지표.csv', index=False, encoding='cp949')

# 종목조회건수 - 종가 변화율

# ETF 티커를 필터링하는 함수
def filter_etf_tickers(stock_customer_info, stock_daily_info):
    etf_ticker_codes = stock_customer_info[stock_customer_info['주식/ETF구분코드'] == 'ETF']['티커종목코드'].unique()
    filtered_etf_info = stock_daily_info[stock_daily_info['티커종목코드'].isin(etf_ticker_codes)].sort_values(by=['티커종목코드', 'BSE_DT']).reset_index(drop=True)
    return filtered_etf_info

# 종목조회건수 증가 및 기준종가 변화 계산 함수
def calculate_changes(filtered_data):
    # 종목조회건수 증가 계산
    filtered_data['종목조회건수_증가'] = filtered_data.groupby('티커종목코드')['종목조회건수'].diff().fillna('-')

    # 기준종가 변화 계산
    filtered_data['기준종가_변화'] = filtered_data.groupby('티커종목코드')['기준종가'].diff().fillna('-')

    # 절댓값 적용
    filtered_data['기준종가_변화'] = filtered_data['기준종가_변화'].apply(lambda x: abs(x) if isinstance(x, (int, float)) else x)
    filtered_data['종목조회건수_증가'] = filtered_data['종목조회건수_증가'].apply(lambda x: abs(x) if isinstance(x, (int, float)) else x)

    return filtered_data

# 날짜 및 NaN 값 처리 함수
def process_dates_and_nan(data):
    # 날짜 변환
    data['BSE_DT'] = pd.to_datetime(data['BSE_DT'], format='%Y%m%d')

    # NaN 값을 0으로 처리
    data['기준종가_변화'] = pd.to_numeric(data['기준종가_변화'], errors='coerce').fillna(0)
    data['종목조회건수_증가'] = pd.to_numeric(data['종목조회건수_증가'], errors='coerce').fillna(0)

    return data

# 스케일링 및 상관계수 계산 함수
def calculate_correlations(data):
    scaler = StandardScaler()

    # 스케일링 적용
    data[['종목조회건수_증가_scaled', '기준종가_변화_scaled']] = scaler.fit_transform(data[['종목조회건수_증가', '기준종가_변화']])

    # 티커별 상관계수 계산
    correlation_dict = {}
    for ticker, group_data in data.groupby('티커종목코드'):
        corr = group_data[['종목조회건수_증가_scaled', '기준종가_변화_scaled']].corr().iloc[0, 1]
        correlation_dict[ticker] = corr

    # 상관계수 데이터프레임 생성
    correlation_df = pd.DataFrame(list(correlation_dict.items()), columns=['티커종목코드', '상관계수'])

    return correlation_df

# 전체 과정 함수
def process_etf_data(stock_customer_info, stock_daily_info):
    # 1. ETF 티커 필터링
    filtered_data = filter_etf_tickers(stock_customer_info, stock_daily_info)

    # 2. 종목조회건수 증가 및 기준종가 변화 계산
    filtered_data = calculate_changes(filtered_data)

    # 3. 날짜 및 NaN 값 처리
    filtered_data = process_dates_and_nan(filtered_data)

    # 4. 상관계수 계산
    correlation_df = calculate_correlations(filtered_data)

    return correlation_df, filtered_data

# 실제 데이터로 함수 실행
correlation_df, 전환율_테이블_종가 = process_etf_data(주식별고객정보, 주식일별정보)
correlation_df.to_csv('종가_조회수_상관관계.csv', encoding='cp949', index=False)

# 상관계수 분포 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(correlation_df['상관계수'], bins=20, color='blue', edgecolor='black')

# 그래프 제목과 축 레이블 설정
plt.title('상관계수 분포')
plt.xlabel('상관계수')
plt.ylabel('빈도')

# 그래프 보여주기
plt.tight_layout()
plt.show()

# 종목조회건수 - 종가변화율 시각화
# 종목조회건수가 높은 것은 종목이 화제가 됐다고 가정하였습니다. (뉴스기사데이터를 통해 검증 완료)
# '종목조회건수-종가변화율' 지표를 통해 종목조회와 실제 구매(종가) 사이의 상관관계를 파악하였습니다.

# 시각화 진행할 5개  티커종목코드
desired_tickers = ['KLIP', 'SCHD', 'SPY', 'VTI', 'VOO']

# 지정된 티커에 해당하는 데이터만 필터링
filtered_data = 전환율_테이블_종가[전환율_테이블_종가['티커종목코드'].isin(desired_tickers)]

# 티커종목코드별로 그래프 그리기
for ticker, group_data in filtered_data.groupby('티커종목코드'):
    fig, ax1 = plt.subplots(figsize=(10, 6))  # 그래프 크기 설정

    # 날짜를 년월일 형식으로 포맷
    group_data['BSE_DT'] = group_data['BSE_DT'].dt.strftime('%Y-%m-%d')

    # 첫 번째 y축: 스케일링된 종목조회건수_증가
    ax1.plot(group_data['BSE_DT'], group_data['종목조회건수_증가_scaled'], label='종목조회건수_증가 (스케일링)', color='blue', marker='o')
    ax1.set_xlabel('날짜 (BSE_DT)')
    ax1.set_ylabel('종목조회건수_증가 (스케일링)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 두 번째 y축: 스케일링된 기준종가_변화
    ax2 = ax1.twinx()  # x축을 공유하는 두 번째 y축 생성
    ax2.plot(group_data['BSE_DT'], group_data['기준종가_변화_scaled'], label='기준종가_변화 (스케일링)', color='orange', marker='x')
    ax2.set_ylabel('기준종가_변화 (스케일링)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    if ticker=='VTI':
        # 그래프 제목 설정
        plt.title(f'티커종목코드: {ticker} - 종목조회건수_증가 & 기준종가_변화 (유사하지 않음)')
    else:
        plt.title(f'티커종목코드: {ticker} - 종목조회건수_증가 & 기준종가_변화 (유사함)')       

    # x축의 날짜 회전 (가독성 향상)
    ax1.set_xticks(range(len(group_data['BSE_DT'])))  # x축에 표시할 위치 설정
    ax1.set_xticklabels(pd.to_datetime(group_data['BSE_DT']).dt.strftime('%m-%d'), rotation=45, fontsize=7)

    # 범례 추가
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 그래프 레이아웃 조정
    plt.tight_layout()

    # 그래프 보여주기
    plt.show()

# 고객분류
베타계수_주식=pd.read_csv('베타계수_주식.csv', encoding='cp949')
표준편차_주식=pd.read_csv('표준편차_주식.csv', encoding='cp949')
투자심리지수_주식=pd.read_csv('투자심리지수_주식.csv', encoding='cp949')
섹터분류_주식=pd.read_csv('섹터분류_주식.csv', encoding='cp949')
화제성_주식=pd.read_csv('화제성_주식.csv', encoding='cp949')

# 각 그룹별 정보
group_info = ['고수 계좌수', '일반 계좌수', '20대이하 계좌수', '30대 계좌수', 
              '40대 계좌수', '50대 계좌수', '60대이상 계좌수', '3000만미만 계좌수', 
              '3000만-1억 계좌수', '1억-10억 계좌수', '10억이상 계좌수']

# 그룹별로 지표의 가중평균을 계산하는 함수
def calculate_weighted_average(df, group_info, beta_col, std_col, trend_col, invest_col, sector_col):
    result = []
    
    for group in group_info:
        # 결과를 저장할 딕셔너리 초기화
        group_result = {'고객분류': group}
        
        # 해당 그룹의 계좌수 기준으로 가중치를 계산
        weights = df[group]

        # 지표에 가중치를 곱하여 가중평균 계산
        weighted_beta = (df[beta_col] * weights).sum() / weights.sum()
        weighted_std = np.sqrt((df[std_col]**2 * weights).sum() / weights.sum())
        weighted_invest = (df[invest_col] * weights).sum() / weights.sum()
        if df.columns.isin([trend_col]).any():
            weighted_trend = (df[trend_col] * weights).sum() / weights.sum()
            group_result['화제성'] = weighted_trend
        
        # 가중평균 값 저장
        group_result['베타계수'] = weighted_beta
        group_result['수익률표준편차'] = weighted_std
        group_result['투자심리지수'] = weighted_invest
        group_result['최대보유섹터'] = df.groupby(sector_col)[group].sum().sort_values(ascending=False).index[0]
    
        # 결과 리스트에 그룹 결과 추가
        result.append(group_result)
    
    return pd.DataFrame(result)

# 데이터프레임을 병합
지수계산 = 주식별고객정보.groupby(['티커종목코드', '주식/ETF구분코드'])[주식별고객정보.columns[주식별고객정보.columns.str.contains('계좌수')]].sum().reset_index()

# 데이터프레임 리스트 생성
dfs = [지수계산, 베타계수_주식, 표준편차_주식, 투자심리지수_주식, 섹터분류_주식, 화제성_주식]

# 공통 키로 병합하는 함수 (on='공통키'로 병합할 열 지정)
지수계산2 = reduce(lambda left, right: pd.merge(left, right, on='티커종목코드'), dfs)

# 베타계수와 수익률표준편차에 대해 가중평균 계산
고객분류 = calculate_weighted_average(지수계산2, group_info, '베타계수', '수익률표준편차', '화제성', '투자심리지수', '섹터분류명')

고객분류[['베타계수', '수익률표준편차', '화제성']].to_csv('고객분류.csv', encoding='cp949', index=False)

print(고객분류)

# 나이대와 금액대 정의
나이대 = ['20대이하 계좌수', '30대 계좌수', '40대 계좌수', '50대 계좌수', '60대이상 계좌수']
금액대 = ['3000만미만 계좌수', '3000만-1억 계좌수', '1억-10억 계좌수', '10억이상 계좌수']

# 나이대와 금액대의 조합 생성
combinations = list(itertools.product(나이대, 금액대))

# 결과 저장을 위한 리스트
results = []

# 각 조합별로 베타계수, 수익률표준편차, 화제성, 섹터분류와 기업구분 비율의 평균 계산
for 나이, 금액 in combinations:
    # 나이대와 금액대에 해당하는 데이터를 필터링하여 평균 계산
    subset = 고객분류[고객분류['고객분류'].isin([나이, 금액])]
    
    beta_avg = subset['베타계수'].mean()
    std_avg = subset['수익률표준편차'].mean()
    popularity_avg = subset['화제성'].mean()
    invest_avg = subset['투자심리지수'].mean()
    
    # 섹터분류 비율 계산
    sector_cols = [col for col in 고객분류.columns if '섹터분류명' in col]
    sector_averages = {col: subset[col].mean() for col in sector_cols}
    
    # 기업구분 비율 계산
    corp_cols = [col for col in 고객분류.columns if '기업구분' in col]
    corp_averages = {col: subset[col].mean() for col in corp_cols}
    
    # 결과 저장
    result = {
        '나이대': 나이,
        '금액대': 금액,
        '베타계수': beta_avg,
        '수익률표준편차': std_avg,
        '화제성': popularity_avg,
        '투자심리지수': invest_avg
    }
    result.update(sector_averages)  # 섹터분류 비율 추가
    result.update(corp_averages)    # 기업구분 비율 추가
    
    results.append(result)

# 최종 결과를 데이터프레임으로 변환
고객분류_20 = pd.DataFrame(results)

고객분류_20[['나이대','금액대', '베타계수', '수익률표준편차', '화제성']].to_csv('고객분류20.csv', index=False, encoding='cp949')

고객분류_20.to_csv('고객분류_특성파악.csv', index=False, encoding='cp949')

# 결과 출력
print(고객분류_20)

고객분류_20 = pd.read_csv('고객분류20.csv', encoding='cp949')

# 투자실력, 나이대, 금액대 결합
고객분류_20['고객그룹'] = 고객분류_20['나이대'] + '_' + 고객분류_20['금액대']

# 클러스터링에 사용할 특징들 선택
features = 고객분류_20[['베타계수', '수익률표준편차', '화제성']]

# 특징들 표준화
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# 엘보우 방법으로 최적의 클러스터 수 찾기
distortions = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(standardized_features)
    distortions.append(kmeans.inertia_)

# 엘보우 곡선 그리기
print(distortions)
plt.plot(range(1, 20), distortions, marker='o')
plt.xlabel('클러스터 수')
plt.ylabel('왜곡(Distortion)')
plt.title('엘보우 방법')
plt.show()

# 선택한 클러스터 수로 K-means 적용 (n_clusters에 최적 값을 넣기)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(standardized_features)

# 원본 데이터프레임에 클러스터 레이블 추가
고객분류_20['군집'] = kmeans.labels_

# 3D 시각화
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 군집의 범위
clusters = np.unique(고객분류_20['군집'])

# 각 군집에 대해 별도로 scatter 플롯을 그리며 범례 추가
for cluster in clusters:
    clustered_data = 고객분류_20[고객분류_20['군집'] == cluster]
    ax.scatter(clustered_data['베타계수'], clustered_data['수익률표준편차'], clustered_data['화제성'],
               label=f'군집 {cluster}', s=50)

# 축 레이블 및 그래프 타이틀 설정
ax.set_xlabel('베타계수', labelpad=10)
ax.set_ylabel('수익률표준편차', labelpad=10)
ax.text2D(0.95, 0.80, "화제성", transform=ax.transAxes, fontsize=10)
plt.title('3D Clustering Visualization', fontsize=16, fontweight='bold')

# 범례 추가 (작게 만들고 적당히 위로 이동)
plt.legend(title="군집", loc="upper right", fontsize=8, title_fontsize=10, bbox_to_anchor=(1.05, 1.1), borderaxespad=0.)

# 그래프 보여주기
plt.show()

고객분류_20.to_csv('고객분류_완료.csv', index=False, encoding='cp949')

고객분류20 = 고객분류_20.copy()

print(고객분류_20.sort_values(by='군집').reset_index(drop=True))

print(고객분류20.groupby('군집')[['베타계수', '수익률표준편차', '화제성']].mean())

# 큐레이션지표 정리
ETF베타계수 = pd.read_csv('베타계수_ETF.csv',encoding='cp949')
ETF표준편차 = pd.read_csv('표준편차_ETF.csv',encoding='cp949')
ETF투자심리지수 = pd.read_csv('투자심리지수_ETF.csv',encoding='cp949')
ETF섹터분류 = pd.read_csv('섹터분류_ETF.csv',encoding='cp949')
ETF화제성 = pd.read_csv('화제성_ETF.csv', encoding='cp949')
종가_조회수_상관관계 = pd.read_csv('종가_조회수_상관관계.csv', encoding='cp949')

# 세 개의 지표를 하나의 데이터 프레임으로 변환
ETF큐레이션지표 = reduce(lambda left, right: pd.merge(left, right, on='티커종목코드', how='inner'), 
                     [ETF베타계수, ETF표준편차, ETF투자심리지수, ETF섹터분류])

# StandardScaler 객체 생성
scaler = StandardScaler()

# 표준화할 열들 선택 (DataFrame에서 숫자형 열만 선택하는 예시)
columns_to_standardize = ['베타계수', '수익률표준편차', '투자심리지수']

# scaler를 적용해 데이터 표준화
ETF큐레이션지표[columns_to_standardize] = scaler.fit_transform(ETF큐레이션지표[columns_to_standardize])

## 트렌드지수생성 ( 화제성 + 종가-검색 변화율 )
# StandardScaler 적용
ETF화제성['화제성_표준화'] = scaler.fit_transform(ETF화제성[['화제성']])
종가_조회수_상관관계['상관계수_표준화'] = scaler.fit_transform(abs(종가_조회수_상관관계[['상관계수']]))

# '티커종목코드', '화제성_표준화', '종가_조회수_상관관계' 열을 가진 '트렌드지수테이블' 생성
트렌드지수테이블 = ETF화제성[['티커종목코드', '화제성_표준화']]
트렌드지수테이블 = pd.merge(트렌드지수테이블, 종가_조회수_상관관계[['티커종목코드', '상관계수_표준화']], on='티커종목코드')

# '화제성_표준화'와 '상관계수_표준화'를 더한 후, 다시 표준화하여 '트렌드지수' 생성
트렌드지수테이블['트렌드_합'] = 트렌드지수테이블['화제성_표준화'] + 트렌드지수테이블['상관계수_표준화']
트렌드지수테이블['트렌드지수'] = scaler.fit_transform(트렌드지수테이블[['트렌드_합']])

# '티커종목코드'와 '트렌드지수' 열만 선택하여 새로운 데이터프레임으로 저장
트렌드지수테이블 = 트렌드지수테이블[['티커종목코드', '트렌드지수']]

큐레이션지표 = pd.merge(트렌드지수테이블, ETF큐레이션지표, on='티커종목코드')
큐레이션지표

큐레이션지표.to_csv('큐레이션지표.csv',encoding='cp949',index=False) # 최종 큐레이션지표 표준화 수치 저장

# 그룹뷴류
# 그룹 정의
groups = {
    '0번그룹': {
        '나이대': ['20대이하 계좌수', '30대 계좌수'],
        '자산군': ['3000만-1억 계좌수' , '1억-10억 계좌수', '10억이상 계좌수']
    },
    '1번그룹': {
        '나이대': ['40대 계좌수', '60대이상 계좌수'],
        '자산군': ['3000만미만 계좌수'],
        '추가_나이대': ['50대 계좌수'],
        '추가_자산군': ['3000만미만 계좌수', '3000만-1억 계좌수', '1억-10억 계좌수', '10억이상 계좌수']
    },
    '2번그룹': {
        '나이대': ['20대이하 계좌수', '30대 계좌수'],
        '자산군': ['3000만미만 계좌수'],
        '추가_나이대': ['40대 계좌수'],
        '추가_자산군': ['3000만-1억 계좌수', '1억-10억 계좌수', '10억이상 계좌수']
    },
    '3번그룹': {
        '나이대': ['60대이상 계좌수'],
        '자산군': ['3000만-1억 계좌수' , '1억-10억 계좌수', '10억이상 계좌수']
    }
}

# 그룹별 계좌수 계산 함수
def calculate_group_accounts(df, groups):
    result = pd.DataFrame()  # 결과를 담을 데이터프레임 생성
    if (df.columns.isin(['기준일자']).any())&(df.columns.isin(['종목조회건수']).any()):
        result['기준일자'] = df['기준일자']
        result['종목조회건수'] = df['종목조회건수']

    result['티커종목코드'] = df['티커종목코드']  # 티커종목코드는 유지
    for group_name, criteria in groups.items():
        # 나이대 비율 더하기
        age_ratio = df[criteria['나이대']].sum(axis=1) / df['총보유계좌수']
        
        # 자산군 비율 더하기
        asset_ratio = df[criteria['자산군']].sum(axis=1) / df['총보유계좌수']
        
        # 추가 나이대와 자산군이 있을 경우, 그 비율을 추가
        if '추가_나이대' in criteria and '추가_자산군' in criteria:
            age_ratio += df[criteria['추가_나이대']].sum(axis=1) / df['총보유계좌수']
            asset_ratio += df[criteria['추가_자산군']].sum(axis=1) / df['총보유계좌수']
        
        # 최종 비율 계산 (나이대 * 자산군)
        final_ratio = age_ratio * asset_ratio
        
        # 그룹별 계좌수 추가
        result[group_name] = df['총보유계좌수'] * final_ratio
    
    return result

# 그룹별 계좌수 계산
그룹별주식비율=calculate_group_accounts(지수계산, groups)
그룹별주식비율.to_csv('그룹별주식비율.csv', index=False, encoding='cp949')
print(그룹별주식비율)

# 지표 계산 함수
def calculate_group_metrics(group_names, portfolio_df, group_info):
    # 주식별 지표와 포트폴리오 데이터를 병합
    merged_df = pd.merge(주식별지표, portfolio_df, on='티커종목코드')
    result = []

    for group_name ,group in zip(group_names, group_info):
        # 결과를 저장할 딕셔너리 초기화
        group_result = {'고객분류': group}
        
        # 해당 그룹의 계좌수 기준으로 가중치를 계산
        weights = merged_df[group]

        # 가중치 합 계산
        total_weight = weights.sum()

        # 결과를 저장할 딕셔너리 초기화
        group_result = {'그룹명': group_name}

        # 지표에 가중치를 곱하여 가중평균 계산
        weighted_beta = (merged_df['베타계수'] * weights).sum() / total_weight
        weighted_std = np.sqrt((merged_df['수익률표준편차']**2 * weights).sum() / total_weight)
        weighted_invest = (merged_df['투자심리지수'] * weights).sum() / total_weight
        weighted_trend = (merged_df['화제성'] * weights).sum() / total_weight
        weighted_sector = merged_df.groupby('섹터분류명')[group].sum().sort_values(ascending=False).index[0]
        
        # 가중평균 값 저장
        group_result['베타계수'] = weighted_beta
        group_result['수익률표준편차'] = weighted_std
        group_result['트렌드지수'] = weighted_trend
        group_result['투자심리지수'] = weighted_invest
        group_result['최대보유섹터'] = weighted_sector

        result.append(group_result)

    return pd.DataFrame(result)

group_metrics=calculate_group_metrics(['부기', '랑이', '아웅이', '숭이'], 그룹별주식비율, ['0번그룹', '1번그룹', '2번그룹', '3번그룹'])

group_metrics.to_csv('그룹별지표.csv', encoding='cp949', index=False)

print(group_metrics)

# 생성형AI
주식별지표=pd.read_csv('주식별지표.csv', encoding='cp949')
큐레이션지표=pd.read_csv('큐레이션지표.csv', encoding='cp949')
큐레이션지표.columns=['티커종목코드', '트렌드지수', '베타계수', '수익률표준편차', '투자심리지수', '섹터']
그룹별주식비율=pd.read_csv('그룹별주식비율.csv', encoding='cp949')

# 고객 DB 생성 및 고객 정보 삽입
# 각 그룹별 상위 10개 티커를 선택하고 비중을 계산하는 함수
def calculate_top_10_by_group(group_df, group_column):
    # 상위 10개 티커 선택
    top_10_df = group_df[['티커종목코드', group_column]].nlargest(10, group_column)
    
    # 비중 계산 (상위 10개 티커의 개수 합을 기준으로 비중 계산)
    total_shares = top_10_df[group_column].sum()
    top_10_df['비중'] = top_10_df[group_column] / total_shares
    
    return top_10_df[['티커종목코드', '비중']]

# 그룹별 상위 10개 티커와 비중 계산
group_columns = ['0번그룹', '1번그룹', '2번그룹', '3번그룹']
top_10_results = {}

for group_column in group_columns:
    top_10_results[group_column[:-2]+'고객'] = calculate_top_10_by_group(그룹별주식비율, group_column).reset_index(drop=True)

# 아래는 DB를 생성하는 코드입니다. 파일 내에 DB가 존재할 경우 오류가 발생할 수 있어, 주석처리했습니다. 또한, sqlite3와 azure DB를 둘다 사용했기 때문에 충돌이 일어날 수 있습니다.

# # SQLite 데이터베이스 파일 경로
# db_file = 'customer_portfolio.db'

# # 데이터베이스 파일이 이미 존재하는지 확인
# if os.path.exists(db_file):
#     try:
#         # 파일 삭제 시도
#         os.remove(db_file)
#         print(f"기존 데이터베이스 파일 '{db_file}'을 삭제했습니다.")
#     except Exception as e:
#         print(f"데이터베이스 파일을 삭제할 수 없습니다: {e}")
#         exit()  # 삭제에 실패하면 프로그램을 종료
# else:
#     print(f"데이터베이스 파일이 존재하지 않습니다. '{db_file}'을 새로 생성합니다.")

# # SQLite 데이터베이스 연결
# conn = sqlite3.connect(db_file)
# cursor = conn.cursor()

# # 1단계: 고객 정보 테이블 생성 (고객DB)
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS 고객DB (
#     고객ID TEXT PRIMARY KEY,
#     비밀번호 TEXT,
#     이름 TEXT,
#     나이 INTEGER,
#     보유자금 INTEGER,
#     투자실력 TEXT
# )
# ''')

# customers = [
#     ('0번고객', '0000', '김철수', 31, 694140000, '고수'),
#     ('1번고객', '1111', '이영희', 54, 443650000, '일반'),
#     ('2번고객', '2222', '박민수', 25, 29830000, '일반'),
#     ('3번고객', '3333', '최영식', 64, 272410000, '고수')
# ]

# cursor.executemany('''
# INSERT INTO 고객DB (고객ID, 비밀번호, 이름, 나이, 보유자금, 투자실력)
# VALUES (?, ?, ?, ?, ?, ?)
# ''', customers)

# conn.commit()

# # 2단계: 주식 포트폴리오 테이블 생성 (포트폴리오DB)
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS 포트폴리오DB (
#     고객ID TEXT,
#     티커종목코드 TEXT,
#     보유수량 REAL,
#     비중 REAL,
#     PRIMARY KEY (고객ID, 티커종목코드),
#     FOREIGN KEY (고객ID) REFERENCES 고객DB(고객ID)
# )
# ''')

# # 3단계: 고객 지표 테이블 생성 (고객의 베타계수, 수익률표준편차 등 지표 정보 저장)
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS 고객지표DB (
#     지표ID INTEGER PRIMARY KEY AUTOINCREMENT,
#     고객ID TEXT,
#     베타계수 REAL,
#     수익률표준편차 REAL,
#     트렌드지수 REAL,
#     투자심리지수 REAL,
#     최대보유섹터 TEXT,
#     FOREIGN KEY (고객ID) REFERENCES 고객DB(고객ID)
# )
# ''')

# conn.commit()


# # 4단계: 보유 자금을 고려하여 보유 주식 수량 계산 함수
# def calculate_shares(portfolio_dict, conn):
#     cursor = conn.cursor()
    
#     stock_prices = {}
#     portfolio_with_shares = []

#     for group_name, df in portfolio_dict.items():
#         for index, row in df.iterrows():
#             ticker = row['티커종목코드']
            
#             # 고객의 보유자금 가져오기
#             cursor.execute("SELECT 보유자금 FROM 고객DB WHERE 고객ID = ?", (group_name[:-2]+'고객',))
#             customer_funds = cursor.fetchone()[0]

#             # yfinance를 사용하여 최근 종가를 가져옴
#             stock = yf.Ticker(ticker)
#             latest_price = stock.history(period='1d')['Close'].iloc[-1]
            
#             # 보유자금과 비중을 고려하여 보유수량 계산
#             allocated_funds = customer_funds * row['비중']  # 보유 자금의 비중에 따라 배분된 금액
#             shares = allocated_funds / latest_price  # 보유수량 계산
            
#             portfolio_with_shares.append((group_name[:-2]+'고객', ticker, shares, row['비중']))

#     return portfolio_with_shares

# # 보유 주식 수량 계산
# calculated_portfolio = calculate_shares(top_10_results, conn)

# # 5단계: 포트폴리오DB에 삽입 또는 업데이트
# for group, ticker, shares, ratio in calculated_portfolio:
#     cursor.execute('''
#     INSERT OR REPLACE INTO 포트폴리오DB (고객ID, 티커종목코드, 보유수량, 비중)
#     VALUES (?, ?, ?, ?)
#     ''', (group, ticker, shares, ratio))

# conn.commit()

# # 지표 계산 함수
# def calculate_customer_metrics(customer_id, portfolio_df):
#     # 주식별 지표와 포트폴리오 데이터를 병합
#     merged_df = pd.merge(주식별지표, portfolio_df, on='티커종목코드')

#     # 결과를 저장할 딕셔너리 초기화
#     group_result = {'고객ID': customer_id}
    
#     # 해당 그룹의 비중 기준으로 가중치를 계산
#     weights = merged_df['비중']

#     # 지표에 가중치를 곱하여 가중평균 계산
#     weighted_beta = (merged_df['베타계수'] * weights).sum()
#     weighted_std = np.sqrt((merged_df['수익률표준편차']**2 * weights).sum())
#     weighted_invest = (merged_df['투자심리지수'] * weights).sum()
#     weighted_trend = (merged_df['화제성'] * weights).sum()
#     weighted_sector = merged_df.groupby('섹터분류명')['비중'].sum().sort_values(ascending=False).index[0]
    
#     # 가중평균 값 저장
#     group_result['베타계수'] = weighted_beta
#     group_result['수익률표준편차'] = weighted_std
#     group_result['화제성'] = weighted_trend
#     group_result['투자심리지수'] = weighted_invest
#     group_result['최대보유섹터'] = weighted_sector

#     return group_result

# # 지표 저장 함수 (고객지표DB에 저장)
# def save_customer_metrics_to_db(customer_metrics):
#     cursor.execute('''
#     INSERT INTO 고객지표DB (고객ID, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수, 최대보유섹터)
#     VALUES (?, ?, ?, ?, ?, ?)
#     ''', (
#         customer_metrics['고객ID'],
#         customer_metrics['베타계수'],
#         customer_metrics['수익률표준편차'],
#         customer_metrics['화제성'],
#         customer_metrics['투자심리지수'],
#         customer_metrics['최대보유섹터']
#     ))
#     conn.commit()

# for customer_id, value in top_10_results.items():
#     # 지표 계산 및 저장 실행
#     customer_metrics = calculate_customer_metrics(customer_id, value)

#     # DB에 지표 저장
#     save_customer_metrics_to_db(customer_metrics)

# # 6단계: 고객 정보, 주식 포트폴리오, 고객 지표를 JOIN하여 조회
# query = '''
# SELECT 고객DB.고객ID, 고객DB.이름, 고객DB.나이, 포트폴리오DB.티커종목코드, 포트폴리오DB.보유수량, 포트폴리오DB.비중,
#        고객지표DB.베타계수, 고객지표DB.수익률표준편차, 고객지표DB.트렌드지수, 고객지표DB.투자심리지수, 고객지표DB.최대보유섹터
# FROM 고객DB
# JOIN 포트폴리오DB ON 고객DB.고객ID = 포트폴리오DB.고객ID
# JOIN 고객지표DB ON 고객DB.고객ID = 고객지표DB.고객ID
# '''

# df_portfolio = pd.read_sql_query(query, conn)
# print(df_portfolio)

# # DB 연결 해제
# conn.close()

db_file = 'customer_portfolio.db'
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

query = '''
SELECT 고객DB.고객ID, 고객DB.이름, 고객DB.나이, 포트폴리오DB.티커종목코드, 포트폴리오DB.보유수량, 포트폴리오DB.비중,
       고객지표DB.베타계수, 고객지표DB.수익률표준편차, 고객지표DB.트렌드지수, 고객지표DB.투자심리지수, 고객지표DB.최대보유섹터
FROM 고객DB
JOIN 포트폴리오DB ON 고객DB.고객ID = 포트폴리오DB.고객ID
JOIN 고객지표DB ON 고객DB.고객ID = 고객지표DB.고객ID
'''

df_portfolio = pd.read_sql_query(query, conn)


# RAG

ETF베타계수=pd.read_csv('베타계수_ETF.csv', encoding='cp949')

documents = [
'''
화제성에 대한 설명 :

화제성 지수는 주식에서만 사용되는 지표입니다. 화제성 지수는 특정 종목의 거래량이 다른 종목과 비교했을 때 상대적으로 얼마나 두드러지게 변화하는지를 나타내는 지표입니다. 고객분류에서 화제성 지수를 활용하면, 다른 종목에 비해 특정 종목이 얼마나 주목받고 있는지를 평가하여 트렌드에 민감한 고객을 구분할 수 있습니다. 높은 지표 값은 고객이 주목도가 높은 주식을 선호하는 경향을 보여주고, 낮은 값은 주식에 대한 관심이 적은 고객으로 분류됩니다. ETF 추천에서는 이 지표를 통해 종목이 상대적으로 얼마나 주목받고 있는지를 파악할 수 있습니다.
''',
'''
베타계수에 대한 설명 :

베타계수는 특정 종목의 변동성이 시장 전체와 얼마나 연관되어 있는지를 측정하는 지표입니다. 고객분류에서는 베타계수를 이용해 시장보다 변동성이 큰 주식을 선호하는 성향의 고객과, 안정적인 주식을 선호하는 고객을 구분할 수 있습니다. 베타계수가 상대적으로 높으면 더 큰 변동성을 기대하는 고객, 낮으면 안정성을 중시하는 고객으로 분류됩니다. ETF 추천에서는 베타계수는 시장 전체와 비교해 해당 종목이 상대적으로 위험하거나 안정적인지를 보여줍니다. 값이 크면 시장보다 더 큰 변동성을 보이고, 값이 작으면 상대적으로 안정적인 특성을 가집니다.
''',
'''
수익률 표준편차에 대한 설명 :

수익률 표준편차는 종목의 가격 변동성을 측정하는 지표로, 고객분류에서 변동성이 큰 종목을 선호하는 고객은 높은 수익률 표준편차를 가진 종목을 더 많이 선택하는 경향이 있습니다. 안정적인 주식을 선호하는 고객은 표준편차가 낮은 종목을 선택합니다. ETF 추천에서는 수익률 표준편차가 높은 ETF는 변동성이 큰 종목들로 구성되어 있어, 위험을 감수하며 높은 수익을 기대하는 투자자에게 적합한 ETF를 추천하는 데 유용합니다.
''',
'''
투자심리지수에 대한 설명 :

투자심리지수는 사람들이 특정 주식이나 ETF에 대해 얼마나 긍정적인 감정을 가지고 있는지를 나타내는 지표입니다. 고객분류에서 투자심리지수가 높은 고객은 긍정적인 감정을 가진 종목을 더 많이 매수하며, 시장의 인기 종목에 집중하는 경향을 보입니다. ETF 추천에서는 투자심리지수가 높은 ETF가 긍정적인 평가를 받고 있는지 확인하며, 많은 투자자들이 긍정적으로 평가하는 ETF를 추천하는 데 도움이 됩니다.
''',

'''
트렌드지수에 대한 설명 :

트렌드 지수는 주식 지표인 '화제성'과 '종목 조회 건수 와 종가 변화율'에 대한 지표를 합산한 지표입니다. 트렌드 지수는 특정 종목의 거래량이 다른 종목과 비교했을 때 상대적으로 얼마나 두드러지게 변화하는지를 나타내는 지표입니다. 높은 트렌드 지수는 해당 종목이 시장에서 큰 관심을 받고 있는 것을 나타내며, 낮은 값은 주목받지 않는다는 것을 의미합니다. 이를 통해 투자자는 트렌드에 민감한 종목을 식별하고 포트폴리오를 조정할 수 있습니다.

''',

'''
부기에 대한 설명 :

안정 기회 추구형 부기 (닌자거북이 스타일)

안정적인 시장에서 트렌드를 분석하여 위험을 최소화하고 꾸준한 수익을 목표로 안전한 종목에 투자하는 투자자 그룹

설명: 닌자거북이 스타일의 부기들은 겉보기엔 느긋하고 신중해 보이지만, 내면적으로는 민첩하고 날카로운 판단력을 지닌 투자자들입니다. 거북이처럼 단단하고 안정적인 전략을 바탕으로 투자하면서도, 트렌드의 변화에 재빠르게 대응하는 특성이 있습니다. 시장이 안정적인 상황에서 일관된 수익을 위해 신중하게 움직이며, 필요할 때는 은밀하고 조용히 기회를 잡아냅니다.

"토끼를 이긴 거북이"처럼, 급하지 않고 차분하게 장기적인 성과를 쌓아가는 성향이 있으며, 안정적이지만 중요한 기회는 놓치지 않는 스타일입니다. 닌자처럼 조용히 움직이며 꾸준히 수익을 지켜가는 모습이 특징입니다.
''',
'''
랑이에 대한 설명 :

리스크 수익 탐색형 랑이 (호랑이 스타일)

변동성이 큰 시장에서 트렌드에 구애받지 않고 고위험 종목에 과감히 투자해 높은 수익을 추구하는 투자자 그룹

설명: 호랑이처럼 리스크 수익 탐색형 랑이들은 자신만의 길을 걸으며, 변동성이 큰 시장에서도 남의 시선에 구애받지 않고 과감히 도전하는 투자자들입니다. 다른 이들의 판단이나 트렌드를 따르지 않고, 강한 자신감과 직감을 바탕으로 독립적인 결정을 내립니다.

호랑이가 주변의 시선을 신경 쓰지 않고 사냥감을 쫓듯이, 이들은 높은 수익 가능성을 놓치지 않고 위험을 감수하며 투자합니다. 때로는 대담하고 때로는 위협적으로 보일 만큼의 강렬한 성향을 가지고 있어, 남다른 길을 선택해도 흔들리지 않는 결단력을 보여줍니다.
''',
'''
아웅이에 대한 설명 :

위기 속 안정형 아웅이 (사막여우 스타일)

불안정한 시장에서 트렌드를 분석하여 위험을 줄이고, 비교적 안정적인 종목에 투자하는 투자자 그룹

설명: 사막여우처럼 위기 속에서도 예리하게 트렌드를 분석하며 안정적인 선택을 하는 아웅이들은 불확실성이 높은 시장에서 자신의 생존 스킬을 발휘하는 투자자들입니다. 사막과 같은 열악한 환경에서도 생존할 수 있는 사막여우의 특징을 닮아, 위기 상황에서도 위험을 줄이고 안전하게 자산을 지켜가는 성향이 강합니다.

상황 변화에 민감하며, 항상 위험을 최소화하는 방향으로 판단을 내립니다. 사막의 예리한 관찰자처럼, 신속하고 정확한 판단으로 위험 요소를 피하고 안전한 선택을 하는데 능숙합니다.
''',
'''
숭이에 대한 설명 :

성장 기회 탐색형 숭이 (원숭이 스타일)

안정적인 시장에서 트렌드를 반영하며 도전적인 종목에 투자해 높은 성장을 목표로 하는 투자자 그룹

설명: 장난기 많고 호기심이 왕성한 원숭이 스타일의 숭이들은 안정적인 환경 속에서도 끊임없이 새로운 기회를 탐색하고 도전하는 투자자들입니다. 안정된 상황에서 다양한 트렌드를 살펴보며, 성장 가능성이 있는 종목을 발견하면 적극적으로 투자해 더 높은 성장을 추구합니다.

원숭이처럼 창의적이고 에너지가 넘치는 스타일로, 안정적인 상황에서도 끊임없이 움직이며 새롭고 흥미로운 종목을 찾는 데에 열정적입니다. 놀이와 탐구를 즐기듯, 기회를 놓치지 않으려는 열정과 활발한 성향이 강합니다.
'''
]


# ETF 티커별로 상위 5개의 보유 종목을 추출하는 함수
for etf_ticker, group in ETF구성종목정보[ETF구성종목정보['대상 ETF 티커'].isin(ETF베타계수['티커종목코드'])].groupby('대상 ETF 티커'):
    # 보유 비중이 높은 순으로 상위 5개 추출
    top_5_stocks = group.nlargest(5, '보유 종목의 비중')

    # 텍스트로 변환
    text = f"ETF {etf_ticker}의 구성 비중 상위 5개 종목:\n"
    for _, row in top_5_stocks.iterrows():
        stock_name_korean = row['fc_sec_eng_nm']
        stock_weight = row['보유 종목의 비중']
        text += f"- {stock_name_korean}: {stock_weight}%\n"

    # 결과 저장
    documents.append(text)

# ETF 티커별로 정보를 요약하는 함수
for etf_ticker, group in ETF점수정보[ETF점수정보['etf_iem_cd'].isin(ETF베타계수['티커종목코드'])].groupby('etf_iem_cd'):
    latest_entry = group.loc[:,~group.columns.isin(['etf_iem_cd'])].mean()
    
    # 필요한 정보 추출
    etf_score = latest_entry['ETF점수']
    z_score_rank = latest_entry['Z점수순위']
    one_month_return = latest_entry['1개월총수익율']
    three_month_return = latest_entry['3개월총수익율']
    one_year_return = latest_entry['1년총수익율']
    
    # Z점수 및 기타 지표 요약
    cumulative_return_z = latest_entry['누적수익율Z점수']
    info_ratio_z = latest_entry['정보비율Z점수']
    sharpe_ratio_z = latest_entry['샤프지수Z점수']
    correlation_z = latest_entry['상관관계Z점수']
    tracking_error_z = latest_entry['트래킹에러Z점수']
    max_drawdown_z = latest_entry['최대낙폭Z점수']
    volatility_z = latest_entry['변동성Z점수']
    
    # 텍스트 포맷팅
    text = f"ETF {etf_ticker} 요약:\n"
    text += f"- 1개월 총수익률: {one_month_return}%\n"
    text += f"- 3개월 총수익률: {three_month_return}%\n"
    text += f"- 1년 총수익률: {one_year_return}%\n"
    text += f"- ETF 점수: {etf_score}\n"
    text += f"- Z점수 순위: {z_score_rank}\n"
    text += f"- 누적 수익률 Z점수: {cumulative_return_z}\n"
    text += f"- 정보비율 Z점수: {info_ratio_z}\n"
    text += f"- 샤프지수 Z점수: {sharpe_ratio_z}\n"
    text += f"- 상관관계 Z점수: {correlation_z}\n"
    text += f"- 트래킹 에러 Z점수: {tracking_error_z}\n"
    text += f"- 최대 낙폭 Z점수: {max_drawdown_z}\n"
    text += f"- 변동성 Z점수: {volatility_z}\n"
    
    # 리스트에 추가
    documents.append(text)


# ETF 티커별로 배당 정보를 요약하는 함수
for etf_ticker, group in ETF배당내역[ETF배당내역['대상 ETF 티커'].isin(ETF베타계수['티커종목코드'])].groupby('대상 ETF 티커'):
    latest_entry = group[['배당금', '수정 배당금']].mean()
    latest_entry['배당 주기'] = group['배당 주기'].mode()[0]

    # 필요한 배당 정보 추출
    dividend = latest_entry['배당금']
    adj_dividend = latest_entry['수정 배당금']
    dividend_frequency = latest_entry['배당 주기']

    # 텍스트 포맷팅
    text = f"ETF {etf_ticker} 배당 요약:\n"
    text += f"- 배당금: {dividend}\n"
    text += f"- 수정 배당금: {adj_dividend}\n"
    text += f"- 배당 주기: {dividend_frequency}\n"

    # 리스트에 추가
    documents.append(text)

# JSON 파일에 저장 (ensure_ascii=False를 사용하여 한글을 그대로 저장)
with open('documents.json', 'w', encoding='utf-8') as f:
    json.dump(documents, f, ensure_ascii=False, indent=4)

print("ETF 데이터 요약 완료 및 JSON 저장")

# 문서 분할 함수 (문서를 단락 단위로 나눔)
def split_document(document, chunk_size=100):
    sentences = document.split(". ")  # 문장을 기준으로 분할
    chunks = []
    chunk = []
    for sentence in sentences:
        chunk.append(sentence)
        if len(chunk) >= chunk_size:
            chunks.append(". ".join(chunk))
            chunk = []
    if chunk:  # 남은 문장들을 마지막 chunk로 추가
        chunks.append(". ".join(chunk))
    return chunks

# 임베딩 모델 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 문서를 분할하고, 임베딩 계산
chunks = []
for doc in documents:
    chunks.extend(split_document(doc))

# 문서 임베딩 계산
document_embeddings = model.encode(chunks)

# FAISS 인덱스 생성 (코사인 유사도를 사용하기 위해 L2 정규화)
document_embeddings = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
embedding_dimension = document_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dimension)  # 내적 기반 검색
index.add(np.array(document_embeddings))

# 모델 디렉토리 설정
model_save_path = "saved_model"
os.makedirs(model_save_path, exist_ok=True)
# 아래의 코드는 이미 모델 파일이 있을 경우 에러가 발생하여 주석처리했습니다.
# model.save(model_save_path)

# FAISS 인덱스 저장
faiss_index_path = "faiss_index.bin"
faiss.write_index(index, faiss_index_path)

print(f"모델은 {model_save_path}에 저장되었습니다.")
print(f"FAISS 인덱스는 {faiss_index_path}에 저장되었습니다.")

# SQLite 데이터베이스 연결
conn = sqlite3.connect('customer_portfolio.db')

# 모든 고객 지표 불러오기 함수 (최신 고객 지표만 불러오기)
def fetch_all_latest_customer_metrics():
    query = '''
    SELECT 고객ID, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수
    FROM 고객지표DB
    WHERE 지표ID IN (
        SELECT MAX(지표ID) FROM 고객지표DB GROUP BY 고객ID
    )
    '''
    customer_metrics_df = pd.read_sql_query(query, conn)
    return customer_metrics_df

# 고객 지표와 그룹 지표를 비교하여 가장 유사한 그룹 찾기
def match_customers_to_group(customer_metrics_df, group_metrics_df):
    metrics_columns = ['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']

    # 고객 지표 표준화
    scaler_customer = StandardScaler()
    scaled_customer_metrics = scaler_customer.fit_transform(customer_metrics_df[metrics_columns])

    # 그룹 지표 표준화
    scaler_group = StandardScaler()
    scaled_group_metrics = scaler_group.fit_transform(group_metrics_df[metrics_columns])

    # 코사인 유사도를 계산하고 각 고객에 대해 가장 유사한 그룹 찾기
    results = []
    for idx, customer_vector in enumerate(scaled_customer_metrics):
        customer_vector = customer_vector.reshape(1, -1)  # 고객 벡터를 2차원으로 변환
        similarities = cosine_similarity(customer_vector, scaled_group_metrics)  # 유사도 계산
        most_similar_group_idx = np.argmax(similarities)  # 가장 유사한 그룹의 인덱스 찾기
        most_similar_group = group_metrics_df.iloc[most_similar_group_idx]
        similarity_score = similarities[0][most_similar_group_idx]
        results.append({
            '고객ID': customer_metrics_df.iloc[idx]['고객ID'],
            '가장 유사한 그룹': most_similar_group['그룹명'],
            '유사도 점수': similarity_score
        })
    
    return pd.DataFrame(results)

# 1. 모든 고객의 최신 지표 불러오기
customer_metrics_df = fetch_all_latest_customer_metrics()

# 2. 각 고객과 그룹을 비교하여 가장 유사한 그룹 찾기
similarity_results_df = match_customers_to_group(customer_metrics_df, group_metrics)

# 3. 결과 출력
print(similarity_results_df)

# DB 연결 해제
conn.close()

# 고객별 지표
print(df_portfolio.groupby('고객ID')[['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']].mean())

# 어투변환

# .tsv 파일 불러오기
file_path = 'smilestyle_dataset.tsv'
어투변환_로우데이터 = pd.read_csv(file_path, sep='\t')  # 탭 구분 파일이므로 sep='\t'로 지정

# 'enfp' , 'seonbi',  'emoticon' , 'halbae', 'azae' 열만 선택하는 코드
어투변환_데이터 = 어투변환_로우데이터[['king' , 'gentle',  'enfp' , 'azae']]

# 열 이름 변경
어투변환_데이터 = 어투변환_데이터.rename(columns={
    'king': '왕',
    'gentle': '젠틀맨',
    'enfp': '어린이',
    'azae': '아저씨'
})

# 선택한 열 출력
어투변환_데이터.head()

# utf-8로 저장
어투변환_데이터.to_csv('어투변환_데이터.csv', index=False, encoding='utf-8')

ment=pd.read_csv('어투변환_데이터.csv', encoding='utf-8')

# 생성형 AI

# OpenAI API 키 설정 - Open API키를 입력해주세요.
API_KEY = 'API KEY를 입력하세요'

# SQLite DB에 연결
conn = sqlite3.connect('customer_portfolio.db')
cursor = conn.cursor()

# 1. 종가 불러오기 함수
def get_stock_prices(tickers):
    stock_prices = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        stock_prices[ticker] = stock.history(period="1d")['Close'].iloc[-1]  # 가장 최근 종가
    return stock_prices

# 어투변환을 사용하시려면 아래 2번함수 주석을 제거해주세요. openai 버전은 1.51.1이어야 합니다.
# 2. GPT
# def get_gpt_response(prompt_type, user_input):
#     if prompt_type == "포트폴리오 입력":
#         prompt = f"""
#         사용자가 다음과 같은 포트폴리오를 입력했습니다: '{user_input}'.
#         종목명을 티커종목코드로 변환해주고, 티커종목코드와 종목수량을 반환해주세요.
#         반환 예시: AAPL 10, TSLA 5, NVDA 3    
#         """

#     elif prompt_type == "포트폴리오 업데이트":
#         prompt = f"""
#         사용자가 다음과 같은 포트폴리오를 입력했습니다: '{user_input}'.
#         종목명을 티커종목코드로 변환해주고, 매수/매도를 판단하여 티커종목코드와 변화량을 +, -로 반환해주세요.
#         반환 예시: AAPL 10, TSLA -5, NVDA 3
#         """

#     client = OpenAI(
#         api_key=API_KEY,
#     )

#     response = client.chat.completions.create(
#         messages=[
#             {"role": "system", "content": "You are a helpful financial assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         model="gpt-3.5-turbo"
#     )
#     return response.choices[0].message.content.strip()

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
    cursor.execute("SELECT 비밀번호 FROM 고객DB WHERE 고객ID = ?", (customer_id,))
    result = cursor.fetchone()
    if result and result[0] == password:
        return True
    return False

# 7. 포트폴리오 저장 함수 (DB에 저장)
def save_portfolio_to_db(customer_id, portfolio_df):
    for _, row in portfolio_df.iterrows():
        cursor.execute('''
        INSERT INTO 포트폴리오DB (고객ID, 티커종목코드, 보유수량, 비중)
        VALUES (?, ?, ?, ?)
        ''', (customer_id, row['티커종목코드'], row['보유수량'], row['비중']))
    conn.commit()

# 8. 포트폴리오 삭제 후 새로 입력 (DB에서 기존 포트폴리오 삭제)
def delete_existing_portfolio(customer_id):
    cursor.execute("DELETE FROM 포트폴리오DB WHERE 고객ID = ?", (customer_id,))
    conn.commit()


# 9. 기존 포트폴리오에서 종목 업데이트 함수
def update_existing_portfolio(customer_id, parsed_changes):
    # 기존 포트폴리오 조회
    existing_portfolio_query = "SELECT 티커종목코드, 보유수량 FROM 포트폴리오DB WHERE 고객ID = ?"
    existing_portfolio_df = pd.read_sql(existing_portfolio_query, conn, params=(customer_id,))

    # 데이터프레임을 딕셔너리로 변환 (주식명 -> 보유수량)
    existing_portfolio = dict(zip(existing_portfolio_df['티커종목코드'], existing_portfolio_df['보유수량']))

    # 종목 추가/삭제 업데이트
    for ticker, change in parsed_changes.items():
        if ticker in existing_portfolio:
            # 기존 종목은 수량을 업데이트 (매수는 더하고, 매도는 뺌)
            existing_portfolio[ticker] += change
            # 매도로 인해 보유수량이 0 이하가 되면 포트폴리오에서 제거
            if existing_portfolio[ticker] <= 0:
                cursor.execute("DELETE FROM 포트폴리오DB WHERE 고객ID = ? AND 주식명 = ?", (customer_id, ticker))
        else:
            # 새로운 종목은 추가
            existing_portfolio[ticker] = change

    # 업데이트된 포트폴리오를 DB에 저장 (0 이하인 종목은 제거)
    for ticker, count in existing_portfolio.items():
        if count > 0:
            cursor.execute('''
            UPDATE 포트폴리오DB
            SET 보유수량 = ?
            WHERE 고객ID = ? AND 주식명 = ?
            ''', (count, customer_id, ticker))

    conn.commit()

# 10. 신규 고객 등록 함수 (고객ID 반환)
def register_new_customer(customer_id, name, age, balance, skill_level, password):
    cursor.execute('''
    INSERT INTO 고객DB (고객ID, 이름, 나이, 보유자금, 투자실력, 비밀번호)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (customer_id, name, age, balance, skill_level, password))
    conn.commit()

# 11. 고객 지표 저장 함수
def save_customer_metrics_to_db(customer_id, metrics):
    cursor.execute('''
    INSERT INTO 고객지표DB (고객ID, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수, 최대보유섹터)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (customer_id, metrics['베타계수'], metrics['수익률표준편차'], metrics['트렌드지수'], metrics['투자심리지수'], metrics['최대보유섹터']))
    conn.commit()

# 12. 지표 계산 함수
def calculate_customer_metrics(customer_id, portfolio_df, stock_info=주식별지표):
    
    merged_df = pd.merge(stock_info, portfolio_df, on='티커종목코드')

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
def process_portfolio_and_metrics(customer_id, portfolio_df):
    # 고객의 지표 계산
    customer_metrics = calculate_customer_metrics(customer_id, portfolio_df)
    save_customer_metrics_to_db(customer_id, customer_metrics)

    return pd.DataFrame([customer_metrics])

# 14. 고객 포트폴리오 불러오기 함수
def load_portfolio_from_db(customer_id):
    query = '''
    SELECT 티커종목코드, 보유수량, 비중
    FROM 포트폴리오DB
    WHERE 고객ID = ?
    '''
    # SQL 쿼리를 사용하여 고객의 포트폴리오를 불러오고, 데이터프레임으로 변환
    portfolio_df = pd.read_sql_query(query, conn, params=(customer_id,))
    
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
    customer_metrics_df = pd.read_sql_query(query, conn)
    return customer_metrics_df

# 16. 고객 ID 중복 여부 확인 함수
def is_customer_id_exist(customer_id):
    cursor.execute("SELECT 1 FROM 고객DB WHERE 고객ID = ?", (customer_id,))
    return cursor.fetchone() is not None

# 17. 고객 지표와 ETF 지표를 사용하여 코사인 유사도 계산 함수
def recommend_etfs(customer_metrics, curation_metrics=큐레이션지표, top_n=5):
    # 고객 지표와 ETF 지표 중 비교할 열들 (트렌드지수, 베타계수, 수익률표준편차, 투자심리지수)
    metrics_columns = ['트렌드지수', '베타계수', '수익률표준편차', '투자심리지수']
    
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
# 긴 문장에서 핵심어 추출
def keyphrase_similarity(query, long_document, model=model):
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
    '부기': '젠틀맨',
    '랑이': '왕',
    '아웅이': '아저씨',
    '숭이': '어린이',
}

def classify_customer_tone(customer_metrics_df, group_metrics_df=group_metrics, tone_dict=tone_dict):
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

    # 고객 지표 표준화
    scaler_customer = StandardScaler()
    scaled_customer_metrics = scaler_customer.fit_transform(customer_metrics_df[metrics_columns])

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
    '섹터',
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
        sector = etf_info['섹터']

        # 사용자 지표와 비교
        user_trend = user_metrics['트렌드지수'].values[0]
        user_beta = user_metrics['베타계수'].values[0]
        user_volatility = user_metrics['수익률표준편차'].values[0]
        user_invest = user_metrics['투자심리지수'].values[0]

        description, desc_dist = find_similar_document(f"{etf_name} 설명", index, documents, model, top_k=3)

        metrics = {
            '트렌드지수': abs(trend-user_trend),
            '베타계수': abs(beta-user_beta),
            '수익률표준편차': abs(volatility-user_volatility),
            '투자심리지수': abs(invest-user_invest)
        }

        min_metric = min(metrics, key=metrics.get)

        metric_description, metric_dist = find_similar_document(f"{min_metric} 설명", index, documents, model, top_k=1)

        # GPT에게 전달할 프롬프트
        prompt = f"""
        ETF {etf_name}는 다음과 같은 특징을 가지고 있습니다:
        - 트렌드 지수: {trend}
        - 베타 계수: {beta}
        - 수익률 표준 편차: {volatility}
        - 투자 심리 지수: {invest}
        - 섹터: {sector}
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

        client = OpenAI(
            api_key=API_KEY,
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo"
        )

        explanation = response.choices[0].message.content
        explanations.append(f"{etf_name} 추천 이유:\n{explanation}\n\n")
        db_scores.append(round((desc_dist.mean()+metric_dist.mean())/2, 3)*100)
        answer_scores.append(evaluate_clarity(explanation))
        answer_db_scores.append(round(calculate_cosine_similarity(explanation, prompt), 3)*100)

    return explanations, db_scores, answer_scores, answer_db_scores

# 23. GPT 응답 함수
def gpt_response_basic(prompt):

    client = OpenAI(
        api_key=API_KEY,
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo"
    )
    
    return response.choices[0].message.content

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
                                cursor.execute('''
                                UPDATE 포트폴리오DB
                                SET 보유수량 = 보유수량 + ?
                                WHERE 고객ID = ? AND 티커종목코드 = ?
                                ''', (change_amount, customer_id, ticker))
                                conn.commit()
                            elif change_type == "매도":
                                cursor.execute('''
                                UPDATE 포트폴리오DB
                                SET 보유수량 = 보유수량 - ?
                                WHERE 고객ID = ? AND 티커종목코드 = ?
                                ''', (change_amount, customer_id, ticker))
                                conn.commit()

                            if (ticker=="종료")|(change_type=="종료")|(change_amount=="종료"):
                                break

                        print(f"\n{ticker} 주식의 포트폴리오가 성공적으로 업데이트되었습니다.\n")

                    elif choice == "3":
                        while True:
                            # GPT를 통해 포트폴리오 수정
                            user_input = input("챗봇에게 보낼 포트폴리오 입력 (예: 애플 10주 추가 매수, 테슬라 5주 매도, 엔비디아 3주 신규 매수했습니다.): ")
                            gpt_response = get_gpt_response("포트폴리오 업데이트", user_input)
                            print("\n챗봇 응답:\n", gpt_response, '\n\n')

                            parsed_portfolio = parse_portfolio_response(gpt_response)

                            print(parsed_portfolio)

                            portfolio_check = input("변경하시려는 수량이 맞나요?\n 답변 (네/아니오): ")

                            if portfolio_check=="네":
                                # 기존 포트폴리오에 업데이트
                                update_existing_portfolio(customer_id, parsed_portfolio)
                            elif portfolio_check=="아니오":
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
                    gpt_response = get_gpt_response("포트폴리오 입력", user_input)
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
                gpt_response = get_gpt_response("포트폴리오 입력", user_input)
                parsed_portfolio = parse_portfolio_response(gpt_response)
                portfolio_df = calculate_portfolio(parsed_portfolio)
                print("고객님의 포트폴리오:")
                print(portfolio_df, '\n\n')
            else:
                break

            customer_metrics = process_portfolio_and_metrics(customer_id, portfolio_df)

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
            print(f"고객 ID {customer_id}님은 '{tone_result[customer_id]}' 말투로 안내됩니다.\n")

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

            gpt_reply = gpt_response_basic(prompt)
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
                gpt_reply = gpt_response_basic(gpt_prompt)
                
                # GPT 답변 출력
                print(f"사용자: {customer_response}\n")
                print("챗봇 답변: ", gpt_reply, '\n\n')

# 단계적 프로그램 실행을 원하시면 아래의 주석들을 풀고 진행해주세요. 어투변환 예시를 보시려면 풀지 않으셔도 됩니다.
# portfolio_input_loop(index, documents, model)

# conn.close()


# 고객 어투 변환 예시
def AI_test(customer_id, index=index, documents=documents ,model=model):
    portfolio_df=load_portfolio_from_db(customer_id)

    customer_metrics = process_portfolio_and_metrics(customer_id, portfolio_df)

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
    recommended_etfs = recommend_etfs(scaled_customer_metrics)
    for etf in recommended_etfs['티커종목코드']:
        etfs.append(etf)

    # 고객을 그룹과 매칭하여 어투 변환 진행
    tone_result, group_result = classify_customer_tone(customer_metrics_df)

    # 고객 어투 변환 결과 안내
    print(f"고객 ID {customer_id}님은 '{tone_result[customer_id]}' 말투로 안내됩니다.\n")

    # 유사한 문장 검색
    similar_documents, _ = find_similar_document(f"{group_result}에 대한 설명", index, documents, model, top_k=1)
    
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
        4. 고객은 위 그룹에 속한 고객입니다. 고객이 충분히 이해하도록 "고객이 그룹했다는 점"과 "위에서 학습한 말투"로 인사를 하고 "학습한 말투"로 "고객의 그룹"을 설명하세요.

        답변 구성: 학습한 말투로 인사, 학습한 말투로 고객이 어떤 그룹에 속해있는지, 학습한 말투로 그룹 설명
    '''
    # 어투변환을 하시려면 아래 주석을 제거해주세요. openai 버전은 반드시 1.51.1이어야 합니다.
    # gpt_reply = gpt_response_basic(prompt)
    # print(gpt_reply, '\n\n', '그룹 분류 신뢰도: ', round(keyphrase_similarity(f"{group_result}에 대한 설명", similar_documents[0])*100, 2), '%')
    
    return etfs

etfs = []

for cus in ['0번고객', '1번고객', '2번고객', '3번고객']:
    etfs.append(AI_test(cus))

# 뉴스, 수익률 변화 추이 제공 서비스
etf_list = ['SHYG', 'PDT', 'PCN', 'TECL', 'RETL', 'YINN', 'XHB', 'ITB', 'EWY', 'PAVE', 'VUG', 'YMAX']
# 종목조회건수의 z-score가 0.95 이상인 날을 아웃라이어로 설정하고 새로운 열 추가
def add_outlier_column(data, threshold=0.7):
    # 종목조회건수의 z-score 계산
    mean_diff = data['종목조회건수'].mean()
    std_diff = data['종목조회건수'].std()
    z_scores = (data['종목조회건수'] - mean_diff) / std_diff

    # 새로운 열 추가: z-score가 threshold 이상이면 1, 그렇지 않으면 0
    data['조회건수_아웃라이어'] = np.where(z_scores >= threshold, 1, 0)
    return data

# 예제 데이터: 종목별로 그룹화한 후 아웃라이어 열 추가
서비스계산 = pd.merge(주식별고객정보, 종목일자별시세_보완, left_on=['티커종목코드', '기준일자'], right_on=['티커종목코드', '거래일자'])

서비스계산 = 서비스계산[서비스계산['티커종목코드'].isin(큐레이션지표['티커종목코드'])]

# 종목별로 그룹화하여 아웃라이어 열 추가
서비스계산 = 서비스계산.groupby('티커종목코드').apply(lambda group: add_outlier_column(group.sort_values('기준일자'))).reset_index(drop=True)[['기준일자', '티커종목코드', '종목조회건수', '조회건수_아웃라이어', '전일대비증감율']]

# 증가/감소 변화 여부를 추출하는 함수
def get_change_pattern(changes):
    return [np.sign(changes[i] * changes[i - 1]) < 0 for i in range(1, len(changes))]

# 변화 패턴이 유사한지 확인하는 함수
def calculate_pattern_similarity(ref_pattern, comp_pattern, days):
    similarity = 0
    for i in range(days):
        if i >= len(ref_pattern) or i >= len(comp_pattern):
            break
        if ref_pattern[i] == comp_pattern[i]:
            similarity += 1
        else:
            break
    return similarity / days  # days에 따라 평균화

# 종목별로 가장 최근 아웃라이어 이후의 변화 패턴을 찾고 유사성을 계산하는 함수
def find_similar_patterns_for_last_outlier(group):
    outlier_indices = group[group['조회건수_아웃라이어'] == 1].index
    if len(outlier_indices) < 2:
        return None

    last_outlier_idx = outlier_indices[-1]
    last_outlier_changes = group['전일대비증감율'].iloc[last_outlier_idx:last_outlier_idx + 4].values
    ref_pattern = get_change_pattern(last_outlier_changes)

    similarities = {'티커종목코드': group['티커종목코드'].iloc[0]}

    # days별로 유사성 점수의 평균 계산
    for days in range(1, 4):
        day_similarities = []
        for idx in outlier_indices[:-1]:
            previous_changes = group['전일대비증감율'].iloc[idx:idx + 4].values
            comp_pattern = get_change_pattern(previous_changes)
            
            # 유사성 점수 계산
            similarity_score = calculate_pattern_similarity(ref_pattern, comp_pattern, days)
            day_similarities.append(similarity_score)

        # days별 유사성 점수의 평균 저장
        similarities[f'유사성_평균_{days}일'] = np.mean(day_similarities)

    return pd.DataFrame([similarities])

# 종목별로 그룹화하고 유사성 계산
results = []
for ticker, group in 서비스계산.groupby('티커종목코드'):
    group = group.sort_values('기준일자').reset_index(drop=True)
    similarities_df = find_similar_patterns_for_last_outlier(group)
    if similarities_df is not None:
        results.append(similarities_df)

# 결과를 데이터프레임으로 변환하여 출력
results_df = pd.concat(results, ignore_index=True)

# 결과 출력
print(results_df.describe())
print(results_df)

# 여러 종목 코드에 대해 마지막 아웃라이어 날짜와 유사성을 반환하는 함수
def get_similarity_for_multiple_tickers(data, ticker_codes):
    results = []

    for ticker_code in ticker_codes:
        # 해당 종목 코드를 필터링
        group = data[data['티커종목코드'] == ticker_code].sort_values('기준일자').reset_index(drop=True)
        outlier_indices = group[group['조회건수_아웃라이어'] == 1].index

        if len(outlier_indices) < 2:
            continue  # 아웃라이어가 2개 미만인 경우 건너뜀

        # 마지막 아웃라이어의 인덱스 및 변화 패턴 추출
        last_outlier_idx = outlier_indices[-1]
        last_outlier_changes = group['전일대비증감율'].iloc[last_outlier_idx:last_outlier_idx + 4].values
        ref_pattern = get_change_pattern(last_outlier_changes)

        # days별 유사성 점수 계산 및 저장
        similarities = {'티커종목코드': ticker_code, '마지막_아웃라이어_날짜': group['기준일자'].iloc[last_outlier_idx]}
        for days in range(1, 4):
            day_similarities = []
            for idx in outlier_indices[:-1]:  # 마지막 아웃라이어 이전의 아웃라이어들만 사용
                previous_changes = group['전일대비증감율'].iloc[idx:idx + 4].values
                comp_pattern = get_change_pattern(previous_changes)
                
                # 유사성 점수 계산
                similarity_score = calculate_pattern_similarity(ref_pattern, comp_pattern, days)
                day_similarities.append(similarity_score)

            # days별 유사성 점수의 평균 저장
            similarities[f'유사성_평균_{days}일'] = np.mean(day_similarities)

        # 결과 저장
        results.append(similarities)

    return pd.DataFrame(results)

# 예시 사용법
print(get_similarity_for_multiple_tickers(서비스계산, etf_list))

# 증가/감소 변화 여부를 추출하는 함수
def get_change_pattern(changes):
    """
    주어진 변화율 데이터를 받아 증가/감소 패턴을 반환
    """
    return [np.sign(changes[i] * changes[i - 1]) < 0 for i in range(1, len(changes))]

# 변화 패턴이 유사한지 확인하는 함수
def calculate_pattern_similarity(ref_pattern, comp_pattern):
    """
    두 패턴(ref_pattern, comp_pattern)의 유사성을 계산
    """
    return (sum(r == c for r, c in zip(ref_pattern, comp_pattern))-1) / (len(ref_pattern)-1)

# 종목별 마지막 아웃라이어 이후 변화 패턴과 과거 데이터 비교
def find_similar_patterns_for_last_outlier(group, max_days=4, top_n=3):
    """
    특정 그룹(종목)에 대해 마지막 아웃라이어 이후 변화 패턴을 찾고
    유사성이 높은 상위 n개를 반환
    """
    outlier_indices = group[group['조회건수_아웃라이어'] == 1].index
    if len(outlier_indices) < 2:
        return None

    # 마지막 아웃라이어 이후 변화 패턴
    last_outlier_idx = outlier_indices[-1]
    last_outlier_changes = group['전일대비증감율'].iloc[last_outlier_idx:last_outlier_idx + max_days + 1].values
    ref_pattern = get_change_pattern(last_outlier_changes)

    similarities = []
    for idx in outlier_indices[:-1]:
        # 비교 대상 패턴 추출
        previous_changes = group['전일대비증감율'].iloc[idx:idx + max_days + 1].values
        comp_pattern = get_change_pattern(previous_changes)
        
        # 유사도 계산
        if len(comp_pattern) == len(ref_pattern):
            similarity_score = calculate_pattern_similarity(ref_pattern, comp_pattern)
            similarities.append({
                '티커종목코드': group['티커종목코드'].iloc[0],
                '비교_기준일자': group['기준일자'].iloc[idx],
                '유사도': similarity_score
            })

    # 유사성 기준 상위 n개 선택
    similarities_df = pd.DataFrame(similarities)
    if not similarities_df.empty:
        return similarities_df.nlargest(top_n, '유사도')
    return None

# 종목별로 그룹화하고 유사성 계산
results = []
for ticker, group in 서비스계산.groupby('티커종목코드'):
    group = group.sort_values('기준일자').reset_index(drop=True)
    similar_patterns = find_similar_patterns_for_last_outlier(group, max_days=4, top_n=3)
    if similar_patterns is not None:
        results.append(similar_patterns)

# 결과를 데이터프레임으로 변환
results_df = pd.concat(results, ignore_index=True)

# 결과 출력
print(results_df)

뉴스서비스 = pd.merge(results_df[results_df['티커종목코드'].isin(etf_list)], 큐레이션지표[['티커종목코드', '섹터']], on='티커종목코드', how='left')
뉴스서비스.to_csv('ETF뉴스서비스.csv', encoding='cp949', index=False)