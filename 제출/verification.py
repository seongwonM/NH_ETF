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
import os

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

'''
파생변수 유의성 검증
1. 수익률_표준편차
'''

## 고객분류 최종본 불러오기
베타계수 = pd.read_csv('베타계수_ETF.csv', encoding='cp949')
수익률_표준편차 = pd.read_csv('표준편차_ETF.csv',encoding='cp949')
화제성 = pd.read_csv('화제성_ETF.csv',encoding='cp949' )
그룹별지표 = pd.read_csv('그룹별지표.csv', encoding='cp949')
# 파생변수 검증에 필요한 지표만 남기기
그룹별지표 = 그룹별지표.drop(columns='최대보유섹터')

# '티커종목코드' 기준으로 병합
고객분류지표 = 베타계수.merge(수익률_표준편차, on='티커종목코드', how='inner')\
                   .merge(화제성, on='티커종목코드', how='inner')

'''
### ETF점수정보 테이블을 가지고 파생변수 지표 검증
- 생성한 파생변수의 유의성을 검증하기 위해 ETF점수정보의 열과 상관관계 측정
- ETF점수정보에서 '티커종목코드' 별로 지표 계산
'''

# 'etf_iem_cd' 별로 그룹화한 후, 각 지표에 대해 평균을 계산
# 대상 기간은 20240528-20240826로, 이 범위 내의 데이터를 필터링
ETF점수정보['거래일자'] = pd.to_datetime(ETF점수정보['거래일자'], format='%Y%m%d')
ETF점수정보_거래일자 = ETF점수정보[(ETF점수정보['거래일자'] >= '2024-05-28') & (ETF점수정보['거래일자'] <= '2024-08-26')]

# 'etf_iem_cd'별로 그룹화한 후 각 지표의 평균을 계산합니다.
평가지표 = ['1개월총수익율', '3개월총수익율', '1년총수익율', 'ETF점수', 'ETFZ점수',
                      'Z점수순위', '누적수익율Z점수', '정보비율Z점수', '샤프지수Z점수', 
                      '상관관계Z점수', '트래킹에러Z점수', '최대낙폭Z점수', '변동성Z점수']

# etf_iem_cd 별로 평균 구하기
티커별_ETF점수정보 = ETF점수정보_거래일자.groupby('etf_iem_cd')[평가지표].mean()

# 인덱스를 열로 변환
티커별_ETF점수정보 = 티커별_ETF점수정보.reset_index()

# 'etf_iem_cd'를 '티커종목코드'로 변경
티커별_ETF점수정보 = 티커별_ETF점수정보.rename(columns={'etf_iem_cd': '티커종목코드'})


'''
##### 1. 수익률 표준편차
- '트레킹에러Z점수'(0.707225)와 '변동성Z점수'(-0.713182) 높은 상관관계를 가진 것을 확인할 수 있습니다.
'''

columns_to_correlate = ['화제성', '베타계수', '수익률표준편차'] + 티커별_ETF점수정보.select_dtypes(include=['float64', 'int64']).columns.tolist()

merged_df = pd.merge(고객분류지표, ETF점수정보, left_on='티커종목코드', right_on='etf_iem_cd', how='inner')

전체혼동행렬 = merged_df[columns_to_correlate].corr()

# '수익률 표준편차'와 다른 모든 변수들에 대한 상관관계를 추출
수익률표준편차_상관계수 = 전체혼동행렬.loc[['수익률표준편차']]

# 결과 출력
print(수익률표준편차_상관계수)

'''
파생변수 유의성 검증
2. 베타계수
- 랜랜덤포레스트 모델 학습을 통해 변수중요도가 높은 변수 변환을 통한 베타계수 유의성 검증
- 'log_변동성Z점수'(-0.607678)와 높은 상관관계를 가진 것을 확인할 수 있습니다.
'''

from sklearn.ensemble import RandomForestRegressor

# X 변수 선택
X = merged_df[['1개월총수익율', '3개월총수익율', '1년총수익율', 'ETF점수', 'ETFZ점수', 
               'Z점수순위', '누적수익율Z점수', '정보비율Z점수', '샤프지수Z점수', 
               '상관관계Z점수', '트래킹에러Z점수', '최대낙폭Z점수', '변동성Z점수']]

# 랜덤 포레스트 모델 학습
model = RandomForestRegressor()
model.fit(X, merged_df['베타계수'])

# 설명력 출력
r_squared = model.score(X, merged_df['베타계수'])
print(f"R² value: {r_squared}")

# 변수 중요도 확인
importances = pd.Series(model.feature_importances_, index=X.columns)
print("변수중요도 확인:")
print(importances.sort_values(ascending=False))

# 변수 중요도가 높은 변수들의 로그 변환
merged_df['log_ETFZ점수'] = np.log1p(merged_df['ETFZ점수'])
merged_df['log_트래킹에러Z점수'] = np.log1p(merged_df['트래킹에러Z점수'])
merged_df['log_변동성Z점수'] = np.log1p(merged_df['변동성Z점수'])

# 상관관계 계산
베타계수_혼동행렬 = merged_df[['베타계수', 'log_ETFZ점수', 'log_트래킹에러Z점수', 'log_변동성Z점수']].corr()

# 베타계수와 다른 변수 간의 상관계수만 추출
베타계수_관련_혼동행렬 = 베타계수_혼동행렬[['베타계수']]

# 결과 출력
print("베타계수와 다른 변수들과의 상관계수:")
print(베타계수_관련_혼동행렬)