
# # ---

'''
# DB를 생성하는 코드입니다. 따라서 실행시 요금이 부과되어 주석처리하였습니다.
# 또한, 에선에서 만들었던 DB를 마이그레이션하는 코드가 포함되어있어 DB가 완성된 현재, 오류가 발생 할 수 있습니니다. 
# 따라서 주석처리하였습니다. 
'''

# # --- 

# import pyodbc
# import sqlite3
# import pandas as pd
# from sqlalchemy import create_engine, inspect, text
# import yfinance as yf
# import json
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity
# import requests

# ### 기본 DB setup
# # Azure SQL Database 연결 설정
# conn_str = (
#     'DRIVER={ODBC Driver 18 for SQL Server};'
#     'SERVER=jy-nh-server01.database.windows.net,1433;'
#     'DATABASE=NH-SQLDB-001;'
#     'UID=jyun22;'
#     'PWD=1emdrkwk!;'
#     'Encrypt=yes;'
#     'TrustServerCertificate=no;'
#     'Connection Timeout=100;'
# )

# # 데이터베이스 연결
# conn = pyodbc.connect(conn_str)
# cursor = conn.cursor()
# print("Azure SQL Database에 연결 성공")

# # 모든 외래 키 제약 조건 삭제
# cursor.execute("""
# DECLARE @sql NVARCHAR(MAX) = N'';
# SELECT @sql += N'ALTER TABLE ' + QUOTENAME(OBJECT_SCHEMA_NAME(parent_object_id)) + '.'
#                + QUOTENAME(OBJECT_NAME(parent_object_id)) 
#                + ' DROP CONSTRAINT ' + QUOTENAME(name) + ';'
# FROM sys.foreign_keys;
# EXEC sp_executesql @sql;
# """)
# conn.commit()

# # 기존 테이블 삭제
# cursor.execute("IF OBJECT_ID('dbo.포트폴리오DB', 'U') IS NOT NULL DROP TABLE dbo.포트폴리오DB")
# cursor.execute("IF OBJECT_ID('dbo.고객지표DB', 'U') IS NOT NULL DROP TABLE dbo.고객지표DB")
# cursor.execute("IF OBJECT_ID('dbo.고객DB', 'U') IS NOT NULL DROP TABLE dbo.고객DB")
# conn.commit()

# # 고객DB 테이블 생성
# cursor.execute('''
# CREATE TABLE 고객DB (
#     고객ID NVARCHAR(50) PRIMARY KEY,
#     비밀번호 NVARCHAR(50),
#     이름 NVARCHAR(50),
#     나이 INT,
#     보유자금 INT,
#     투자실력 NVARCHAR(50)
# )
# ''')

# # 포트폴리오DB 테이블 생성
# cursor.execute('''
# CREATE TABLE 포트폴리오DB (
#     고객ID NVARCHAR(50),
#     티커종목코드 NVARCHAR(50),
#     보유수량 FLOAT,
#     비중 FLOAT,
#     PRIMARY KEY (고객ID, 티커종목코드),
#     FOREIGN KEY (고객ID) REFERENCES 고객DB(고객ID)
# )
# ''')

# # 고객지표DB 테이블 생성
# cursor.execute('''
# CREATE TABLE 고객지표DB (
#     지표ID INT PRIMARY KEY IDENTITY(1,1),
#     고객ID NVARCHAR(50),
#     베타계수 FLOAT,
#     수익률표준편차 FLOAT,
#     트렌드지수 FLOAT,
#     투자심리지수 FLOAT,
#     최대보유섹터 NVARCHAR(50),
#     FOREIGN KEY (고객ID) REFERENCES 고객DB(고객ID)
# )
# ''')

# # 초기 고객 데이터 삽입
# customers = [
#     ('0번고객', '0000', '김철수', 31, 694140000, '고수'),
#     ('1번고객', '1111', '이영희', 54, 443650000, '일반'),
#     ('2번고객', '2222', '박민수', 25, 29830000, '일반'),
#     ('3번고객', '3333', '최영식', 64, 272410000, '고수')
# ]

# for customer in customers:
#     cursor.execute('''
#         INSERT INTO 고객DB (고객ID, 비밀번호, 이름, 나이, 보유자금, 투자실력)
#         VALUES (?, ?, ?, ?, ?, ?)
#     ''', customer)

# conn.commit()
# print("초기 데이터가 성공적으로 추가되었습니다.")
# conn.close()


# ### data_migration
# # SQLite 데이터베이스 연결
# sqlite_conn = sqlite3.connect('customer_portfolio.db')

# # Azure SQL Database 연결 설정
# conn_str = (
#     'DRIVER={ODBC Driver 18 for SQL Server};'
#     'SERVER=jy-nh-server01.database.windows.net,1433;'
#     'DATABASE=NH-SQLDB-001;'
#     'UID=jyun22;'
#     'PWD=1emdrkwk!;'
#     'Encrypt=yes;'
#     'TrustServerCertificate=no;'
#     'Connection Timeout=100;'
# )

# # 데이터베이스 연결
# conn = pyodbc.connect(conn_str)
# cursor = conn.cursor()

# # SQLite 테이블 데이터 로드
# customers_df = pd.read_sql_query("SELECT * FROM 고객DB", sqlite_conn)
# portfolio_df = pd.read_sql_query("SELECT * FROM 포트폴리오DB", sqlite_conn)
# metrics_df = pd.read_sql_query("SELECT * FROM 고객지표DB", sqlite_conn)
# sqlite_conn.close()
# print("SQLite에서 데이터 로드 완료")

# # 기존 데이터 삭제
# cursor.execute("DELETE FROM 고객DB")
# cursor.execute("DELETE FROM 포트폴리오DB")
# cursor.execute("DELETE FROM 고객지표DB")
# conn.commit()
# print("기존 데이터 삭제 완료")

# # 새로운 데이터 삽입
# # 고객DB 데이터 삽입
# for _, row in customers_df.iterrows():
#     cursor.execute('''
#         INSERT INTO 고객DB (고객ID, 비밀번호, 이름, 나이, 보유자금, 투자실력) 
#         VALUES (?, ?, ?, ?, ?, ?)
#     ''', row['고객ID'], row['비밀번호'], row['이름'], row['나이'], row['보유자금'], row['투자실력'])

# # 포트폴리오DB 데이터 삽입
# for _, row in portfolio_df.iterrows():
#     cursor.execute('''
#         INSERT INTO 포트폴리오DB (고객ID, 티커종목코드, 보유수량, 비중) 
#         VALUES (?, ?, ?, ?)
#     ''', row['고객ID'], row['티커종목코드'], row['보유수량'], row['비중'])

# # 고객지표DB 데이터 삽입
# for _, row in metrics_df.iterrows():
#     cursor.execute('''
#         INSERT INTO 고객지표DB (고객ID, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수, 최대보유섹터) 
#         VALUES (?, ?, ?, ?, ?, ?)
#     ''', row['고객ID'], row['베타계수'], row['수익률표준편차'], row['트렌드지수'], row['투자심리지수'], row['최대보유섹터'])

# conn.commit()
# print("Azure SQL Database로 데이터 마이그레이션 완료")
# conn.close()

# ### data_insertion.py
# # Azure SQL Database 연결 설정
# conn_str = (
#     'DRIVER={ODBC Driver 18 for SQL Server};'
#     'SERVER=jy-nh-server01.database.windows.net,1433;'
#     'DATABASE=NH-SQLDB-001;'
#     'UID=jyun22;'
#     'PWD=1emdrkwk!;'
#     'Encrypt=yes;'
#     'TrustServerCertificate=no;'
#     'Connection Timeout=90;'
# )

# # SQLAlchemy 엔진 생성
# engine = create_engine("mssql+pyodbc:///?odbc_connect={}".format(conn_str))

# # pyodbc 연결
# connection = pyodbc.connect(conn_str)
# cursor = connection.cursor()

# # NewsData 테이블 생성 함수
# def create_news_data_table():
#     try:
#         # NewsData 테이블이 이미 존재하는지 확인
#         cursor.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'NewsData'")
#         table_exists = cursor.fetchone()[0] > 0

#         # 테이블이 없으면 생성
#         if not table_exists:
#             create_table_query = """
#             CREATE TABLE NewsData (
#                 Date VARCHAR(8) COLLATE Korean_Wansung_CI_AS, 
#                 Title NVARCHAR(MAX) COLLATE Korean_Wansung_CI_AS
#             );
#             """
#             cursor.execute(create_table_query)
#             connection.commit()  # 명시적 커밋
#             print("'NewsData' 테이블을 Korean_Wansung_CI_AS 정렬 규칙으로 생성했습니다.")
#         else:
#             print("'NewsData' 테이블이 이미 존재하므로 생성하지 않았습니다.")
#     except Exception as e:
#         print(f"'NewsData' 테이블 생성 중 오류 발생: {e}")

# # NewsData 테이블 생성 함수 호출
# create_news_data_table()


# # 테이블 생성 후 데이터가 없을 경우에만 삽입하는 함수
# def check_and_create_table(table_name, df, collation=None):
#     inspector = inspect(engine)
    
#     # 테이블이 존재하지 않는 경우에만 생성
#     if table_name not in inspector.get_table_names():
#         try:
#             with engine.connect() as connection:
#                 # group_metrics 테이블 생성
#                 if table_name == 'group_metrics' and collation:
#                     create_table_query = f"""
#                     CREATE TABLE {table_name} (
#                         그룹명 NVARCHAR(100) COLLATE {collation},
#                         베타계수 FLOAT,
#                         수익률표준편차 FLOAT,
#                         트렌드지수 FLOAT,
#                         투자심리지수 FLOAT,
#                         최대보유섹터 NVARCHAR(50),
#                         PRIMARY KEY (그룹명)
#                     )
#                     """
#                     connection.execute(text(create_table_query))
#                     print(f"'{table_name}' 테이블을 Korean_Wansung_CI_AS 정렬 규칙으로 생성했습니다.")
                
#                 # 어투변환_로우데이터 테이블 생성
#                 elif table_name == '어투변환_로우데이터' and collation:
#                     create_table_query = f"""
#                     CREATE TABLE {table_name} (
#                         formal NVARCHAR(500) COLLATE {collation},
#                         informal NVARCHAR(500) COLLATE {collation},
#                         android NVARCHAR(500) COLLATE {collation},
#                         azae NVARCHAR(500) COLLATE {collation},
#                         chat NVARCHAR(500) COLLATE {collation},
#                         choding NVARCHAR(500) COLLATE {collation},
#                         emoticon NVARCHAR(500) COLLATE {collation},
#                         enfp NVARCHAR(500) COLLATE {collation},
#                         gentle NVARCHAR(500) COLLATE {collation},
#                         halbae NVARCHAR(500) COLLATE {collation},
#                         halmae NVARCHAR(500) COLLATE {collation},
#                         joongding NVARCHAR(500) COLLATE {collation},
#                         king NVARCHAR(500) COLLATE {collation},
#                         naruto NVARCHAR(500) COLLATE {collation},
#                         seonbi NVARCHAR(500) COLLATE {collation},
#                         sosim NVARCHAR(500) COLLATE {collation},
#                         translator NVARCHAR(500) COLLATE {collation}
#                     )
#                     """
#                     connection.execute(text(create_table_query))
#                     print(f"'{table_name}' 테이블을 Korean_Wansung_CI_AS 정렬 규칙으로 생성했습니다.")

#                 # 큐레이션지표 테이블 생성
#                 elif table_name == '큐레이션지표' and collation:
#                     create_table_query = f"""
#                     CREATE TABLE {table_name} (
#                         티커종목코드 NVARCHAR(50),
#                         트렌드지수 FLOAT,
#                         베타계수 FLOAT,
#                         수익률표준편차 FLOAT,
#                         투자심리지수 FLOAT,
#                         섹터분류명 NVARCHAR(100) COLLATE {collation}
#                     )
#                     """
#                     connection.execute(text(create_table_query))
#                     print(f"'{table_name}' 테이블을 Korean_Wansung_CI_AS 정렬 규칙으로 생성했습니다.")
                
#                 # 테이블 생성 후 데이터 삽입
#                 df.to_sql(table_name, engine, if_exists='append', index=False)
#                 print(f"'{table_name}' 테이블에 데이터를 삽입했습니다.")
#         except Exception as e:
#             print(f"테이블 생성 중 오류 발생: {e}")
#     else:
#         # 테이블이 이미 존재할 경우, 데이터가 비어 있는지 확인 후 삽입
#         with engine.connect() as connection:
#             try:
#                 # 테이블의 데이터가 비어 있는지 확인
#                 result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
#                 row_count = result.scalar()

#                 if row_count == 0:
#                     # 데이터가 비어 있을 경우에만 데이터 삽입
#                     df.to_sql(table_name, engine, if_exists='append', index=False)
#                     print(f"'{table_name}' 테이블에 데이터를 삽입했습니다.")
#                 else:
#                     print(f"'{table_name}' 테이블에 이미 데이터가 존재하므로 삽입하지 않았습니다.")
#             except Exception as e:
#                 print(f"데이터 삽입 중 오류 발생: {e}")




# # 데이터 파일 불러오기 (encoding='cp949' 추가)
# 주식별지표 = pd.read_csv('./data/주식별지표.csv', encoding='cp949')
# 큐레이션지표 = pd.read_csv('./data/큐레이션지표.csv')
# 그룹별주식비율 = pd.read_csv('./data/그룹별주식비율.csv', encoding='cp949')
# ETF베타계수 = pd.read_csv('./data/베타계수_ETF.csv', encoding='cp949')
# 종가_조회수_상관관계 = pd.read_csv('./data/종가_조회수_상관관계.csv', encoding='cp949')
# ETF투자심리지수 = pd.read_csv('./data/투자심리지수_ETF.csv', encoding='cp949')
# ETF표준편차 = pd.read_csv('./data/표준편차_ETF.csv', encoding='cp949')
# ETF화제성 = pd.read_csv('./data/화제성_ETF.csv', encoding='cp949')

# 어투변환_로우데이터 = pd.read_csv('./data/smilestyle_dataset.tsv', sep='\t') 
# group_metrics = pd.read_csv('./data/그룹별지표.csv')

# # (제공)본선데이터 8개
# 고객보유정보 = pd.read_csv('./data/고객보유정보.csv')
# 매수매도계좌정보 = pd.read_csv('./data/매수매도계좌정보.csv')
# 종목일자별시세 = pd.read_csv('./data/종목일자별시세.csv')
# 주식일별정보 = pd.read_csv('./data/주식일별정보.csv')
# 해외종목정보 = pd.read_csv('./data/해외종목정보.csv')
# ETF구성종목정보 = pd.read_csv('./data/ETF구성종목정보.csv')
# ETF배당내역 = pd.read_csv('./data/ETF배당내역.csv')
# ETF점수정보 = pd.read_csv('./data/ETF점수정보.csv')


# # 테이블 존재 여부 확인 후 데이터 삽입
# check_and_create_table('주식별지표', 주식별지표)
# check_and_create_table('그룹별주식비율', 그룹별주식비율)
# check_and_create_table('ETF베타계수', ETF베타계수)
# check_and_create_table('종가_조회수_상관관계', 종가_조회수_상관관계)
# check_and_create_table('ETF투자심리지수', ETF투자심리지수)
# check_and_create_table('ETF표준편차', ETF표준편차)
# check_and_create_table('ETF화제성', ETF화제성)

# # group_metrics 및 어투변환_로우데이터 테이블 생성 및 데이터 삽입 (Korean_Wansung_CI_AS 정렬 규칙 적용)
# check_and_create_table('group_metrics', group_metrics, collation='Korean_Wansung_CI_AS')
# check_and_create_table('어투변환_로우데이터', 어투변환_로우데이터, collation='Korean_Wansung_CI_AS')
# check_and_create_table('큐레이션지표', 큐레이션지표, collation='Korean_Wansung_CI_AS')

# check_and_create_table('고객보유정보', 고객보유정보)
# check_and_create_table('매수매도계좌정보', 매수매도계좌정보)
# check_and_create_table('종목일자별시세', 종목일자별시세)
# check_and_create_table('주식일별정보', 주식일별정보)
# check_and_create_table('해외종목정보', 해외종목정보)
# check_and_create_table('ETF구성종목정보', ETF구성종목정보)
# check_and_create_table('ETF배당내역', ETF배당내역)
# check_and_create_table('ETF점수정보', ETF점수정보)

# # pyodbc 연결 닫기
# engine.dispose()
# print("데이터 삽입 완료")


# ### data_loader
# # Azure SQL Database 연결
# conn_str = (
#     'DRIVER={ODBC Driver 18 for SQL Server};'
#     'SERVER=jy-nh-server01.database.windows.net,1433;'
#     'DATABASE=NH-SQLDB-001;'
#     'UID=jyun22;'
#     'PWD=1emdrkwk!;'
#     'Encrypt=yes;'
#     'TrustServerCertificate=no;'
#     'Connection Timeout=90;'
# )
# conn = pyodbc.connect(conn_str)

# # 그룹별 상위 10개 티커와 비중 계산 함수
# def calculate_top_10_by_group(group_df, group_column):
#     top_10_df = group_df[['티커종목코드', group_column]].nlargest(10, group_column)
#     total_shares = top_10_df[group_column].sum()
#     top_10_df['비중'] = top_10_df[group_column] / total_shares
#     return top_10_df[['티커종목코드', '비중']]

# # 전체 그룹별주식비율 데이터 로드
# group_df = pd.read_sql("SELECT * FROM 그룹별주식비율", conn)

# # 그룹별 상위 10개 티커와 비중 계산 실행
# group_columns = ['0번그룹', '1번그룹', '2번그룹', '3번그룹']
# top_10_results = {}

# for group_column in group_columns:
#     top_10_results[group_column] = calculate_top_10_by_group(group_df, group_column)

# # 보유 주식 수량 계산
# def calculate_shares(portfolio_dict, conn):
#     cursor = conn.cursor()

#     stock_prices = {}
#     portfolio_with_shares = []

#     for group_name, df in portfolio_dict.items():
#         for _, row in df.iterrows():
#             ticker = row['티커종목코드']
#             cursor.execute("SELECT 보유자금 FROM 고객DB WHERE 고객ID = ?", (group_name[:-2]+'고객',))
#             customer_funds = cursor.fetchone()[0]

#             stock = yf.Ticker(ticker)
#             latest_price = stock.history(period='1d')['Close'].iloc[-1]
#             allocated_funds = customer_funds * row['비중']
#             shares = allocated_funds / latest_price
#             portfolio_with_shares.append((group_name[:-2]+'고객', ticker, shares, row['비중']))

#     return portfolio_with_shares

# # 보유 주식 수량 계산 실행
# calculated_portfolio = calculate_shares(top_10_results, conn)

# # 포트폴리오DB에 계산된 보유 주식 수량 업데이트
# cursor = conn.cursor()
# for group, ticker, shares, ratio in calculated_portfolio:
#     cursor.execute('''
#     MERGE 포트폴리오DB AS target
#     USING (SELECT ? AS 고객ID, ? AS 티커종목코드) AS source
#     ON target.고객ID = source.고객ID AND target.티커종목코드 = source.티커종목코드
#     WHEN MATCHED THEN 
#         UPDATE SET 보유수량 = ?, 비중 = ?
#     WHEN NOT MATCHED THEN
#         INSERT (고객ID, 티커종목코드, 보유수량, 비중)
#         VALUES (?, ?, ?, ?);
#     ''', (group, ticker, shares, ratio, group, ticker, shares, ratio))

# conn.commit()
# conn.close()
# print("보유 주식 수량 계산 및 포트폴리오DB 업데이트 완료")


# ### data_summary
# # Azure SQL Database 연결 설정
# conn_str = (
#     'DRIVER={ODBC Driver 18 for SQL Server};'
#     'SERVER=jy-nh-server01.database.windows.net,1433;'
#     'DATABASE=NH-SQLDB-001;'
#     'UID=jyun22;'
#     'PWD=1emdrkwk!;'
#     'Encrypt=yes;'
#     'TrustServerCertificate=no;'
#     'Connection Timeout=100;'
# )
# conn = pyodbc.connect(conn_str)

# # 결과를 저장할 리스트 초기화
# documents = []


# # 지표 설명 리스트
# # 지표 설명
# documents = [
# '''
# 화제성 지수는 특정 종목의 거래량이 다른 종목과 비교했을 때 상대적으로 얼마나 두드러지게 변화하는지를 나타내는 지표입니다. 고객분류에서 화제성 지수를 활용하면, 다른 종목에 비해 특정 종목이 얼마나 주목받고 있는지를 평가하여 트렌드에 민감한 고객을 구분할 수 있습니다. 높은 지표 값은 고객이 주목도가 높은 주식을 선호하는 경향을 보여주고, 낮은 값은 주식에 대한 관심이 적은 고객으로 분류됩니다. ETF 추천에서는 이 지표를 통해 종목이 상대적으로 얼마나 주목받고 있는지를 파악할 수 있습니다.
# ''',
# '''
# 베타계수는 특정 종목의 변동성이 시장 전체와 얼마나 연관되어 있는지를 측정하는 지표입니다. 고객분류에서는 베타계수를 이용해 시장보다 변동성이 큰 주식을 선호하는 성향의 고객과, 안정적인 주식을 선호하는 고객을 구분할 수 있습니다. 베타계수가 상대적으로 높으면 더 큰 변동성을 기대하는 고객, 낮으면 안정성을 중시하는 고객으로 분류됩니다. ETF 추천에서는 베타계수는 시장 전체와 비교해 해당 종목이 상대적으로 위험하거나 안정적인지를 보여줍니다. 값이 크면 시장보다 더 큰 변동성을 보이고, 값이 작으면 상대적으로 안정적인 특성을 가집니다.
# ''',
# '''
# 수익률 표준편차는 종목의 가격 변동성을 측정하는 지표로, 고객분류에서 변동성이 큰 종목을 선호하는 고객은 높은 수익률 표준편차를 가진 종목을 더 많이 선택하는 경향이 있습니다. 안정적인 주식을 선호하는 고객은 표준편차가 낮은 종목을 선택합니다. ETF 추천에서는 수익률 표준편차가 높은 ETF는 변동성이 큰 종목들로 구성되어 있어, 위험을 감수하며 높은 수익을 기대하는 투자자에게 적합한 ETF를 추천하는 데 유용합니다.
# ''',
# ''';
# 투자심리지수는 사람들이 특정 주식이나 ETF에 대해 얼마나 긍정적인 감정을 가지고 있는지를 나타내는 지표입니다. 고객분류에서 투자심리지수가 높은 고객은 긍정적인 감정을 가진 종목을 더 많이 매수하며, 시장의 인기 종목에 집중하는 경향을 보입니다. ETF 추천에서는 투자심리지수가 높은 ETF가 긍정적인 평가를 받고 있는지 확인하며, 많은 투자자들이 긍정적으로 평가하는 ETF를 추천하는 데 도움이 됩니다.
# ''',
# '''
# 안정 기회 추구형 부기 (닌자거북이 스타일)
# 안정적인 시장에서 트렌드를 분석하여 위험을 최소화하고 꾸준한 수익을 목표로 안전한 종목에 투자하는 투자자 그룹
# 신중하고 안정적인 성향을 지닌 부기 투자자에게 예의바르고 정중한 어투로 추천을 시작합니다. 
# 설명: 닌자거북이 스타일의 부기들은 겉보기엔 느긋하고 신중해 보이지만, 내면적으로는 민첩하고 날카로운 판단력을 지닌 투자자들입니다. 거북이처럼 단단하고 안정적인 전략을 바탕으로 투자하면서도, 트렌드의 변화에 재빠르게 대응하는 특성이 있습니다. 시장이 안정적인 상황에서 일관된 수익을 위해 신중하게 움직이며, 필요할 때는 은밀하고 조용히 기회를 잡아냅니다.
# ''',
# '''
# 리스크 수익 탐색형 랑이 (동화 속 호랑이 스타일)
# 변동성이 큰 시장에서 트렌드에 구애받지 않고 고위험 종목에 과감히 투자해 높은 수익을 추구하는 투자자 그룹
# 자신의 기준을 확고히 하고 과감한 결정을 내리는 랑이 투자자에게  자신감 넘치고 결단력 있는 어투로 추천을 시작합니다.
# 설명: 동화 속 호랑이처럼 리스크 수익 탐색형 랑이들은 자신의 직관과 판단을 믿고, 트렌드에 흔들리지 않고 과감한 결정을 내리는 투자자들입니다. 이들은 변동성이 큰 시장에서 다른 이들의 판단이나 트렌드에 구애받지 않고 자신의 기준을 따르며, 필요할 때는 독자적인 길을 가는 것을 주저하지 않습니다.
# ''',
# '''
# 위기 속 안정형 아웅이 (사막여우 스타일)
# 불안정한 시장에서 트렌드를 분석하여 위험을 줄이고, 비교적 안정적인 종목에 투자하는 투자자 그룹
# 불확실한 상황속에서도 안정적으로 자산을 지키는 아웅이 투자자에게 연륜있으면서도 재미있는 어투로 추천을 시작합니다.
# 설명: 사막여우처럼 위기 속에서도 예리하게 트렌드를 분석하며 안정적인 선택을 하는 아웅이들은 불확실성이 높은 시장에서 자신의 생존 스킬을 발휘하는 투자자들입니다. 사막과 같은 열악한 환경에서도 생존할 수 있는 사막여우의 특징을 닮아, 위기 상황에서도 위험을 줄이고 안전하게 자산을 지켜가는 성향이 강합니다.
# ''',
# '''
# 성장 기회 탐색형 숭이 (원숭이 스타일)
# 안정적인 시장에서 트렌드를 반영하며 도전적인 종목에 투자해 높은 성장을 목표로 하는 투자자 그룹
# 새로운 기회를 탐색하고 도전적인 종목에 투자하는 숭이 투자자에게 활발하고 활기를 더해주는 어투로 추천을 시작합니다.
# 설명: 장난기 많고 호기심이 왕성한 원숭이 스타일의 숭이들은 안정적인 환경 속에서도 끊임없이 새로운 기회를 탐색하고 도전하는 투자자들입니다. 안정된 상황에서 다양한 트렌드를 살펴보며, 성장 가능성이 있는 종목을 발견하면 적극적으로 투자해 더 높은 성장을 추구합니다.
# '''
# ]


# # ETF 티커별로 상위 5개의 보유 종목을 추출하는 코드
# def summarize_top_5_stocks():
#     etf_구성종목정보 = pd.read_sql("SELECT * FROM ETF구성종목정보", conn)
#     etf_베타계수 = pd.read_sql("SELECT * FROM ETF베타계수", conn)
    
#     for etf_ticker, group in etf_구성종목정보[etf_구성종목정보['대상 ETF 티커'].isin(etf_베타계수['티커종목코드'])].groupby('대상 ETF 티커'):
#         top_5_stocks = group.nlargest(5, '보유 종목의 비중')
#         text = f"ETF {etf_ticker}의 구성 비중 상위 5개 종목:\n"
#         for _, row in top_5_stocks.iterrows():
#             stock_name_korean = row['fc_sec_krl_nm']
#             stock_weight = row['보유 종목의 비중']
#             text += f"- {stock_name_korean}: {stock_weight}%\n"
#         documents.append(text)

# # ETF 티커별로 정보를 요약하는 코드
# def summarize_etf_metrics():
#     etf_점수정보 = pd.read_sql("SELECT * FROM ETF점수정보", conn)
#     etf_베타계수 = pd.read_sql("SELECT * FROM ETF베타계수", conn)
    
#     for etf_ticker, group in etf_점수정보[etf_점수정보['etf_iem_cd'].isin(etf_베타계수['티커종목코드'])].groupby('etf_iem_cd'):
#         latest_entry = group.loc[:, ~group.columns.isin(['etf_iem_cd'])].mean()
        
#         # 필요한 정보 추출
#         etf_score = latest_entry['ETF점수']
#         z_score_rank = latest_entry['Z점수순위']
#         one_month_return = latest_entry['1개월총수익율']
#         three_month_return = latest_entry['3개월총수익율']
#         one_year_return = latest_entry['1년총수익율']
        
#         # Z점수 및 기타 지표 요약
#         cumulative_return_z = latest_entry['누적수익율Z점수']
#         info_ratio_z = latest_entry['정보비율Z점수']
#         sharpe_ratio_z = latest_entry['샤프지수Z점수']
#         correlation_z = latest_entry['상관관계Z점수']
#         tracking_error_z = latest_entry['트래킹에러Z점수']
#         max_drawdown_z = latest_entry['최대낙폭Z점수']
#         volatility_z = latest_entry['변동성Z점수']
        
#         # 텍스트 포맷팅
#         text = f"ETF {etf_ticker} 요약:\n"
#         text += f"- 1개월 총수익률: {one_month_return}%\n"
#         text += f"- 3개월 총수익률: {three_month_return}%\n"
#         text += f"- 1년 총수익률: {one_year_return}%\n"
#         text += f"- ETF 점수: {etf_score}\n"
#         text += f"- Z점수 순위: {z_score_rank}\n"
#         text += f"- 누적 수익률 Z점수: {cumulative_return_z}\n"
#         text += f"- 정보비율 Z점수: {info_ratio_z}\n"
#         text += f"- 샤프지수 Z점수: {sharpe_ratio_z}\n"
#         text += f"- 상관관계 Z점수: {correlation_z}\n"
#         text += f"- 트래킹 에러 Z점수: {tracking_error_z}\n"
#         text += f"- 최대 낙폭 Z점수: {max_drawdown_z}\n"
#         text += f"- 변동성 Z점수: {volatility_z}\n"
        
#         # 결과를 documents 리스트에 추가
#         documents.append(text)

# # ETF 티커별로 배당 정보를 요약하는 코드
# def summarize_etf_dividends():
#     etf_배당내역 = pd.read_sql("SELECT * FROM ETF배당내역", conn)
#     etf_베타계수 = pd.read_sql("SELECT * FROM ETF베타계수", conn)
    
#     for etf_ticker, group in etf_배당내역[etf_배당내역['대상 ETF 티커'].isin(etf_베타계수['티커종목코드'])].groupby('대상 ETF 티커'):
#         latest_entry = group[['배당금', '수정 배당금']].mean()
#         latest_entry['배당 주기'] = group['배당 주기'].mode()[0]
#         text = f"ETF {etf_ticker} 배당 요약:\n"
#         text += f"- 배당금: {latest_entry['배당금']}\n"
#         text += f"- 수정 배당금: {latest_entry['수정 배당금']}\n"
#         text += f"- 배당 주기: {latest_entry['배당 주기']}\n"
#         documents.append(text)

# # 요약 실행 및 결과 저장
# summarize_top_5_stocks()
# summarize_etf_metrics()
# summarize_etf_dividends()

# # JSON 파일에 저장 (ensure_ascii=False를 사용하여 한글을 그대로 저장)
# with open('documents.json', 'w', encoding='utf-8') as f:
#     json.dump(documents, f, ensure_ascii=False, indent=4)

# conn.close()
# print("ETF 데이터 요약 완료 및 JSON 저장")


# ### rag_inference
# # 문서 분할 및 임베딩 계산 함수
# def split_document(document, chunk_size=100):
#     sentences = document.split(". ")
#     chunks = []
#     chunk = []
#     for sentence in sentences:
#         chunk.append(sentence)
#         if len(chunk) >= chunk_size:
#             chunks.append(". ".join(chunk))
#             chunk = []
#     if chunk:
#         chunks.append(". ".join(chunk))
#     return chunks

# # 임베딩 모델 로드 및 FAISS 인덱스 생성
# def create_rag_index(documents):
#     model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    
#     # 문서를 분할하고, 임베딩 계산
#     chunks = []
#     for doc in documents:
#         chunks.extend(split_document(doc))
    
#     # 문서 임베딩 계산
#     document_embeddings = model.encode(chunks)
    
#     # FAISS 인덱스 생성 (코사인 유사도를 사용하기 위해 L2 정규화)
#     document_embeddings = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
#     embedding_dimension = document_embeddings.shape[1]
#     index = faiss.IndexFlatIP(embedding_dimension)  # 내적 기반 검색
#     index.add(np.array(document_embeddings))
    
#     return index

# # Azure Blob Storage URL에서 JSON 파일 다운로드 및 로드
# def load_documents_from_json_url(url):
#     response = requests.get(url)
#     response.raise_for_status()
    
#     # JSON 파일을 메모리 상에서 로드하여 파싱
#     documents = response.json()  # JSON 파일 내용을 파싱하여 Python 객체로 변환
#     return documents

# # 사용 예시 - documents URL을 Azure Blob의 JSON 파일 URL로 지정
# documents_url = "https://nh02.blob.core.windows.net/documents/documents.json"
# documents = load_documents_from_json_url(documents_url)

# create_rag_index(documents)



# ### customer_metrics
# # Azure SQL Database 연결
# conn_str = (
#     'DRIVER={ODBC Driver 18 for SQL Server};'
#     'SERVER=jy-nh-server01.database.windows.net,1433;'
#     'DATABASE=NH-SQLDB-001;'
#     'UID=jyun22;'
#     'PWD=1emdrkwk!;'
#     'Encrypt=yes;'
#     'TrustServerCertificate=no;'
#     'Connection Timeout=200;'
# )
# conn = pyodbc.connect(conn_str)

# # 최신 고객 지표 불러오기
# def fetch_all_latest_customer_metrics():
#     query = '''
#     SELECT 고객ID, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수
#     FROM 고객지표DB
#     WHERE 지표ID IN (
#         SELECT MAX(지표ID) FROM 고객지표DB GROUP BY 고객ID
#     )
#     '''
#     customer_metrics_df = pd.read_sql_query(query, conn)
#     return customer_metrics_df

# # 그룹 지표 데이터 불러오기 (group_metrics 테이블에서 가져오기)
# def fetch_group_metrics():
#     query = '''
#     SELECT 그룹명, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수
#     FROM group_metrics
#     '''
#     group_metrics = pd.read_sql_query(query, conn)
#     return group_metrics

# # 유사도 분석 함수
# def match_customers_to_group(customer_metrics_df, group_metrics_df):
#     metrics_columns = ['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']
#     scaler_customer = StandardScaler()
#     scaled_customer_metrics = scaler_customer.fit_transform(customer_metrics_df[metrics_columns])

#     scaler_group = StandardScaler()
#     scaled_group_metrics = scaler_group.fit_transform(group_metrics_df[metrics_columns])

#     results = []
#     for idx, customer_vector in enumerate(scaled_customer_metrics):
#         customer_vector = customer_vector.reshape(1, -1)  # 고객 벡터를 2차원으로 변환
#         similarities = cosine_similarity(customer_vector, scaled_group_metrics)  # 유사도 계산
#         most_similar_group_idx = np.argmax(similarities)  # 가장 유사한 그룹의 인덱스 찾기
#         most_similar_group = group_metrics_df.iloc[most_similar_group_idx]
#         similarity_score = similarities[0][most_similar_group_idx]
#         results.append({
#             '고객ID': customer_metrics_df.iloc[idx]['고객ID'],
#             '가장 유사한 그룹': most_similar_group['그룹명'],
#             '유사도 점수': similarity_score
#         })
    
#     return pd.DataFrame(results)

# # 실행

# # 1. 모든 고객의 최신 지표 불러오기
# customer_metrics_df = fetch_all_latest_customer_metrics()

# # 2. 그룹 지표 불러오기
# group_metrics = fetch_group_metrics()  # 이 부분에서 group_metrics를 정의합니다.

# # 3. 각 고객과 그룹을 비교하여 가장 유사한 그룹 찾기
# similarity_results_df = match_customers_to_group(customer_metrics_df, group_metrics)

# # 4. 결과 출력
# print(similarity_results_df)

# print(group_metrics)

# # DB 연결 해제
# conn.close()


# ### model_inference
# # Azure SQL Database 연결 설정
# conn_str = (
#     'DRIVER={ODBC Driver 18 for SQL Server};'
#     'SERVER=jy-nh-server01.database.windows.net,1433;'
#     'DATABASE=NH-SQLDB-001;'
#     'UID=jyun22;'
#     'PWD=1emdrkwk!;'
#     'Encrypt=yes;'
#     'TrustServerCertificate=no;'
#     'Connection Timeout=60;'
# )
# # 연결
# conn = pyodbc.connect(conn_str)


# # 고객 지표 계산 함수
# def calculate_customer_metrics(customer_id, portfolio_df, 주식별지표):
#     merged_df = pd.merge(주식별지표, portfolio_df, on='티커종목코드')
#     weights = merged_df['비중']
#     group_result = {'고객ID': customer_id}
#     group_result['베타계수'] = (merged_df['베타계수'] * weights).sum()
#     group_result['수익률표준편차'] = np.sqrt((merged_df['수익률표준편차']**2 * weights).sum())
#     group_result['트렌드지수'] = (merged_df['트렌드지수'] * weights).sum()
#     group_result['투자심리지수'] = (merged_df['투자심리지수'] * weights).sum()
#     group_result['최대보유섹터'] = merged_df.groupby('섹터분류명')['비중'].sum().sort_values(ascending=False).index[0]
#     return group_result

# # 지표 저장 함수
# def save_customer_metrics_to_db(customer_metrics):
#     conn = pyodbc.connect(conn_str)
#     cursor = conn.cursor()
#     cursor.execute('''
#     INSERT INTO 고객지표DB (고객ID, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수, 최대보유섹터)
#     VALUES (?, ?, ?, ?, ?, ?)
#     ''', (
#         customer_metrics['고객ID'],
#         customer_metrics['베타계수'],
#         customer_metrics['수익률표준편차'],
#         customer_metrics['트렌드지수'],
#         customer_metrics['투자심리지수'],
#         customer_metrics['최대보유섹터']
#     ))
#     conn.commit()
#     conn.close()

# ### query_data
# # Azure SQL Database 연결 설정
# conn_str = (
#     'DRIVER={ODBC Driver 18 for SQL Server};'
#     'SERVER=jy-nh-server01.database.windows.net,1433;'
#     'DATABASE=NH-SQLDB-001;'
#     'UID=jyun22;'
#     'PWD=1emdrkwk!;'
#     'Encrypt=yes;'
#     'TrustServerCertificate=no;'
#     'Connection Timeout=90;'
# )
# conn = pyodbc.connect(conn_str)

# # 고객 정보, 주식 포트폴리오, 고객 지표 JOIN하여 조회
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

