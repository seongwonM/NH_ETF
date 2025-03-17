from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import pyodbc

# Azure SQL Database 연결 설정
conn_str = (
    'DRIVER={ODBC Driver 18 for SQL Server};'
    'SERVER=jy-nh-server01.database.windows.net,1433;'
    'DATABASE=NH-SQLDB-001;'
    'UID=jyun22;'
    'PWD=1emdrkwk!;'
    'Encrypt=yes;'
    'TrustServerCertificate=no;'
    'Connection Timeout=100;'
)

# 크롤링 함수 정의
def fetch_news_data():
    # 당일 날짜로 크롤링
    date = datetime.now().strftime('%Y%m%d')
    all_news_titles = []

    for page in range(1, 11):  # 1페이지부터 10페이지까지
        url = f"https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=403&date={date}&page={page}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')

            titles_dd = soup.select('#contentarea_left ul li dl dd a')
            titles_dt = soup.select('#contentarea_left ul li dl dt a')

            if not titles_dt and not titles_dd:
                continue

            for title in titles_dt + titles_dd:
                news_title = title.get_text().strip()
                if news_title:
                    all_news_titles.append((date, news_title))

            time.sleep(1)
        except requests.exceptions.HTTPError as e:
            print(f"Error fetching page {page} for date {date}: {e}")
            continue

    # Azure SQL Database에 데이터 저장
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # 데이터 삽입
        for news_data in all_news_titles:
            cursor.execute("INSERT INTO NewsData (Date, Title) VALUES (?, ?)", news_data[0], news_data[1])
        
        conn.commit()
    except Exception as e:
        print(f"Error saving to database: {e}")
    finally:
        # cursor와 conn이 정의된 경우에만 close 호출
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# DAG 기본 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
with DAG(
    'daily_news_crawler',
    default_args=default_args,
    description='A daily task to crawl overseas stock news and save to Azure SQL Database',
    schedule_interval='@daily',  # 매일 실행
    start_date=datetime(2024, 5, 28),
    catchup=False,
) as dag:
    
    # 뉴스 크롤링 작업
    news_crawl_task = PythonOperator(
        task_id='fetch_news_data',
        python_callable=fetch_news_data
    )

    news_crawl_task
