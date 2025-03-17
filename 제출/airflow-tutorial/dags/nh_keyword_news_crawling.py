from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pyodbc
import random
import time

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

# 키워드 목록
etf_keywords = [
    '기술 관련주', '에너지 관련주', '회사채', '아프리카', '미국 채권', '헬스케어 관련주', '미국 주식', '부동산 관련주', 
    'AMD', '경기 소비재 관련주', '임의 소비재 관련주', '아르헨티나 주식', '산업재 관련주', '중국 주식', 
    '금융 서비스 관련주', '브라질', 'SnP 500', '기초 소재 관련주', '중국', '다우존스', 'DIS', '미국 국채', 
    '통신 서비스 관련주', '호주 주식', '독일 주식', '일본 주식', '멕시코 주식', '한국 주식', '브라질 주식', 
    '남아프리카 주식', '유럽 주식', '공공사업 관련주', '미국 주식', '인도 주식', '러셀2000', '필수 소비재 관련주', 
    '사우디아라비아 주식', 'TIPS', 'MSCI USA 지수', 'MSCI World'
]

def search_naver_news(query, date, num_news):
    url = f"https://search.naver.com/search.naver?where=news&query={query}&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds={date}&de={date}&docid=&related=0&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from{date.replace('.', '')}to{date.replace('.', '')},a:all&start=1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {query} on {date}: {e}")
        return []  # 오류 발생 시 빈 리스트 반환

    soup = BeautifulSoup(response.text, 'html.parser')
    news_data = []
    articles = soup.select("div.group_news ul.list_news > li")

    if not articles:
        print(f"No news found for {query} on {date}")
        return []

    for idx, article in enumerate(articles[:num_news]):
        title_tag = article.select_one("a.news_tit")
        content_tag = article.select_one("div.news_dsc > div > a")

        if title_tag and content_tag:
            title = title_tag.get_text()
            content = content_tag.get_text()
            link = title_tag["href"]

            news_data.append({
                "date": date,
                "keyword": query,
                "news_index": idx + 1,
                "title": title,
                "content": content,
                "url": link
            })

    return news_data

def save_to_azure_sql(news_data):
    """크롤링한 뉴스 데이터를 Azure SQL Database의 keyword_news 테이블에 저장합니다."""
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()

            # 데이터 삽입
            insert_query = """
            INSERT INTO keyword_news (date, keyword, news_index, title, content, url)
            VALUES (?, ?, ?, ?, ?, ?);
            """
            for news_item in news_data:
                cursor.execute(insert_query, news_item["date"], news_item["keyword"], news_item["news_index"],
                               news_item["title"], news_item["content"], news_item["url"])
            conn.commit()
            print("Azure SQL Database의 keyword_news 테이블에 데이터 저장 완료!")
    except Exception as e:
        print(f"Azure SQL Database에 데이터 저장 실패: {e}")

def crawl_news():
    all_news = []
    today_date = datetime.now().strftime("%Y.%m.%d")  # 오늘 날짜

    for keyword in etf_keywords:
        print(f"Crawling news for '{keyword}' on {today_date}")
        news = search_naver_news(keyword, today_date, 3)

        if news:
            all_news.extend(news)
        time.sleep(random.randint(5, 10))  # 불규칙한 요청 간격 설정

    if all_news:
        save_to_azure_sql(all_news)
    else:
        print("수집된 뉴스 데이터가 없습니다.")

# Airflow DAG 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'keyword_news',  # DAG 이름을 keyword_news로 설정
    default_args=default_args,
    description='네이버 뉴스 키워드 크롤링 DAG',
    schedule_interval='0 6 * * *',  # 매일 오전 6시에 실행
    start_date=datetime(2024, 11, 15),
    catchup=False,
    tags=['news', 'crawling'],
) as dag:
    
    crawl_task = PythonOperator(
        task_id='crawl_news_task',
        python_callable=crawl_news
    )

crawl_task
