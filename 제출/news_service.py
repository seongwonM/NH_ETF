import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import pandas as pd
import random
import pandas as pd
import glob

'''
### 키워드 기반 뉴스 크롤링
### 키워드명 : SnP 500 (특수문자가 인식이 되지 않아 SnP 500으로 크롤링 후 이후 코드에서 다시 S&P 500으로 변경합니다.)
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import pandas as pd
import random

def generate_dates(start_date, end_date):
    """주어진 시작 날짜와 끝 날짜 사이의 모든 날짜를 생성합니다."""
    date_list = []
    current_date = start_date

    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y.%m.%d"))  # 형식에 맞춰 문자열로 변환
        current_date += timedelta(days=1)  # 하루씩 증가

    return date_list


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
                "keyword": query,  # 검색한 키워드 추가
                "news_index": idx + 1,
                "title": title,
                "content": content,
                "url": link
            })

    return news_data

def crawl_news(keywords, start_date, end_date, num_news_per_day=3):
    all_news = []
    news_counter = 1

    # 날짜 생성
    dates = generate_dates(start_date, end_date)

    for date in dates:
        for keyword in keywords:
            print(f"Crawling news for '{keyword}' on {date}")
            news = search_naver_news(keyword, date, num_news_per_day)

            for news_item in news:
                news_item["news_index"] = news_counter
                all_news.append(news_item)
                news_counter += 1

            time.sleep(random.randint(5, 10))  # 불규칙한 요청 간격 설정

    print(f"크롤링 완료! 총 {len(all_news)} 개의 뉴스 데이터가 수집되었습니다.")
    return all_news


# 입력 // '-'는 제외하고 키워드 기반 검색 진행
etf_keywords = ['SnP 500']
start_date = datetime.strptime("2024.05.28", "%Y.%m.%d")  # 시작 날짜
end_date = datetime.strptime("2024.08.26", "%Y.%m.%d")  # 끝 날짜
news_list = crawl_news(etf_keywords, start_date, end_date)

# 크롤링한 뉴스 리스트 출력 (필요한 경우)
for news in news_list:
    print(news)

SP = pd.DataFrame(news_list)
SP.to_csv('SP.csv', index=False)
print("CSV 파일 저장 완료!")
'''

'''
### SnP 500 을 제외한 다른 키워드 기반 뉴스 크롤링 코드
### 커널 종료 문제로 월별로 (5월,6월,7월,8월,8월2) 나눠서 크롤링을 진행 후 병합했습니다.
### 아래 코드를 날짜를 바꿔서 실행하시면 해당 날짜의 키워드 기반 뉴스가 크롤링됩니다.

import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import pandas as pd
import random


def generate_dates(start_date, end_date):
    """주어진 시작 날짜와 끝 날짜 사이의 모든 날짜를 생성합니다."""
    date_list = []
    current_date = start_date

    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y.%m.%d"))  # 형식에 맞춰 문자열로 변환
        current_date += timedelta(days=1)  # 하루씩 증가

    return date_list

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
                "keyword": query,  # 검색한 키워드 추가
                "news_index": idx + 1,
                "title": title,
                "content": content,
                "url": link
            })

    return news_data

def crawl_news(keywords, start_date, end_date, num_news_per_day=3):
    all_news = []
    news_counter = 1

    # 날짜 생성
    dates = generate_dates(start_date, end_date)

    for date in dates:
        for keyword in keywords:
            print(f"Crawling news for '{keyword}' on {date}")
            news = search_naver_news(keyword, date, num_news_per_day)

            for news_item in news:
                news_item["news_index"] = news_counter
                all_news.append(news_item)
                news_counter += 1

            time.sleep(random.randint(5, 10))  # 불규칙한 요청 간격 설정

    print(f"크롤링 완료! 총 {len(all_news)} 개의 뉴스 데이터가 수집되었습니다.")
    return all_news

# 입력
etf_keywords = ['기술 관련주', '에너지 관련주', '회사채', '아프리카', '미국 채권', '헬스케어 관련주', '미국 주식', '부동산 관련주', 'AMD', '경기 소비재 관련주', '임의 소비재 관련주', '아르헨티나 주식', '산업재 관련주', '중국 주식', '금융 서비스 관련주', '브라질', '기초 소재 관련주', '중국', '다우존스', 'DIS', '미국 국채', '통신 서비스 관련주', '호주 주식', '독일 주식', '일본 주식', '멕시코 주식', '한국 주식', '브라질 주식', '남아프리카 주식', '유럽 주식', '공공사업 관련주', ' 미국 주식', '인도 주식', '러셀2000', '필수 소비재 관련주', ' 사우디아라비아 주식', 'TIPS', 'MSCI USA 지수', 'MSCI World']
start_date = datetime.strptime("2024.05.28", "%Y.%m.%d")  # 시작 날짜
end_date = datetime.strptime("2024.05.31", "%Y.%m.%d")  # 끝 날짜
news_list = crawl_news(etf_keywords, start_date, end_date)

# 크롤링한 뉴스 리스트 출력 (필요한 경우)
for news in news_list:
    print(news)

df_5 = pd.DataFrame(news_list)
df_5.to_csv('5월.csv', index=False)
print("CSV 파일 저장 완료!")
'''

'''
뉴스 추천 서비스 구축
- 유사도 : 3일 동안 종가 변화 추이에 대한 유사도입니다.
'''

# SnP 500 크롤링된 파일 불러오기
SP = pd.read_csv('SP.csv')

# `keyword` 열에서 `SnP 500`을 `S&P 500`으로 변경
SP.loc[SP["keyword"] == "SnP 500", "keyword"] = "S&P 500"

# CSV 파일 경로 리스트
file_paths = ['5월.csv', '6월.csv', '7월.csv', '8월.csv', '8월2.csv']

# 빈 리스트 생성
dataframes = []

# 각 CSV 파일을 읽고 리스트에 추가
for file_path in file_paths:
    df = pd.read_csv(file_path)  # 파일 읽기
    dataframes.append(df)  # 리스트에 추가

# 모든 데이터프레임 병합
merged_df = pd.concat(dataframes, ignore_index=True)

# 'news_index' 열 제거, 'keyword' 열에서 S&P 500인 행 제거
merged_df = merged_df[merged_df["keyword"] != "S&P 500"]

# merged_df와 SP(s&p500) 두 데이터프레임 합치기
combined_df = pd.concat([merged_df, SP])
combined_df = combined_df.drop(columns=["news_index"],inplace=False)

# date 열로 정렬
combined_df["date"] = pd.to_datetime(combined_df["date"])  # 날짜 형식 변환
combined_df = combined_df.sort_values(by="date").reset_index(drop=True)
combined_df.to_csv('키워드 기반 뉴스 크롤링.csv', index=False)

news_service = pd.read_csv('ETF뉴스서비스.csv',encoding='cp949')
news_service['비교_기준일자'] = pd.to_datetime(news_service['비교_기준일자'], format='%Y%m%d')

# 'date'와 'keyword'를 '비교_기준일자'와 '섹터'로 병합
merged_with_title = pd.merge(
    news_service,
    combined_df[['date', 'keyword', 'title', 'content']],
    left_on=['비교_기준일자', '섹터'],
    right_on=['date', 'keyword'],
    how='left'
)

# 병합 후 열 이름 변경
merged_with_title.rename(columns={'title': '기사 제목', 'content': '기사 내용'}, inplace=True)

# 'date'와 'keyword' 열 삭제 (필요에 따라 삭제하지 않고 유지할 수도 있습니다)
merged_with_title.drop(columns=['date', 'keyword'], inplace=True)

# 결과 출력
merged_with_title.to_csv('뉴스 제공 서비스.csv',index=False, encoding='cp949')
print(merged_with_title)


# 아래의 주석은 openai 1.51.1에서만 실행이 가능합니다. azure가 호환되는 0.28 에서는 불가능합니다. 1.51.1로 아래의 주석을 풀고 실행하시면 뉴스 요약 결과를 확인하실 수 있습니다.
'''
import pandas as pd
from openai import OpenAI

client = OpenAI(
    api_key = ''
)

# 중요도 평가 함수
def evaluate_article_importance(title, content):
    prompt = (
        f"다음 기사를 읽고, 사람들이 주목할 만한 정도와 파격적인 요소를 고려하여 중요도를 1에서 10 사이로 평가해 주세요:\n\n"
        f"제목: {title}\n내용: {content}\n\n"
        "이 기사의 중요도 (1-10):"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=10
    )
    importance_score = response.choices[0].message.content.strip()
    try:
        return int(importance_score)
    except ValueError:
        return 5  # 오류 발생 시 기본값 반환

# 기사 요약 함수
def summarize_article(title, content):
    prompt = f"""
    당신은 금융 분야의 요약 전문가입니다.
    키워드와 관련 뉴스 본문을 읽고 전체 내용을 50단어 이내의 한국어로 요약해 주세요.
    요약은 객관적이고 중립적인 톤을 유지해 주세요.
    중복된 내용이 없어야 하고, 문맥이 어색하지 않게 요약해 주세요.
    전문 용어가 있다면 간단히 설명을 덧붙여 주세요.
    :\n제목: {title}\n내용: {content}
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# 요약 추가 함수
def add_summary_based_on_importance(df):
    summaries = []
    for (ticker, date, sector), group in df.groupby(["티커종목코드", "비교_기준일자", "섹터"]):
        group['중요도'] = group.apply(lambda row: evaluate_article_importance(row['기사 제목'], row['기사 내용']), axis=1)
        important_article = group.loc[group['중요도'].idxmax()]
        summary = summarize_article(important_article['기사 제목'], important_article['기사 내용'])
        summaries.extend([summary if idx == important_article.name else "" for idx in group.index])
    df["기사 요약"] = summaries
    return df

# 데이터프레임 초기화 후 실행
merged_with_title = add_summary_based_on_importance(merged_with_title)

# 결측치 제거
merged_with_title = merged_with_title.dropna(subset=['기사 제목']).reset_index(drop=True)

# '기사 요약' 열이 빈 문자열("")인 행을 삭제
merged_with_title = merged_with_title[merged_with_title['기사 요약'] != ""]

# 결과 확인
print(merged_with_title)
merged_with_title.to_csv('키워드뉴스_요약제공.csv', encoding='cp949')
'''
