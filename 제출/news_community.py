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
from transformers import pipeline
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
#### 네이버페이 해외증시뉴스 크롤링 ####

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import pandas as pd

# 특정 기간 동안의 날짜 목록 생성
start_date = datetime(2024, 5, 28)
end_date = datetime(2024, 8, 26)
date_list = [(start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range((end_date - start_date).days + 1)]

# 크롤링할 결과를 저장할 리스트
all_news_titles = []

# 각 날짜별로 10페이지까지 뉴스 제목을 크롤링
for date in date_list:
    for page in range(1, 11):  # 1페이지부터 10페이지까지
        url = f"https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=403&date={date}&page={page}"
        
        try:
            # 페이지 요청
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')

            # 뉴스 제목을 가져오기 (dt와 dd 태그 모두 포함)
            titles_dd = soup.select('#contentarea_left ul li dl dd a')
            titles_dt = soup.select('#contentarea_left ul li dl dt a')
        

            # 뉴스가 없는 페이지를 건너뛰기
            if not titles_dt and not titles_dd:
                print(f"No news found on {date}, page {page}. Skipping...")
                continue

            # 각 제목을 리스트에 추가
            for title in titles_dt + titles_dd:  # 두 리스트를 합쳐서 추가
                news_title = title.get_text().strip()
                if news_title:  # 제목이 비어있지 않은 경우에만 추가
                    all_news_titles.append((date, news_title))

            time.sleep(1)  # 요청 간에 약간의 지연 추가

        except requests.exceptions.HTTPError as e:
            print(f"Error fetching page {page} for date {date}: {e}")
            continue

# 수집된 뉴스 제목을 DataFrame으로 저장
news_df = pd.DataFrame(all_news_titles, columns=['Date', 'Title'])

# DataFrame을 출력
news_df

# CSV 파일로 저장
news_df.to_csv('뉴스크롤링.csv', index=False, encoding='cp949')
'''


# 댓글 데이터 크롤링 및 전처리 코드입니다.
# 실행 시점에 따라 총 크롤링되는 결과가 달라지고, 직접 삭제한 열이 몇군데 있어 주석처리하였습니다.  

'''

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import time as tm
import random
from user_agent import generate_user_agent, generate_navigator
import pandas as pd
from tqdm.notebook import tqdm
from selenium.webdriver.common.by import By
import os
from datetime import datetime
import re
import numpy as np
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import WebDriverException
import openai
import time
from concurrent.futures import ThreadPoolExecutor


# reddit 댓글 크롤링

options = webdriver.ChromeOptions()
options.add_argument('--incognito')

def lazy_scroll(driver):
    current_height = driver.execute_script('return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight );')
    while True:
        driver.execute_script('window.scrollTo(0,document.body.scrollHeight);')
        tm.sleep(3)
        new_height = driver.execute_script('return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight );')
        if new_height == current_height:
            html = driver.page_source
            break
        current_height = new_height
    return html

def reddit_scrape(subreddit, driver):
    url = subreddit
    try:
        driver.get(url)
    except WebDriverException:
        tm.sleep(10)
        driver.quit()
        driver = webdriver.Chrome()
        driver.get(url)

    tm.sleep(5)
    try:
        driver.maximize_window()
    except WebDriverException as e:
        print("WebDriverException occurred while maximizing window:", e)

    tm.sleep(5)
    lazy_scroll(driver)
    tm.sleep(5)
    post_links = driver.find_elements(By.TAG_NAME, 'shreddit-post')

    print('PostLinks: ' + str(post_links))

    post_data = []
    if len(post_links) > 0:
        for post in post_links:
            href_attribute = post.get_attribute("permalink")
            print(f"Element Href: {href_attribute}")
            post_data.append({'Permalink': href_attribute})

        df = pd.concat([pd.DataFrame(post_data)], ignore_index=True)
        
        result_df = pd.DataFrame(columns=['post_detail', 'platform', 'genre', 'post_like', 'post_created_time', 'post_source'])

        postNum = 1

        for post in df['Permalink']:
            print("Post: " + str(postNum))
            postNum += 1
            url = 'https://old.reddit.com' + post + '?sort=confidence'
            print(url)
            try:
                driver.get(url)
            except WebDriverException:
                tm.sleep(10)
                driver.quit()
                driver = webdriver.Chrome()
                driver.get(url)

            tm.sleep(3)
            lazy_scroll(driver)
            tm.sleep(2)
            parent = driver.find_elements(By.CLASS_NAME, 'comment')

            # 댓글 개수를 세기 위한 변수
            comment_count = 0

            for element in parent:
                if comment_count >= 100:
                    break

                comment = element.find_element(By.CLASS_NAME, 'tagline')
                try:
                    votescore = comment.find_element(By.CLASS_NAME, 'unvoted')
                    score = votescore.get_attribute('title')
                    print(score)
                except:
                    score = 0

                time = comment.find_element(By.TAG_NAME, 'time')
                orgTime = time.get_attribute('datetime')
                original_time = datetime.strptime(orgTime, "%Y-%m-%dT%H:%M:%S%z")
                formatted_time_str = original_time.strftime("%Y-%m-%d %H:%M:%S")

                commentBox = element.find_element(By.CLASS_NAME, 'usertext-body')
                commentText = commentBox.find_elements(By.TAG_NAME, 'p')
                resultText = ''

                for text in commentText:
                    resultText = resultText + ' ' + text.text
                print(resultText)

                # Set values directly using loc accessor
                result_df.loc[len(result_df)] = {
                    'post_detail': resultText,
                    'platform': 'Reddit',
                    'post_like': score,
                    'post_created_time': formatted_time_str,
                    'post_source': url
                }

                # 댓글 수집 개수 증가
                comment_count += 1

        # 댓글이 없을 경우 None 값 추가
        if len(result_df) == 0:
            result_df.loc[0] = {
                'post_detail': None,
                'platform': 'Reddit',
                'post_like': None,
                'post_created_time': None,
                'post_source': url
            }

        result_df['post_detail'].replace('', np.nan, inplace=True)
        result_df['post_detail'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
        result_df = result_df.dropna(subset=['post_detail'])

        return result_df
    else:
        # 게시물이 없을 경우 빈 데이터프레임 반환
        return pd.DataFrame(columns=['post_detail', 'platform', 'post_like', 'post_created_time', 'post_source'])

subs = pd.read_excel('./SubredditList.xlsx')
subreddits = subs['url']
final_df = pd.DataFrame(columns=['post_detail', 'platform', 'post_like', 'post_created_time', 'post_source'])
for sub in subreddits:
    driver = webdriver.Chrome()
    subreddit_url = f'{sub}'
    print(subreddit_url)
    subreddit_df = reddit_scrape(subreddit_url, driver)
    final_df = pd.concat([final_df, subreddit_df], ignore_index=True)
    driver.quit()

final_df.to_csv('./RedditCrawlerResults.csv', index=False)



# 댓글 데이터 정리
# 정규식을 써도 지워지지 않아서 깨진 이모티콘으로 이루어진 열만 지웠습니다. 
df = pd.read_csv(".\RedditCrawlerResults_정리본.csv")

df = df.drop(columns=['genre'])
df = df.iloc[:-1]
df_filter = df.copy()

# 정규 표현식을 이용해 원하는 부분 추출
df_filter['post_source'] = df_filter['post_source'].str.extract(r'/([^/]+)/[^/]+/?$', expand=False)

# NaN 값이 있는지 확인 후 문자열이 아닌 행 제거
df_filter = df_filter[df_filter['post_source'].notna()]

# 문자열로 변환 (숫자 등 다른 타입의 데이터가 있는 경우를 대비)
df_filter['post_source'] = df_filter['post_source'].astype(str)

# 월_일_년도 형식의 패턴
pattern = r'.*_(may|june|july|august)_[0-9]{1,2}_[0-9]{4}$'

# 패턴에 맞지 않는 게시글 필터링
non_matching_posts = df_filter[~df_filter['post_source'].str.lower().str.match(pattern)]

# 월_일_년도 형식의 패턴
pattern = r'.*_(may|june|july|august)_[0-9]{1,2}_[0-9]{4}$'

# 패턴에 맞는 행만 남기기
df_filter = df_filter[df_filter['post_source'].str.lower().str.match(pattern, na=False)]

# 월을 숫자로 매핑하는 딕셔너리
month_map = {
    'may': '05', 'june': '06', 'july': '07', 'august': '08'
}

# 정규 표현식을 이용해 월, 일, 년도 부분 추출
df_filter[['month', 'day', 'year']] = df_filter['post_source'].str.extract(r'_(may|june|july|august)_(\d{1,2})_(\d{4})$', expand=True)

# 월을 숫자로 변환
df_filter['month'] = df_filter['month'].map(month_map)

# 일자를 2자리 숫자로 변환
df_filter['day'] = df_filter['day'].str.zfill(2)

# 년도, 월, 일 합쳐서 YYYYMMDD 형식 만들기
df_filter['post_date'] = df_filter['year'] + df_filter['month'] + df_filter['day']

# 불필요한 중간 열 삭제
df_filter = df_filter.drop(columns=['month', 'day', 'year'])

# 정규 표현식을 사용하여 "월_일_년도" 부분 제거
df_filter['post_source'] = df_filter['post_source'].str.replace(r'_(may|june|july|august)_[0-9]{1,2}_[0-9]{4}$', '', regex=True)

# "daily_discussion_thread_for"를 "daily_discussion_thread"로 변경
df_filter['post_source'] = df_filter['post_source'].replace('daily_discussion_thread_for', 'daily_discussion_thread')

# "weekend_discussion_thread_for" 행 제거
df_filter = df_filter[df_filter['post_source'] != 'weekend_discussion_thread_for']

# 인덱스를 재정렬하여 0부터 순차적으로 설정
df_filter = df_filter.reset_index(drop=True)

# post_source 열의 이름을 post_title로 변경
df_filter = df_filter.rename(columns={'post_source': 'post_title'})

# 열 순서를 변경하고자 하는 순서대로 지정
new_column_order = ['post_date', 'post_title', 'post_detail', 'post_like', 'post_created_time', 'platform']

# 열 순서 변경
df_filter = df_filter[new_column_order]

# post_detail 열의 이름을 post_comment로 변경
df_filter = df_filter.rename(columns={'post_detail': 'post_comment'})

# df_filter를 CSV 파일로 저장
df_filter.to_csv("웰스트리트베츠_댓글데이터.csv", index=False, encoding='utf-8-sig')

comment = pd.read_csv('/content/drive/MyDrive/NH/웰스트리트베츠_댓글데이터.csv')

# post_created_time을 datetime 형태로 변환
comment['post_created_time'] = pd.to_datetime(comment['post_created_time'])

# 주 단위로 구분 (주 번호를 새로운 열로 추가)
comment['week'] = comment['post_created_time'].dt.isocalendar().week

# 주제별 데이터 분리
daily_discussion = comment[comment['post_title'] == 'daily_discussion_thread']
moves_tomorrow = comment[comment['post_title'] == 'what_are_your_moves_tomorrow']


# 새로운 데이퍼 프레임으로 할당
Daily = daily_discussion.copy()
tomorrow = moves_tomorrow.copy()


# OpenAI API 키 설정
openai.api_key = "개인 openai api_key 입력"


########### 댓글에서 종목추출
# 1. '티커종목코드'를 리스트로 추출하고 대문자로 변환
ticker_codes = 주식별고객정보['티커종목코드'].dropna().unique().tolist()
ticker_codes_upper = [code.upper() for code in ticker_codes]

# 2. 종목 코드 패턴 생성 (단어 경계를 사용하여 정확히 일치하는 코드만 매칭)
# 예: r'\b(AAPL|GOOGL|MSFT|AMZN|TSLA)\b'
pattern = r'\b(' + '|'.join(re.escape(code) for code in ticker_codes_upper) + r')\b'

# 3. 정규표현식을 미리 컴파일하여 성능 최적화
ticker_regex = re.compile(pattern)

# 4. 종목 추출 함수 정의
def extract_stock_symbol(comment):
    # 댓글이 문자열인지 확인
    if not isinstance(comment, str):
        return None
    # 정규표현식을 사용하여 매칭되는 종목 코드 찾기
    match = ticker_regex.search(comment)
    if match:
        return match.group(1)  # 매칭된 종목 코드 반환
    else:
        return None  # 매칭되지 않으면 None 반환

# 5. '종목' 열 추가
Daily['종목'] = Daily['post_comment'].apply(extract_stock_symbol)

# 6. '종목' 열에 NaN이 있는 경우 제거
Daily = Daily.dropna(subset=['종목'])


#############대표댓글 추출
# 대표 댓글 추출 함수
def get_representative_comment(comments):
    prompt = f"""
    다음은 여러 댓글입니다. 이 중에서 가장 대표적인 의견을 반영하는 하나의 댓글을 선택하세요:
    {comments}

    결과 형식:
    대표 댓글: [선택한 대표 댓글]
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for selecting representative comments."},
            {"role": "user", "content": prompt}
        ]
    )

    representative_comment = response['choices'][0]['message']['content'].strip()
    return representative_comment.replace("대표 댓글:", "").strip()

##############감성분석
# 감성 분석 함수
def analyze_sentiment(comment):
    prompt = f"""
    다음은 주식 및 ETF와 관련된 커뮤니티 댓글입니다. 이 댓글에서 감정을 분석하세요.
    주식 커뮤니티에서는 긍정, 부정, 중립의 감정이 다음과 같은 의미로 표현될 수 있습니다:

    - 긍정: 종목에 대해 높은 기대를 나타내거나 긍정적인 평가
    - 부정: 종목에 대한 우려나 실망, 부정적인 평가
    - 중립: 특별한 감정 없이 정보 공유나 단순 의견 제시

    댓글 내용: '{comment}'

    **결과는 반드시 "긍정", "부정", "중립" 중 하나만 단어로 응답하세요.**
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        sentiment = response['choices'][0]['message']['content'].strip()

        if re.fullmatch(r'긍정|부정|중립', sentiment):
            return sentiment
        else:
            return "중립"

    except Exception as e:
        print(f"감성 분석 중 오류 발생: {e}")
        return "중립"
    

# 결과
# 데이터프레임 처리 함수
def process_group(date, stock, comments):
    representative_comment = get_representative_comment(comments)
    sentiment = analyze_sentiment(representative_comment)
    return {
        '날짜': date,
        '종목': stock,
        '대표_댓글': representative_comment,
        '감정_분석': sentiment
    }

# 병렬 처리를 통해 데이터프레임 처리
grouped_data = []
with ThreadPoolExecutor() as executor:
    futures = []
    for (date, stock), group in Daily.groupby(['post_date', '종목']):
        comments = "\n".join(group['post_comment'].tolist())
        futures.append(executor.submit(process_group, date, stock, comments))

    for future in futures:
        grouped_data.append(future.result())

# 최종 데이터프레임 생성
final_Daily_df = pd.DataFrame(grouped_data)


# 내일 동향데이터
# 1. '티커종목코드'를 리스트로 추출하고 대문자로 변환
ticker_codes = 주식별고객정보['티커종목코드'].dropna().unique().tolist()
ticker_codes_upper = [code.upper() for code in ticker_codes]

# 2. 종목 코드 패턴 생성 (단어 경계를 사용하여 정확히 일치하는 코드만 매칭)
# 예: r'\b(AAPL|GOOGL|MSFT|AMZN|TSLA)\b'
pattern = r'\b(' + '|'.join(re.escape(code) for code in ticker_codes_upper) + r')\b'

# 3. 정규표현식을 미리 컴파일하여 성능 최적화
ticker_regex = re.compile(pattern)

# 4. 종목 추출 함수 정의
def extract_stock_symbol(comment):
    # 댓글이 문자열인지 확인
    if not isinstance(comment, str):
        return None
    # 정규표현식을 사용하여 매칭되는 종목 코드 찾기
    match = ticker_regex.search(comment)
    if match:
        return match.group(1)  # 매칭된 종목 코드 반환
    else:
        return None  # 매칭되지 않으면 None 반환

# 5. '종목' 열 추가
tomorrow['종목'] = tomorrow['post_comment'].apply(extract_stock_symbol)

# 6. '종목' 열에 NaN이 있는 경우 제거
tomorrow = tomorrow.dropna(subset=['종목'])

#######대표댓글 추출
# 대표 댓글 추출 함수
def get_representative_comment(comments):
    prompt = f"""
    다음은 여러 댓글입니다. 이 중에서 가장 대표적인 의견을 반영하는 하나의 댓글을 선택하세요:
    {comments}

    결과 형식:
    대표 댓글: [선택한 대표 댓글]
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for selecting representative comments."},
            {"role": "user", "content": prompt}
        ]
    )

    representative_comment = response['choices'][0]['message']['content'].strip()
    return representative_comment.replace("대표 댓글:", "").strip()

###감성분석
# 감성 분석 함수
def analyze_sentiment(comment):
    prompt = f"""
    다음은 주식 및 ETF와 관련된 커뮤니티 댓글입니다. 이 댓글에서 감정을 분석하세요.
    주식 커뮤니티에서는 긍정, 부정, 중립의 감정이 다음과 같은 의미로 표현될 수 있습니다:

    - 긍정: 종목에 대해 높은 기대를 나타내거나 긍정적인 평가
    - 부정: 종목에 대한 우려나 실망, 부정적인 평가
    - 중립: 특별한 감정 없이 정보 공유나 단순 의견 제시

    댓글 내용: '{comment}'

    **결과는 반드시 "긍정", "부정", "중립" 중 하나만 단어로 응답하세요.**
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        sentiment = response['choices'][0]['message']['content'].strip()

        if re.fullmatch(r'긍정|부정|중립', sentiment):
            return sentiment
        else:
            return "중립"

    except Exception as e:
        print(f"감성 분석 중 오류 발생: {e}")
        return "중립"

### 결과
# 데이터프레임 처리 함수
def process_group(date, stock, comments):
    representative_comment = get_representative_comment(comments)
    sentiment = analyze_sentiment(representative_comment)
    return {
        '날짜': date,
        '종목': stock,
        '대표_댓글': representative_comment,
        '감정_분석': sentiment
    }

# 병렬 처리를 통해 데이터프레임 처리
grouped_data = []
with ThreadPoolExecutor() as executor:
    futures = []
    for (date, stock), group in tomorrow.groupby(['post_date', '종목']):
        comments = "\n".join(group['post_comment'].tolist())
        futures.append(executor.submit(process_group, date, stock, comments))

    for future in futures:
        grouped_data.append(future.result())

# 최종 데이터프레임 생성
final_tomorrow_df = pd.DataFrame(grouped_data)
final_tomorrow_df

# 이름 변경 후 새로운 데이터 프레임으로 형성
final_daily_discussion_thread = final_Daily_df.copy()
final_what_are_your_moves_tomorrow = final_tomorrow_df.copy()

# 열 이름 변경 코드
final_daily_discussion_thread.rename(columns={
    '날짜': '기준일자',
    '종목': '티커종목코드',
    '대표_댓글': '대표_댓글',
    '감정_분석': '대표_댓글_감정'
}, inplace=True)

final_what_are_your_moves_tomorrow.rename(columns={
    '날짜': '기준일자',
    '종목': '티커종목코드',
    '대표_댓글': '대표_댓글',
    '감정_분석': '대표_댓글_감정'
}, inplace=True)
final_what_are_your_moves_tomorrow

# 파일로 저장
daily.to_csv('daily.csv', index=False, encoding='utf-8-sig')
print(" 'daily.csv' 파일로 저장되었습니다.")

final_what_are_your_moves_tomorrow.to_csv('tomorrow.csv', index=False, encoding='utf-8-sig')
print(" 'tomorrow.csv' 파일로 저장되었습니다.")

'''

주식별고객정보 = pd.read_csv('주식별고객정보.csv', encoding='cp949')
뉴스크롤링 = pd.read_csv('뉴스크롤링.csv', encoding='cp949')

종목코드 = 주식별고객정보['티커종목코드'].drop_duplicates().reset_index(drop=True)
종목코드 = 종목코드.to_frame(name='티커종목코드')

# 두 데이터프레임을 티커 종목 코드를 기준으로 결합
# left_on: 주식 일별 정보 데이터프레임의 티커 열 이름
# right_on: 해외 종목 코드 데이터프레임의 티커 열 이름
merged_df = 종목코드.merge(
    해외종목정보,
    left_on='티커종목코드',
    right_on='티커종목코드',
    how='inner'  # 겹치는 부분만 결합
)

# 필요한 열만 선택 (티커종목코드, 외화증권한글명, 외화증권영문명)
티커종목정보 = merged_df[['티커종목코드', '외화증권한글명', '외화증권영문명']]

# 기업명 리스트 생성 (한글명과 영문명 포함)
기업명_리스트 = 티커종목정보[['외화증권한글명', '외화증권영문명']].values.tolist()

# 회사 언급 횟수를 추출하는 함수 정의
def extract_company_mentions(news, company_names):
    # 결과를 저장할 리스트를 초기화
    records = []
    
    # 각 뉴스 제목을 순회
    for idx, row in news.iterrows():
        date = row['Date']    # 뉴스 날짜
        title = row['Title']  # 뉴스 제목
        mentioned_companies = []  # 언급된 회사를 저장할 리스트
        
        # 각 회사가 제목에 언급되었는지 확인
        for company_kr, company_en in company_names:
            if company_kr in title or company_en in title:
                mentioned_companies.append(company_kr)  # 일관성을 위해 한국어 이름 사용
        
        # 적어도 하나의 회사가 언급된 경우에만 기록 추가
        if mentioned_companies:
            records.append({
                'Date': date,
                'Title': title,
                'Mentioned_Companies': ', '.join(mentioned_companies)  # 언급된 회사들을 쉼표로 구분하여 저장
            })
    
    # 기록 리스트를 데이터프레임으로 변환
    result_df = pd.DataFrame(records)
    return result_df


# 함수를 실행하여 결과 도출
news_ticker = extract_company_mentions(뉴스크롤링, 기업명_리스트)

뉴스크롤링 = pd.read_csv('뉴스크롤링.csv', encoding='cp949')

# 회사 언급 횟수를 추출하는 함수 정의
def extract_company_mentions(news, company_names):
    # 결과를 저장할 리스트를 초기화
    records = []
    
    # 각 뉴스 제목을 순회
    for idx, row in news.iterrows():
        date = row['Date']    # 뉴스 날짜
        title = row['Title']  # 뉴스 제목
        mentioned_companies = []  # 언급된 회사를 저장할 리스트
        
        # 각 회사가 제목에 언급되었는지 확인
        for company_kr, company_en in company_names:
            if company_kr in title or company_en in title:
                mentioned_companies.append(company_kr)  # 일관성을 위해 한국어 이름 사용
        
        # 적어도 하나의 회사가 언급된 경우에만 기록 추가
        if mentioned_companies:
            records.append({
                'Date': date,
                'Title': title,
                'Mentioned_Companies': ', '.join(mentioned_companies)  # 언급된 회사들을 쉼표로 구분하여 저장
            })
    
    # 기록 리스트를 데이터프레임으로 변환
    result_df = pd.DataFrame(records)
    return result_df


# 함수를 실행하여 결과 도출
news_ticker = extract_company_mentions(뉴스크롤링, 기업명_리스트)

# 결과 출력
print(news_ticker)

#### 해외증시뉴스 감성분석

news = news_ticker.copy()

kor_clf_sentiment = pipeline("sentiment-analysis", "snunlp/KR-FinBert-SC")

news['label'] = news['Title'].apply(lambda x: kor_clf_sentiment(x)[0]['label'])

# 라벨 -> 숫자
news['label'].replace({'positive':1, 'neutral':0,'negative':-1 }, inplace=True)

# 'Date' 열을 datetime 형식으로 변환
news['Date'] = pd.to_datetime(news['Date'], format='%Y%m%d')

# Mentioned_Companies 열을 분리하여 행 확장
news = news.assign(Mentioned_Companies=news['Mentioned_Companies'].str.split(', ')).explode('Mentioned_Companies')

# 각 날짜와 기업별 등장 횟수, 긍정 뉴스 개수, 중립 뉴스 개수, 부정 뉴스 개수 계산
result = news.groupby(['Date', 'Mentioned_Companies']).agg(
    count=('Mentioned_Companies', 'size'),                 # 등장 횟수
    positive_num=('label', lambda x: (x == 1).sum()),          # label이 1인 긍정 뉴스 개수
    neutral_num=('label', lambda x: (x == 0).sum()),          # label이 0인 중립 뉴스 개수
    negative_num=('label', lambda x: (x == -1).sum())          # label이 -1인 부정 뉴스 개수
).reset_index()

# 종목명 티커코드로 변환
# 해외종목정보에서 '티커종목코드', '외화증권한글명', '외화증권영문명' 정보 추출
ticker_mapping = 해외종목정보[['티커종목코드', '외화증권한글명', '외화증권영문명']]

# ticker_mapping을 이용하여 Mentioned_Companies 열을 티커종목코드로 변환
news_data = result.merge(ticker_mapping[['티커종목코드', '외화증권한글명']], left_on='Mentioned_Companies', right_on='외화증권한글명', how='left')

# 불필요한 '티커종목코드'와 '외화증권한글명' 열 삭제하고 이름을 '티커종목코드'로 변경
news_data.drop(columns=['Mentioned_Companies', '외화증권한글명'], inplace=True)
news_data.rename(columns={'티커종목코드': 'Mentioned_Companies'}, inplace=True)

# 새로운 열 순서 정의 (날짜 다음에 Mentioned_Companies 위치)
new_order = ['Date', 'Mentioned_Companies', 'count','positive_num','neutral_num','negative_num']

# 데이터프레임 재구성
news_data = news_data[new_order]

# 열 이름 한글로변경
news_data = news_data.rename(columns={
    'Date': '기준일자',
    'Mentioned_Companies': '티커종목코드',
    'count':'등장횟수',
    'positive_num':'긍정',
    'neutral_num':'중립',
    'negative_num':'부정'
})

# 결과 확인
news_data

news_data.to_csv('final_news_data.csv',encoding='cp949',index=False)

#### 뉴스(해외증시뉴스)와 댓글(해외커뮤니티 사이트 reddit_월스트리츠베츠) 추이 분석
news = pd.read_csv("final_news_data.csv", encoding='cp949')
today_mention = pd.read_csv("final_daily_discussion_thread.csv")
tomorrow_mention = pd.read_csv("final_what_are_your_moves_tomorrow.csv")

# 원본 데이터 프레임 유지
news_df = news.copy()
today = today_mention.copy()
tomorrow = tomorrow_mention.copy()

# 데이터 타입을 정리하고 날짜 형식 변환
news_df['기준일자'] = pd.to_datetime(news_df['기준일자'])
today['기준일자'] = pd.to_datetime(today['기준일자'], format='%Y%m%d')
tomorrow['기준일자'] = pd.to_datetime(tomorrow['기준일자'], format='%Y%m%d')

'''
"오늘의 해외증시뉴스" 와 "Reddit플랫폼의 월스트리츠비츠 커뮤니티오늘의 투자 리포트 게시물의 댓글" 데이터 사이의 관계 파악
- 같은 날짜의 해당 종목의 해외증시뉴스의 제목 감성분석 결과와, 커뮤니티 댓글 감정 결과가 일관된 경향을 보이는 것을 알 수 있습니다.
- 이를 통해 투자 커뮤니티(투자자들의 심리)는 실제 해당 종목의 동향(뉴스 데이터)과 연관이 있음을 확인했습니다.
'''

merged_df = pd.merge(news_df, today, on=['기준일자', '티커종목코드'])

# 감정 변화 패턴 분석을 위해 긍정, 중립, 부정의 비율 계산
merged_df['긍정비율'] = merged_df['긍정'] / merged_df['등장횟수']
merged_df['중립비율'] = merged_df['중립'] / merged_df['등장횟수']
merged_df['부정비율'] = merged_df['부정'] / merged_df['등장횟수']

# 감정 변화 패턴 추출
def detect_sentiment_pattern(row):
    if row['대표_댓글_감정'] == '긍정' and row['부정비율'] > 0.5:
        return '긍정 -> 부정 가능성'
    elif row['대표_댓글_감정'] == '부정' and row['긍정비율'] > 0.5:
        return '부정 -> 긍정 가능성'
    else:
        return '일관된 감정'

# 감정 변화 패턴 컬럼 생성
merged_df['감정변화패턴'] = merged_df.apply(detect_sentiment_pattern, axis=1)

# 티커종목코드와 기준일자 순으로 정렬
merged_df = merged_df.sort_values(by=['티커종목코드', '기준일자']).reset_index(drop=True)

# '일관된 감정' 비율 계산
def calculate_consistent_sentiment_ratio(df):
    # 전체 행 개수와 '일관된 감정'의 개수를 계산
    total_count = len(df)
    consistent_count = len(df[df['감정변화패턴'] == '일관된 감정'])
    # 비율 계산
    return consistent_count / total_count * 100

# 티커별로 비율 계산
consistent_sentiment_ratio = (
    merged_df.groupby('티커종목코드')
    .apply(calculate_consistent_sentiment_ratio)
    .reset_index(name='일관된 감정 비율 (%)')
)

# 결과 출력
print("티커별 일관된 감정 비율:")
print(consistent_sentiment_ratio)
