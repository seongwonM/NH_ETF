# Airflow 이미지를 기반으로 빌드
FROM apache/airflow:2.10.3

# 필요한 패키지 설치 및 pyodbc 설치
USER root
RUN apt-get update && \
    apt-get install -y curl apt-transport-https gnupg && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/12/prod.list -o /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# airflow 사용자로 전환한 후에 pip로 pyodbc 설치
USER airflow
RUN pip install pyodbc
