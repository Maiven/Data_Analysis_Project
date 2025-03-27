[airbnb_large_sample.csv](https://github.com/user-attachments/files/19481573/airbnb_large_sample.csv)좋습니다. 첨부해주신 이미지를 기반으로, 아래와 같이 **Airbnb - 데이터 분석 & 인사이트 도출 프로젝트**를 완성형 실무 프로젝트 포맷으로 정리해드리겠습니다. 실제 현업에서도 활용할 수 있는 방식으로 구성해드릴게요.


---

# 📊 AirBnB - 데이터 분석 & 인사이트 도출 프로젝트

### ✅ 프로젝트 개요

- **목적**: Airbnb 숙소 가격에 영향을 주는 주요 요인 도출 및 세분화된 지표(Metrics) 기반 인사이트 추출
- **핵심 질문**:
  1. 어떤 변수들이 Airbnb 가격에 영향을 주는가?
  2. 지역·숙소 유형·리뷰 수·가용 일수 등에 따라 세분화된 패턴은 존재하는가?
  3. 데이터 전반의 분포와 특성을 기반으로 어떤 비즈니스 인사이트를 도출할 수 있는가?

---

### 🔎 문제정의

- Airbnb 가격에 영향을 주는 주요 요인을 도출하고자 함
- Seg별 주요 Metrics를 추출하여 유의미한 인사이트 도출
- 즉, 보유 중인 Airbnb 데이터를 전반적으로 활용하여 주요 지표를 파악하고 데이터의 상태와 트렌드를 이해하는 것이 목표

---

### 🛠️ 프로젝트 배경 및 필요성

- 현업에서는 분석 목적이 모호할 수 있지만, 데이터를 **전반적으로 이해하는 일**이 반드시 필요함
- 이는 데이터 기반 의사결정의 출발점으로, 데이터 특성에 대한 직관적 이해 없이는 비즈니스 전략 수립이 어렵기 때문
- 본 프로젝트는 다음과 같은 목표를 갖고 진행됨:
  - 전체 Airbnb 데이터를 구조화하고 시각화함으로써, 인사이트 및 패턴 파악
  - 분석 결과를 바탕으로 실질적인 운영 전략 수립이 가능하도록 지원

---

### 🧩 분석 절차

이미지 속 흐름도를 기반으로 다음과 같은 분석 단계를 설계할 수 있습니다:

#### 1. **Airbnb 데이터 로드**
- CSV 파일 로드 및 스키마 확인
- 결측치, 이상치, 데이터 타입 확인
- 예: `price`, `number_of_reviews`, `availability_365`, `neighbourhood`, `room_type`

#### 2. **데이터 정리**
- 전처리: 가격 단위 통일 (`$` 제거, 정수형 변환), 날짜 데이터 파싱
- 파생 변수 생성: 예) `price_per_person`, `review_rate_per_month` 등
- 범주형 변수 정리: 지역, 숙소유형 등 Label Encoding 또는 One-hot Encoding 적용

#### 3. **가격 분포 계산**
- 전체 가격 분포 및 로그변환 확인 (Outlier 확인 목적)
- 지역, 숙소 유형별 평균 및 중앙값 계산
- Boxplot / KDE Plot / Histogram 사용

#### 4. **가격 영향 요인 식별**
- 상관관계 분석 (피어슨/스피어만)
- 회귀 분석: 다중 선형 회귀로 주요 Feature 도출
- 랜덤포레스트 등 트리 기반 모델로 Feature Importance 시각화

#### 5. **가격 분포 시각화**
- 지역별 가격 히트맵 (지도 기반 시각화)
- 카테고리별 Price 분포 비교 (Violin Plot, Stripplot)
- 숙소 유형/지역/가용일수/리뷰 수에 따른 분포 비교

#### 6. **패턴 분석**
- 클러스터링 기반 Seg 도출 (K-means, DBSCAN)
- 고객 리뷰 수와 가격의 관계 분석
- 시즌성 분석 (월별, 요일별)

#### 7. **비즈니스 인사이트 도출**
- 고가격 숙소의 공통 특성은 무엇인가?
- 지역/숙소 유형별 최적 가격대는 어떻게 구성되어야 하는가?
- 가격 정책 및 마케팅 전략에 반영할 수 있는 핵심 패턴 정리

---

### 📌 기대 효과

| 구분 | 기대 결과 |
|------|-----------|
| 데이터 정제 | 이상치 제거 및 정규화된 Airbnb 데이터셋 확보 |
| 메트릭 도출 | 평균가격, 가용률, 리뷰 수 등 핵심 지표 추출 |
| 인사이트 | 숙소 가격 결정 요인 및 고가 숙소의 특성 이해 |
| 시각화 | 다양한 차트와 지도 기반 시각화로 설득력 강화 |
| 의사결정 지원 | 지역별 가격 정책 설계 및 운영 전략 도출 가능 |

---

### 💬 개인적 인사이트 및 제언

Airbnb는 단순 숙박 플랫폼이 아닌, 도시 경제·관광 트렌드와 연결된 **복합 생태계 데이터**입니다. 이 프로젝트는 ‘숙소 가격’이라는 경제적 신호를 통해 **거시적 도시 변화, 미시적 소비자 선호**를 동시에 들여다볼 수 있는 기회를 제공합니다.

이 데이터를 통해 단지 통계분석에 머무르는 것이 아니라, **현장 운영 전략에 바로 적용 가능한 통찰**로 연결시키는 것이야말로 데이터 분석가의 실무적 역량이라 생각합니다.

---

필요하시면 이 분석을 Python + Pandas + Seaborn + Scikit-learn으로 코드화하거나, Tableau로 시각화 설계해드릴 수 있습니다. 어느 쪽으로 확장하시겠어요?




---

```python
# Airbnb 데이터 분석 & 인사이트 도출 프로젝트

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 로드
# 파일 경로를 사용자가 지정하도록 수정 가능
data = pd.read_csv("airbnb_data.csv")

# 2. 데이터 정리
data = data.copy()

# price에서 $ 제거 후 숫자형으로 변환
data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)

# 불필요한 컬럼 제거 (예시)
data.drop(['id', 'name', 'host_name', 'last_review'], axis=1, inplace=True, errors='ignore')

# 결측치 처리
data.fillna({
    'reviews_per_month': 0,
    'review_rate_number': 0
}, inplace=True)

# 범주형 인코딩
categorical_cols = ['neighbourhood_group', 'neighbourhood', 'room_type']
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# 3. 가격 분포 시각화
plt.figure(figsize=(10,6))
sns.histplot(data['price'], bins=100, kde=True)
plt.title('Airbnb 가격 분포')
plt.xlim(0, 500)
plt.xlabel('가격($)')
plt.show()

# 로그 변환 분포 보기 (이상치 완화)
data['log_price'] = np.log1p(data['price'])
plt.figure(figsize=(10,6))
sns.histplot(data['log_price'], bins=100, kde=True)
plt.title('Log-변환 Airbnb 가격 분포')
plt.xlabel('log(price + 1)')
plt.show()

# 4. 상관관계 분석
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('변수 간 상관관계')
plt.show()

# 5. Feature Importance - Random Forest
features = data.drop(columns=['price', 'log_price'])
target = data['price']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 변수 중요도 시각화
feature_importance = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('Feature Importance')
plt.xlabel('중요도')
plt.ylabel('변수')
plt.show()

# 6. 지역별 평균 가격 시각화
if 'neighbourhood' in data.columns:
    avg_price_by_area = data.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=avg_price_by_area.values, y=avg_price_by_area.index)
    plt.title('지역별 평균 숙소 가격(상위 10개)')
    plt.xlabel('평균 가격($)')
    plt.ylabel('지역')
    plt.show()
```


코드화를 완료했습니다. 이 프로젝트는 다음을 수행합니다:

- Airbnb 가격 데이터 정제 및 전처리
- 가격 분포 및 로그 분포 시각화
- 상관관계 분석과 랜덤포레스트 기반 주요 영향 변수 도출
- 지역별 평균 가격 분석

💡 다음 단계로 아래 중 무엇을 확장하시겠어요?

1. **클러스터링 기반 Segmentation (KMeans)** 추가
2. **월별 또는 요일별 시즌성 분석** 추가
3. **Tableau용 분석 리포트** 또는 **Power BI 연결용 데이터 가공**

원하시는 방향 알려주시면 이어서 도와드리겠습니다!



---




```python
# Airbnb 데이터 분석 & 인사이트 도출 프로젝트

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import datetime

# 1. 데이터 로드
# 파일 경로를 사용자가 지정하도록 수정 가능
data = pd.read_csv("airbnb_data.csv")

# 2. 데이터 정리
data = data.copy()

data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)
data.drop(['id', 'name', 'host_name', 'last_review'], axis=1, inplace=True, errors='ignore')
data.fillna({
    'reviews_per_month': 0,
    'review_rate_number': 0
}, inplace=True)

categorical_cols = ['neighbourhood_group', 'neighbourhood', 'room_type']
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# 날짜 정보 처리
data['date'] = pd.to_datetime(data.get('date', pd.Timestamp.today()))
data['month'] = data['date'].dt.month
if 'date' in data.columns:
    data['weekday'] = data['date'].dt.day_name()

# 3. 가격 분포 시각화
plt.figure(figsize=(10,6))
sns.histplot(data['price'], bins=100, kde=True)
plt.title('Airbnb 가격 분포')
plt.xlim(0, 500)
plt.xlabel('가격($)')
plt.show()

# 로그 변환
data['log_price'] = np.log1p(data['price'])
plt.figure(figsize=(10,6))
sns.histplot(data['log_price'], bins=100, kde=True)
plt.title('Log-변환 Airbnb 가격 분포')
plt.xlabel('log(price + 1)')
plt.show()

# 4. 상관관계 분석
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('변수 간 상관관계')
plt.show()

# 5. Feature Importance
features = data.drop(columns=['price', 'log_price'])
target = data['price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
feature_importance = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('Feature Importance')
plt.xlabel('중요도')
plt.ylabel('변수')
plt.show()

# 6. 지역별 평균 가격 시각화
if 'neighbourhood' in data.columns:
    avg_price_by_area = data.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=avg_price_by_area.values, y=avg_price_by_area.index)
    plt.title('지역별 평균 숙소 가격(상위 10개)')
    plt.xlabel('평균 가격($)')
    plt.ylabel('지역')
    plt.show()

# 7. 클러스터링 기반 세분화
cluster_data = data[['price', 'minimum_nights', 'number_of_reviews', 'availability_365']].copy()
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(cluster_scaled)

plt.figure(figsize=(10,6))
sns.scatterplot(x='price', y='availability_365', hue='cluster', data=data, palette='Set2')
plt.title('Airbnb 가격 vs 가용일수 - 클러스터별 시각화')
plt.xlabel('가격($)')
plt.ylabel('가용일수')
plt.legend(title='Cluster')
plt.show()

# 8. 월별 시즌성 분석
if 'month' in data.columns:
    plt.figure(figsize=(10,6))
    sns.boxplot(x='month', y='price', data=data)
    plt.title('월별 Airbnb 가격 분포')
    plt.xlabel('월')
    plt.ylabel('가격($)')
    plt.show()

# 9. 요일별 시즌성 분석
if 'weekday' in data.columns:
    plt.figure(figsize=(10,6))
    sns.boxplot(x='weekday', y='price', data=data, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title('요일별 Airbnb 가격 분포')
    plt.xlabel('요일')
    plt.ylabel('가격($)')
    plt.xticks(rotation=45)
    plt.show()
```

✅ 클러스터링 기반 Segmentation(KMeans)과 **월별/요일별 시즌성 분석**을 코드에 추가 완료했습니다. 주요 기능은 아래와 같습니다:

1. **KMeans 클러스터링**:
   - `price`, `minimum_nights`, `number_of_reviews`, `availability_365` 기준으로 숙소를 4개 그룹으로 세분화
   - 가격과 가용일수 간의 클러스터 시각화

2. **시즌성 분석**:
   - `month`를 기준으로 **월별 가격 분포** 확인
   - `weekday`를 기준으로 **요일별 가격 트렌드** 시각화

📌 이제 이 분석은 Airbnb 가격 전략, 성수기 예측, 그리고 데이터 기반의 숙소 세분화 전략까지 가능하게 해줍니다.

원하시면 다음 단계로:
- **Tableau 시각화용 Aggregated Dataset 생성**
- **리포트 자동 생성 (PDF/HTML)**
- **Streamlit 웹앱으로 대시보드화**

이 중 어떤 쪽으로 확장하고 싶으신가요?


---


```python
# Airbnb 데이터 분석 & 인사이트 도출 프로젝트

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import datetime
import streamlit as st

st.set_page_config(page_title="Airbnb 데이터 분석 대시보드", layout="wide")
st.title("🏠 Airbnb 데이터 분석 & 인사이트 대시보드")

# 데이터 로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # 데이터 정리
    data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)
    data.drop(['id', 'name', 'host_name', 'last_review'], axis=1, inplace=True, errors='ignore')
    data.fillna({
        'reviews_per_month': 0,
        'review_rate_number': 0
    }, inplace=True)

    categorical_cols = ['neighbourhood_group', 'neighbourhood', 'room_type']
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    data['date'] = pd.to_datetime(data.get('date', pd.Timestamp.today()))
    data['month'] = data['date'].dt.month
    data['weekday'] = data['date'].dt.day_name()
    data['log_price'] = np.log1p(data['price'])

    # 클러스터링
    cluster_data = data[['price', 'minimum_nights', 'number_of_reviews', 'availability_365']].copy()
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['cluster'] = kmeans.fit_predict(cluster_scaled)

    tab1, tab2, tab3 = st.tabs(["📊 가격 분석", "📍 클러스터링", "📆 시즌성 분석"])

    with tab1:
        st.subheader("가격 분포 시각화")
        fig, ax = plt.subplots()
        sns.histplot(data['price'], bins=100, kde=True, ax=ax)
        ax.set_xlim(0, 500)
        st.pyplot(fig)

        st.subheader("Log-변환 가격 분포")
        fig2, ax2 = plt.subplots()
        sns.histplot(data['log_price'], bins=100, kde=True, ax=ax2)
        st.pyplot(fig2)

        st.subheader("Feature Importance")
        features = data.drop(columns=['price', 'log_price'])
        target = data['price']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        feature_importance = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
        fig3, ax3 = plt.subplots()
        sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax3)
        st.pyplot(fig3)

    with tab2:
        st.subheader("클러스터링 결과")
        fig4, ax4 = plt.subplots()
        sns.scatterplot(x='price', y='availability_365', hue='cluster', data=data, palette='Set2', ax=ax4)
        st.pyplot(fig4)

    with tab3:
        st.subheader("월별 가격 분포")
        fig5, ax5 = plt.subplots()
        sns.boxplot(x='month', y='price', data=data, ax=ax5)
        st.pyplot(fig5)

        st.subheader("요일별 가격 분포")
        fig6, ax6 = plt.subplots()
        sns.boxplot(x='weekday', y='price', data=data, 
                    order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ax=ax6)
        plt.xticks(rotation=45)
        st.pyplot(fig6)
else:
    st.info("왼쪽 사이드바에서 Airbnb CSV 파일을 업로드해주세요.")
```

Streamlit 웹앱 코드로 변환이 완료되었습니다! 🎉  
이제 다음과 같은 기능이 포함된 대시보드를 실행할 수 있습니다:

### 📊 탭별 기능 구성

1. **📊 가격 분석 탭**
   - 가격 히스토그램 & KDE 플롯
   - Log(price+1) 변환 분포
   - 랜덤포레스트 기반 Feature Importance 시각화

2. **📍 클러스터링 탭**
   - 가격 vs 가용일수 기반 클러스터 결과 시각화

3. **📆 시즌성 분석 탭**
   - 월별 가격 Boxplot
   - 요일별 가격 Boxplot (요일 순 정렬 포함)

---

### ▶ 실행 방법

1. `airbnb_dashboard.py` 등의 이름으로 저장
2. 터미널에서 아래 명령어 실행:

```bash
streamlit run airbnb_dashboard.py
```

3. 웹 브라우저에서 대시보드 확인

---

필요하시면:
- **지역별 지도 시각화 (folium, pydeck 등)**
- **클러스터별 특성 테이블 요약**
- **다운로드 기능(데이터/차트)**

도 추가해드릴 수 있습니다. 어떤 확장을 원하시나요?




