from pathlib import Path
import glob
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="FMCG 판매 데이터 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: #F5F7FB; }
    section[data-testid="stSidebar"] { background: #101C35; }
    section[data-testid="stSidebar"] * { color: #F4F7FC !important; }
    .main-title { font-size: 2rem; font-weight: 800; color: #14213D; margin-bottom: 0.15rem; }
    .sub-title { color: #64748B; font-size: 0.95rem; margin-bottom: 1.3rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #E5EAF2; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 2px 8px rgba(20,33,61,.04);
    }
    div[data-testid="stMetricLabel"] { color: #64748B; }
    div[data-testid="stMetricValue"] { color: #14213D; font-weight: 750; }
    .section-label { color: #64748B; font-size: .8rem; font-weight: 700; letter-spacing: .08em; }
    .insight {
        background: white; border-left: 4px solid #3156A5; padding: 14px 18px;
        border-radius: 8px; color: #334155; margin: 8px 0 18px 0;
    }
    h1, h2, h3 { color: #14213D; }
    [data-testid="stDataFrame"] { background: white; }
</style>
""", unsafe_allow_html=True)

DATA_CANDIDATES = [
    "FMCG_2022_2024.csv",
    "FMCG_2022_2024_preprocessed.csv",
    "FMCG_2022_2024_preprocessed(1).csv",
    "FMCG_2022_2024_preprocessed_2.csv",
]

@st.cache_data(show_spinner="CSV 데이터를 불러오는 중...")
def load_data():
    paths = []
    for name in DATA_CANDIDATES:
        paths.extend(glob.glob(str(Path(__file__).resolve().parent / name)))
    if not paths:
        # 파일명이 조금 달라도 FMCG 데이터 CSV를 자동 탐색
        paths = glob.glob(str(Path(__file__).resolve().parent / "FMCG*.csv"))
    if not paths:
        raise FileNotFoundError(
            "앱 파일과 같은 폴더에서 FMCG_2022_2024.csv 또는 FMCG로 시작하는 CSV 파일을 찾지 못했습니다."
        )
    path = paths[0]
    try:
        data = pd.read_csv(path)
    except UnicodeDecodeError:
        data = pd.read_csv(path, encoding="cp949")
    return data, Path(path).name
def prepare_data(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = ["date", "units_sold"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = ["price_unit", "promotion_flag", "delivery_days",
                    "stock_available", "delivered_qty", "units_sold"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "units_sold"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"] = df["date"].dt.day_name()
    return df

def fmt_num(value):
    return f"{value:,.0f}" if pd.notna(value) else "-"

def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)

def section_title(title, desc=None):
    st.subheader(title)
    if desc:
        st.caption(desc)

def available(cols, df):
    return [c for c in cols if c in df.columns]

# ---------------------------------------------------------
# 데이터 로드 및 전처리 상태
# ---------------------------------------------------------
try:
    raw_df, source_name = load_data()
    if "working_df" not in st.session_state:
        st.session_state.working_df = prepare_data(raw_df)
    df = st.session_state.working_df.copy()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.info("app.py와 CSV 파일을 같은 폴더에 넣은 뒤 다시 실행하세요.")
    st.stop()

if df.empty:
    st.error("분석할 수 있는 데이터가 없습니다.")
    st.stop()

# ---------------------------------------------------------
# 사이드바
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 FMCG ANALYTICS")
    st.caption("판매 데이터 분석 · 수요 예측 · 재고 관리")
    st.divider()
    page = st.radio(
        "분석 메뉴",
        [
            "메인 대시보드",
            "탐구 질문 분석",
            "데이터 탐색",
            "머신러닝 예측",
            "재고 관리",
            "데이터 전처리",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.caption(f"데이터 파일: {source_name}")
    st.caption(f"분석 데이터: {len(df):,}행")
# ---------------------------------------------------------
# 1. 메인 대시보드
# ---------------------------------------------------------
if page == "메인 대시보드":
    st.markdown('<div class="main-title">FMCG 판매 데이터 분석 대시보드</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">2022–2024 소비재 판매 데이터를 바탕으로 판매 현황과 수요 특성을 파악합니다.</div>', unsafe_allow_html=True)

    total_sales = df["units_sold"].sum()
    avg_sales = df["units_sold"].mean()
    sku_count = df["sku"].nunique() if "sku" in df else 0
    date_min, date_max = df["date"].min(), df["date"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("누적 판매량", fmt_num(total_sales))
    c2.metric("레코드당 평균 판매량", f"{avg_sales:,.2f}")
    c3.metric("분석 상품 수(SKU)", f"{sku_count:,}" if sku_count else "-")
    c4.metric("분석 데이터 수", f"{len(df):,}")

    st.caption(f"분석 기간: {date_min:%Y-%m-%d} ~ {date_max:%Y-%m-%d}")
    st.divider()

    left, right = st.columns([1.55, 1])
    with left:
        section_title("날짜별 판매량 추이", "일별 판매량을 합산해 전체 수요의 흐름을 확인합니다.")
        daily = df.groupby("date", as_index=False)["units_sold"].sum()
        daily["이동평균(7일)"] = daily["units_sold"].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["units_sold"], mode="lines",
                                 name="일별 판매량", line=dict(color="#9DB3D8", width=1)))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["이동평균(7일)"], mode="lines",
                                 name="7일 이동평균", line=dict(color="#234A91", width=3)))
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if "category" in df:
            section_title("카테고리별 누적 판매량")
            cat = df.groupby("category", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)
            fig = px.pie(cat, names="category", values="units_sold", hole=.58,
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(height=360, margin=dict(l=5,r=5,t=20,b=5),
                              paper_bgcolor="white", legend_title="")
            st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        if "channel" in df:
            section_title("판매 채널별 판매량")
            channel = df.groupby("channel", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)
            fig = px.bar(channel, x="channel", y="units_sold", text_auto=".3s",
                         color="units_sold", color_continuous_scale="Blues")
            fig.update_layout(height=320, coloraxis_showscale=False, xaxis_title="", yaxis_title="판매량",
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
    with b:
        if "region" in df:
            section_title("지역별 판매량")
            region = df.groupby("region", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)
            fig = px.bar(region, x="region", y="units_sold", text_auto=".3s",
                         color="units_sold", color_continuous_scale="Blues")
            fig.update_layout(height=320, coloraxis_showscale=False, xaxis_title="", yaxis_title="판매량",
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    section_title("핵심 데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    insight("대시보드의 판매량은 개별 거래·관측 행의 units_sold를 합산한 값입니다. 상품·채널·지역별 차이는 뒤의 탐구 질문 분석에서 구체적으로 확인합니다.")

# ---------------------------------------------------------
# 2. 탐구 질문 분석
# ---------------------------------------------------------
elif page == "탐구 질문 분석":
    st.markdown('<div class="main-title">탐구 질문 분석</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">다섯 가지 질문을 통해 판매량에 영향을 줄 수 있는 요인과 데이터의 패턴을 살펴봅니다.</div>', unsafe_allow_html=True)

    if "category" in df:
        section_title("질문 1. 상품 종류에 따라 평균 판매량은 차이가 있을까?")
        q1 = df.groupby("category", as_index=False)["units_sold"].mean().sort_values("units_sold", ascending=False)
        fig = px.bar(q1, x="category", y="units_sold", text_auto=".2f",
                     color="units_sold", color_continuous_scale="Blues")
        fig.update_layout(height=330, coloraxis_showscale=False, xaxis_title="상품 종류", yaxis_title="평균 판매량",
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        if len(q1):
            top = q1.iloc[0]
            low = q1.iloc[-1]
            insight(f"평균 판매량이 가장 높은 상품 종류는 {top['category']}({top['units_sold']:.2f})이며, 가장 낮은 종류는 {low['category']}({low['units_sold']:.2f})입니다. 평균의 차이는 관찰된 연관성이며 원인으로 단정할 수 없습니다.")

    if "region" in df and "channel" in df:
        section_title("질문 2. 지역과 판매 채널에 따라 판매량은 어떻게 달라질까?")
        q2 = df.pivot_table(index="region", columns="channel", values="units_sold", aggfunc="mean")
        fig = px.imshow(q2, text_auto=".2f", aspect="auto", color_continuous_scale="Blues",
                        labels=dict(x="판매 채널", y="지역", color="평균 판매량"))
        fig.update_layout(height=330, paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("색이 진할수록 해당 지역·채널 조합의 평균 판매량이 높습니다.")
    if "promotion_flag" in df:
        section_title("질문 3. 프로모션 여부에 따라 평균 판매량은 달라질까?")
        q3 = df.copy()
        q3["프로모션"] = q3["promotion_flag"].map({0: "프로모션 없음", 1: "프로모션 진행"})
        q3 = q3.groupby("프로모션", as_index=False).agg(
            **{"평균 판매량": ("units_sold", "mean"), "관측 수": ("units_sold", "size")}
        )
        fig = px.bar(q3, x="프로모션", y="평균 판매량", text_auto=".2f",
                     color="프로모션", color_discrete_sequence=["#A9BBD8", "#3156A5"])
        fig.update_layout(height=310, showlegend=False, xaxis_title="", yaxis_title="평균 판매량",
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(q3, use_container_width=True, hide_index=True)
        insight("프로모션 그룹 간 평균 판매량 차이는 가격, 상품 구성, 시기 등 다른 요인의 영향을 함께 받을 수 있으므로 프로모션만의 순수한 효과로 해석하지 않습니다.")

    numeric_rel = available(["price_unit", "stock_available", "delivered_qty", "delivery_days", "units_sold"], df)
    if len(numeric_rel) >= 2:
        section_title("질문 4. 가격·재고량·입고량·배송 소요일은 판매량과 어떤 관계가 있을까?")
        corr = df[numeric_rel].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu",
                        color_continuous_midpoint=0, aspect="auto")
        fig.update_layout(height=390, paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("상관계수는 두 변수의 선형적 관계를 나타냅니다. 상관관계만으로 인과관계를 판단할 수 없습니다.")

    section_title("질문 5. 날짜에 따라 판매량의 추세나 반복되는 패턴이 나타날까?")
    monthly = df.set_index("date").resample("MS")["units_sold"].sum().reset_index()
    fig = px.line(monthly, x="date", y="units_sold", markers=True,
                  labels={"date":"월", "units_sold":"월간 판매량"})
    fig.update_traces(line_color="#3156A5", line_width=3)
    fig.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    month_avg = df.groupby("month", as_index=False)["units_sold"].mean()
    fig = px.bar(month_avg, x="month", y="units_sold", text_auto=".2f",
                 labels={"month":"월", "units_sold":"평균 판매량"})
    fig.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    insight("월별 합계는 전체 데이터의 규모와 기간별 추세를, 월 번호별 평균은 달력상 월에 따른 반복 패턴 가능성을 보여줍니다. 연도별 구성 차이도 함께 고려해야 합니다.")

# ---------------------------------------------------------
# 3. 데이터 탐색
# ---------------------------------------------------------
elif page == "데이터 탐색":
    st.markdown('<div class="main-title">데이터 탐색</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">변수의 분포와 기초 통계량을 확인하고 분석에 사용할 데이터의 특성을 파악합니다.</div>', unsafe_allow_html=True)

    nrows, ncols = df.shape
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("데이터 행 수", f"{nrows:,}")
    c2.metric("변수 수", f"{ncols:,}")
    c3.metric("중복 행", f"{df.duplicated().sum():,}")
    c4.metric("결측 셀", f"{int(df.isna().sum().sum()):,}")

    section_title("변수별 자료형 및 결측치")
    overview = pd.DataFrame({
        "변수": df.columns,
        "자료형": [str(df[c].dtype) for c in df.columns],
        "결측치 수": [int(df[c].isna().sum()) for c in df.columns],
        "고유값 수": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(overview, use_container_width=True, hide_index=True)
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    if numeric:
        section_title("수치형 변수의 기초 통계량")
        st.dataframe(df[numeric].describe().T, use_container_width=True)

        selected = st.selectbox("분포를 확인할 수치형 변수", numeric,
                                index=numeric.index("units_sold") if "units_sold" in numeric else 0)
        fig = px.histogram(df, x=selected, nbins=40, marginal="box",
                           color_discrete_sequence=["#3156A5"])
        fig.update_layout(height=370, plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title=selected, yaxis_title="빈도")
        st.plotly_chart(fig, use_container_width=True)

    categorical = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    if categorical:
        section_title("범주형 변수의 구성")
        selected_cat = st.selectbox("분포를 확인할 범주형 변수", categorical)
        counts = df[selected_cat].fillna("(결측)").value_counts().head(20).reset_index()
        counts.columns = [selected_cat, "관측 수"]
        fig = px.bar(counts, x=selected_cat, y="관측 수", text_auto=".3s",
                     color="관측 수", color_continuous_scale="Blues")
        fig.update_layout(height=350, coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 4. 머신러닝 예측
# ---------------------------------------------------------
elif page == "머신러닝 예측":
    st.markdown('<div class="main-title">머신러닝 판매량 예측</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">과거 데이터를 학습하고 이후 시점의 판매량을 예측합니다. 시간 순서를 지켜 학습·평가 데이터를 분리합니다.</div>', unsafe_allow_html=True)

    target = "units_sold"
    default_features = [c for c in ["sku","category","channel","region","price_unit","promotion_flag","delivery_days","month","day_of_week"] if c in df.columns]
    all_features = [c for c in df.columns if c not in [target, "date", "day_name"]]
    features = st.multiselect("예측에 사용할 독립변수", all_features, default=default_features)
    model_name = st.selectbox("회귀 모델", ["다중 선형 회귀", "릿지 회귀"])
    test_ratio = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, 0.05)

    if st.button("모델 학습 및 성능 평가", type="primary", use_container_width=True):
        if not features:
            st.error("독립변수를 하나 이상 선택하세요.")
        else:
            work = df.sort_values("date").dropna(subset=[target]).copy()
            X = work[features].copy()
            y = work[target].astype(float)
            split = int(len(work) * (1 - test_ratio))
            if split < 10 or len(work) - split < 2:
                st.error("학습 또는 테스트 데이터가 너무 적습니다.")
            else:
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]
                cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
                num_cols = [c for c in features if c not in cat_cols]
                transformers = []
                if num_cols:
                    transformers.append(("num", Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ]), num_cols))
                if cat_cols:
                    transformers.append(("cat", Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ]), cat_cols))
                preprocessor = ColumnTransformer(transformers)
                regressor = LinearRegression() if model_name == "다중 선형 회귀" else Ridge(alpha=1.0)
                pipe = Pipeline([("preprocess", preprocessor), ("model", regressor)])
                with st.spinner("모델을 학습하고 있습니다..."):
                    pipe.fit(X_train, y_train)
                    pred = np.maximum(0, pipe.predict(X_test))
                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred)
                st.session_state.ml_result = {
                    "pipe": pipe, "features": features, "metrics": (mae,rmse,r2),
                    "actual": y_test.to_numpy(), "pred": pred,
                    "dates": work["date"].iloc[split:].to_numpy(),
                    "test_ratio": test_ratio, "model": model_name
                }
    if "ml_result" in st.session_state:
        result = st.session_state.ml_result
        mae,rmse,r2 = result["metrics"]
        st.divider()
        st.markdown("### 모델 평가 결과")
        m1,m2,m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.3f}", help="평균 절대 오차. 낮을수록 좋습니다.")
        m2.metric("RMSE", f"{rmse:.3f}", help="큰 오차에 더 민감한 평균 제곱근 오차. 낮을수록 좋습니다.")
        m3.metric("R²", f"{r2:.3f}", help="모델이 종속변수의 변동을 설명하는 정도입니다.")
        st.caption(f"모델: {result['model']} | 학습·테스트 분리: 시간순 {100*(1-result['test_ratio']):.0f}% / {100*result['test_ratio']:.0f}%")
        compare = pd.DataFrame({"실제 판매량":result["actual"], "예측 판매량":result["pred"]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=compare["실제 판매량"], mode="lines", name="실제값", line=dict(color="#9DB3D8")))
        fig.add_trace(go.Scatter(y=compare["예측 판매량"], mode="lines", name="예측값", line=dict(color="#234A91")))
        fig.update_layout(height=370, xaxis_title="테스트 데이터 순서", yaxis_title="판매량",
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(compare.head(30), use_container_width=True, hide_index=True)
        insight("테스트 데이터는 날짜 기준으로 학습 데이터 뒤에 배치했습니다. R²가 낮거나 음수일 경우 선택한 변수와 선형 회귀만으로 판매량 변동을 충분히 설명하지 못한다는 뜻입니다.")

# ---------------------------------------------------------
# 5. 재고 관리
# ---------------------------------------------------------
elif page == "재고 관리":
    st.markdown('<div class="main-title">예측 기반 재고 관리</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">현재 관측된 재고와 머신러닝 예측 판매량을 비교해 재고 부족 가능성을 살펴봅니다.</div>', unsafe_allow_html=True)

    if "ml_result" not in st.session_state:
        st.warning("먼저 '머신러닝 예측' 메뉴에서 모델을 학습해 주세요.")
        st.stop()
    if "sku" not in df.columns or "stock_available" not in df.columns:
        st.error("재고 관리에 필요한 sku 또는 stock_available 열이 없습니다.")
        st.stop()

    result = st.session_state.ml_result
    period = st.number_input("재고 계획 기간(일)", min_value=1, max_value=90, value=7)
    safety_days = st.number_input("안전재고 기간(일)", min_value=0, max_value=30, value=2)
    pipe, features = result["pipe"], result["features"]

    # SKU·채널·지역 조합별 가장 최근 행을 기준으로 미래 시나리오를 생성
    group_cols = [c for c in ["sku","channel","region"] if c in df.columns]
    latest = df.sort_values("date").groupby(group_cols, as_index=False).tail(1).copy()
    future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=int(period), freq="D")
    forecast_rows = []
    for _, row in latest.iterrows():
        future = pd.DataFrame([row.to_dict()] * len(future_dates))
        future["date"] = future_dates
        future["month"] = future["date"].dt.month
        future["day_of_week"] = future["date"].dt.dayofweek
        future["year"] = future["date"].dt.year
        future["day_name"] = future["date"].dt.day_name()
        # 미래에 알 수 없는 변수는 가장 최근 관측값을 유지하는 시나리오
        for f in features:
            if f not in future.columns:
                future[f] = 0
        try:
            preds = np.maximum(0, pipe.predict(future[features]))
            forecast_rows.append({
                "sku": row["sku"],
                "channel": row["channel"] if "channel" in row else "전체",
                "region": row["region"] if "region" in row else "전체",
                "현재 재고": float(row["stock_available"]),
                "예측 기간 판매량": float(np.sum(preds)),
                "일평균 예측 판매량": float(np.mean(preds)),
            })
        except Exception:
            continue
    if not forecast_rows:
        st.error("재고 예측을 생성하지 못했습니다. 모델의 독립변수 구성을 확인하세요.")
        st.stop()

    inv = pd.DataFrame(forecast_rows)
    inv["안전재고"] = inv["일평균 예측 판매량"] * safety_days
    inv["필요 재고량"] = inv["예측 기간 판매량"] + inv["안전재고"]
    inv["재고 차이"] = inv["현재 재고"] - inv["필요 재고량"]
    inv["권장 발주량"] = np.ceil(np.maximum(0, inv["필요 재고량"] - inv["현재 재고"]))
    inv["재고 상태"] = np.where(inv["재고 차이"] < 0, "재고 부족",
                         np.where(inv["재고 차이"] > inv["일평균 예측 판매량"] * 5, "과잉 가능", "적정"))
    a,b,c = st.columns(3)
    a.metric("분석 조합 수", f"{len(inv):,}")
    b.metric("재고 부족 조합", f"{(inv['재고 상태']=='재고 부족').sum():,}")
    c.metric("총 권장 발주량", fmt_num(inv["권장 발주량"].sum()))
    status = inv["재고 상태"].value_counts().rename_axis("재고 상태").reset_index(name="조합 수")
    fig = px.bar(status, x="재고 상태", y="조합 수", color="재고 상태",
                 color_discrete_map={"재고 부족":"#C65A5A","적정":"#4F8A78","과잉 가능":"#D5A34A"})
    fig.update_layout(height=300, showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    section_title("상품별 재고 현황")
    st.dataframe(inv.sort_values("권장 발주량", ascending=False).round(1),
                 use_container_width=True, hide_index=True)
    insight(f"예측 기간은 {period}일, 안전재고는 일평균 예측 판매량의 {safety_days}일분으로 계산했습니다. 미래 가격·프로모션·입고량 등은 별도 계획 정보가 없으므로 각 조합의 최근 관측값을 유지한 시나리오 예측입니다. 따라서 발주 확정값이 아닌 의사결정 참고값입니다.")

# ---------------------------------------------------------
# 6. 데이터 전처리
# ---------------------------------------------------------
elif page == "데이터 전처리":
    st.markdown('<div class="main-title">데이터 전처리 및 품질 진단</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">원본 데이터의 품질을 진단하고 선택한 처리 결과를 분석 전체에 적용합니다.</div>', unsafe_allow_html=True)

    current = st.session_state.working_df.copy()
    missing = current.isna().sum()
    duplicate_count = int(current.duplicated().sum())
    c1,c2,c3 = st.columns(3)
    c1.metric("전체 행 수", f"{len(current):,}")
    c2.metric("중복 행 수", f"{duplicate_count:,}")
    c3.metric("결측 셀 수", f"{int(missing.sum()):,}")

    section_title("열별 결측치 진단")
    miss_table = pd.DataFrame({"변수":missing.index, "결측치":missing.values,
                               "결측 비율(%)":(missing.values/len(current)*100).round(2)})
    st.dataframe(miss_table.sort_values("결측치", ascending=False), use_container_width=True, hide_index=True)
    section_title("전처리 방법 선택")
    drop_duplicates = st.checkbox("중복 행 제거", value=True)
    fill_numeric = st.selectbox("수치형 결측치 처리", ["중앙값으로 대체", "평균값으로 대체", "결측 행 제거"], index=0)
    fill_categorical = st.selectbox("범주형 결측치 처리", ["최빈값으로 대체", "결측 행 제거"], index=0)
    if st.button("선택한 전처리 적용", type="primary"):
        cleaned = current.copy()
        before_rows = len(cleaned)
        if drop_duplicates:
            cleaned = cleaned.drop_duplicates()
        num_cols = cleaned.select_dtypes(include=np.number).columns
        cat_cols = cleaned.select_dtypes(include=["object","category","bool"]).columns
        if fill_numeric == "결측 행 제거":
            cleaned = cleaned.dropna(subset=list(num_cols))
        else:
            for col in num_cols:
                if cleaned[col].isna().any():
                    value = cleaned[col].median() if fill_numeric == "중앙값으로 대체" else cleaned[col].mean()
                    cleaned[col] = cleaned[col].fillna(value)
        if fill_categorical == "결측 행 제거":
            cleaned = cleaned.dropna(subset=list(cat_cols))
        else:
            for col in cat_cols:
                if cleaned[col].isna().any():
                    mode = cleaned[col].mode(dropna=True)
                    if len(mode):
                        cleaned[col] = cleaned[col].fillna(mode.iloc[0])
        st.session_state.working_df = cleaned
        st.session_state.pop("ml_result", None)
        st.success(f"전처리 완료: {before_rows:,}행 → {len(cleaned):,}행. 변경 사항은 현재 세션의 모든 분석 페이지에 적용됩니다.")
        st.rerun()

    section_title("이상치 탐색(IQR 기준)")
    numeric = current.select_dtypes(include=np.number).columns.tolist()
    if numeric:
        out_col = st.selectbox("이상치를 확인할 변수", numeric,
                               index=numeric.index("units_sold") if "units_sold" in numeric else 0)
        q1, q3 = current[out_col].quantile([.25,.75])
        iqr = q3-q1
        lower, upper = q1-1.5*iqr, q3+1.5*iqr
        out_count = int(((current[out_col] < lower) | (current[out_col] > upper)).sum())
        x1,x2,x3 = st.columns(3)
        x1.metric("IQR 하한", f"{lower:.2f}")
        x2.metric("IQR 상한", f"{upper:.2f}")
        x3.metric("이상치 후보 수", f"{out_count:,}")
        fig = px.box(current, y=out_col, points="outliers", color_discrete_sequence=["#3156A5"])
        fig.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("IQR 기준을 벗어난 값은 이상치 후보입니다. 실제 오류인지 정상적인 극단값인지는 데이터 생성 맥락을 확인한 뒤 판단해야 합니다.")

    st.download_button("현재 전처리 데이터 CSV 다운로드",
                       data=current.to_csv(index=False).encode("utf-8-sig"),
                       file_name="FMCG_analysis_preprocessed.csv",
                       mime="text/csv")
