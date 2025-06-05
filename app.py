import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import japanize_matplotlib

st.set_page_config(page_title="求人分析ツール", layout="centered")
st.title("📊 求人データ分析アプリ")

# ファイルアップロード
uploaded_files = st.file_uploader("CSVファイルをアップロード（複数可）", type="csv", accept_multiple_files=True)

if uploaded_files:
    # 複数ファイル結合
    dfs = [pd.read_csv(file) for file in uploaded_files]
    df_all = pd.concat(dfs, ignore_index=True)

    st.subheader("🔍 アップロードされたデータ（先頭5行）")
    st.dataframe(df_all.head())

    # 求人単位に平均CTR・ARを集計
    df_grouped = df_all.groupby("求人").agg({
        "クリック率（CTR）": "mean",
        "応募率 (AR)": "mean"
    }).reset_index()

    # 応募率0を除外
    df_grouped = df_grouped[df_grouped["応募率 (AR)"] > 0].copy()

    # 線形回帰で分類
    X = df_grouped[["クリック率（CTR）"]].values
    y = df_grouped["応募率 (AR)"].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    df_grouped["分類"] = np.where(y > y_pred, "応募率が高い群（原稿良好）", "応募率が低い群（原稿要改善）")
    df_grouped["位置"] = np.where(X.flatten() > np.median(X), "応募率高", "応募率低")

    def classify(row):
        if row["分類"] == "応募率が高い群（原稿良好）":
            return "タイトル改善群" if row["位置"] == "応募率低" else "ベストプラクティス群"
        else:
            return "全面改善群" if row["位置"] == "応募率低" else "原稿改善群"

    df_grouped["群"] = df_grouped.apply(classify, axis=1)

    # グラフ描画
    st.subheader("📈 分析結果（散布図）")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "ベストプラクティス群": "green",
        "タイトル改善群": "blue",
        "原稿改善群": "orange",
        "全面改善群": "red"
    }

    for group, data in df_grouped.groupby("群"):
        ax.scatter(data["クリック率（CTR）"], data["応募率 (AR)"],
                   color=colors[group], label=group, alpha=0.7)
        for i in data.index:
            x = data.at[i, "クリック率（CTR）"]
            y = data.at[i, "応募率 (AR)"]
            ax.text(x, y, str(i), fontsize=8)

    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, linestyle="--", color="black", label="回帰直線")

    ax.set_xlabel("クリック率（CTR）")
    ax.set_ylabel("応募率 (AR)")
    ax.set_title("求人別 応募率とクリック率の平均散布図（群分類付き）")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # 分析結果表示
    st.subheader("📄 群分類結果（表）")
    st.dataframe(df_grouped)

else:
    st.info("まずはCSVファイルをアップロードしてください。")