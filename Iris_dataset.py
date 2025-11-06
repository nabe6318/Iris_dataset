# Iris 総当り（全組合せ）散布図アプリ / Streamlit
# - 先頭50行の表示
# - 総当りの散布図（ペアごとの散布図 + 対角はヒストグラム）
# - 特徴量の選択・表示行数の調整が可能（既定50行）
# -------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pandas.plotting import register_matplotlib_converters

from sklearn.datasets import load_iris

register_matplotlib_converters()

st.set_page_config(page_title="Iris 総当り散布図", layout="wide")

# ------------------------------------
# はじめに：IRISデータセットの説明
# ------------------------------------
st.title("🌸 機械学習における Iris データセットの概要")
st.markdown(
    """
    **Iris（アイリス）データセット** は、機械学習の入門で最もよく使われるサンプルデータの1つです。
    - アヤメ属の3種類の花（*setosa*, *versicolor*, *virginica*）のデータを含みます。
    - 各花について、以下の4つの特徴量が測定されています。
        1. がく片の長さ (*sepal length*)
        2. がく片の幅 (*sepal width*)
        3. 花弁の長さ (*petal length*)
        4. 花弁の幅 (*petal width*)
    - 各特徴量はセンチメートル単位で測定され、合計150個体分のデータがあります。

    このデータは1950年代に統計学者 **R.A. Fisher** によって収集され、
    現在では「分類問題（Classification）」を学ぶための代表的教材として利用されています。

    ここではまずデータの構造を確認し、4つの変数の組み合わせによる**総当り散布図**で
    どの特徴量の組み合わせがクラス分離に適しているかを可視化します。
    -  雑草研・作物生産システム研ゼミ
    """
)

# ------------------------------------
# データ読み込み
# ------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame.copy()  # features + target
# species 名を付与
species_names = dict(enumerate(iris.target_names))
df["species"] = df["target"].map(species_names)

feature_cols = list(iris.feature_names)

# ------------------------------------
# サイドバー設定
# ------------------------------------
st.sidebar.header("表示オプション / Options")
show_rows = st.sidebar.number_input("表示行数 / Rows to show", min_value=10, max_value=len(df), value=50, step=10)
selected_features = st.sidebar.multiselect("特徴量の選択 / Select features", feature_cols, default=feature_cols)
alpha = st.sidebar.slider("点の透過度 / Alpha", 0.1, 1.0, 0.7, 0.1)
marker_size = st.sidebar.slider("点サイズ / Marker size", 5, 50, 18, 1)

if len(selected_features) < 2:
    st.sidebar.warning("少なくとも2つの特徴量を選択してください。")

st.title("Iris データ：先頭50行→総当り散布図")

# ------------------------------------
# 1) 先頭50行の表示（既定）
# ------------------------------------
st.markdown("### 1) データの確認（先頭行）")
st.dataframe(df[selected_features + ["species"]].head(show_rows), use_container_width=True)
st.caption("まず表で特徴量のスケールや外れ値の有無を確認します。")

# ------------------------------------
# 2) 総当りの散布図
# ------------------------------------
if len(selected_features) >= 2:
    st.markdown("### 2) 総当り散布図（全組合せ）")
    st.caption("行列形式で全ての組合せを比較します。対角はヒストグラム。凡例は右上に1回のみ表示。")

    feats = selected_features
    n = len(feats)
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(3.2*n, 3.2*n), dpi=130)

    # 既定でaxesは2次元配列
    if n == 1:
        axes = np.array([[axes]])

    species_unique = df["species"].unique()

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            xcol = feats[j]
            ycol = feats[i]
            if i == j:
                # ヒストグラム（クラス別重ね）
                for k, sp in enumerate(species_unique):
                    vals = df.loc[df["species"] == sp, xcol]
                    ax.hist(vals, bins=12, alpha=0.5, label=sp)
                ax.set_ylabel("")
                ax.set_xlabel(xcol)
            else:
                # 散布図（クラス別）
                for k, sp in enumerate(species_unique):
                    dsub = df[df["species"] == sp]
                    ax.scatter(dsub[xcol], dsub[ycol], s=marker_size, alpha=alpha, label=sp)
                if j == 0:
                    ax.set_ylabel(ycol)
                else:
                    ax.set_ylabel("")
                ax.set_xlabel("")
            # 軸ラベル（下段と左列だけに付ける）
            if i == n - 1:
                ax.set_xlabel(xcol)
            if j == 0 and i != j:
                ax.set_ylabel(ycol)

    # 凡例は右上の1面だけにまとめる
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        axes[0, -1].legend(loc="upper right", bbox_to_anchor=(1.05, 1.0))

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ------------------------------------
# 3) 補足
# ------------------------------------
with st.expander("📝 使い方のヒント"):
    st.markdown(
        """
        - **どのペアが分離しやすいか**（クラスが重ならないか）を目視で比較します。
        - 一般に **petal length × petal width** は分離が良く、CARTの可視化（決定境界）にも向きます。
        - 特徴量を2つ以上選ぶと、自動で総当り行列を描画します。
        """
    )
