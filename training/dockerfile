FROM tensorflow/tensorflow:latest

# 必要なパッケージのインストール
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    libsndfile1

# Jupyter Labのインストール
RUN pip install jupyterlab

# ワーキングディレクトリの設定
WORKDIR /app

# コードとデータのコピー
COPY requirements.txt /app
COPY . /app

# pipで必要なパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# ポートの公開
EXPOSE 8888

# Jupyter Labの起動コマンド
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]