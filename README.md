# Bradley-Terry レーティング分析システム
# 📊 [B-TRatingの表・グラフ説明はこちら](docs/graph_explanation.md)

CSVファイルから試合結果を読み込み、Bradley-Terryモデルを使用してプレイヤーの強さを分析するWebアプリケーションです。

## 機能

- ✅ CSV形式の試合データをアップロード
- ✅ Bradley-Terryモデルによるレーティング計算（ベイズ推定）
- ✅ 95%信頼区間付きレーティング表示
- ✅ 対戦勝率マトリックス
- ✅ インタラクティブな勝率可視化（選手クリックで色表示）
- ✅ 複数の分析チャート生成
- ✅ 結果の保存と共有
- ✅ 保存済み結果の一覧表示

## スクリーンショット

### メイン画面
- CSVファイルのドラッグ&ドロップアップロード
- レーティングランキング表示
- 対戦勝率マトリックス

### 分析結果
- 信頼区間付きレーティングランキング
- レーティング推移グラフ
- 収束分析
- 対戦勝率の色分け表示

## 必要要件

- Python 3.8以上
- pip（Pythonパッケージマネージャー）

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone <このリポジトリのURL>
cd rating-app
```

### 2. 仮想環境の作成と有効化

```bash
python3 -m venv venv
source venv/bin/activate  # Linuxの場合
# venv\Scripts\activate  # Windowsの場合
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 4. 必要なディレクトリの作成

```bash
mkdir -p logs uploads results
```

### 5. アプリケーションの起動

#### 開発環境

```bash
python app.py
```

ブラウザで `http://localhost:5000` にアクセスしてください。

#### 本番環境（Gunicorn使用）

```bash
gunicorn -c gunicorn_config.py app:app
```

デフォルトでは `http://127.0.0.1:8000` で起動します。

### 6. リバースプロキシの設定（オプション）

Nginxを使用してサブドメインでアクセスする場合の設定例：

```nginx
server {
    listen 80;
    server_name rating.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 7. 日本語フォント設定（文字化け対策）

日本語（漢字・カナ）を含むグラフの軸やラベルが文字化けする場合は、
matplotlibのフォントを `Noto Sans CJK JP` に設定してください。

1. サーバーに日本語フォントをインストール

```bash
sudo apt install fonts-noto-cjk
```

2. `app.py` のmatplotlib設定直後に以下を追加

```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
```

これでグラフの日本語が正しく表示されます。

#### VPS/WSLで日本語フォントを有効化するための追加手順

1. 日本語フォント（Noto Sans CJK JP）とfontconfigをインストール

```bash
sudo apt update
sudo apt install fonts-noto-cjk fontconfig
```

2. インストール確認（Noto Sans CJK JPが表示されればOK）
```bash
fc-list | grep "Noto Sans CJK JP"
```

3. Flask/gunicornアプリを再起動

これでmatplotlibの日本語グラフが正しく表示されます。

## データフォーマット

CSVファイルは以下の形式である必要があります：

```csv
winner,loser
アリス,ボブ
キャロル,アリス
ボブ,キャロル
```

- デフォルトでは `winner` と `loser` という列名を使用
- カスタム列名も設定可能

## プロジェクト構成

```
rating-app/
├── app.py                    # Flaskアプリケーションメイン
├── bt_rating_bayesian.py     # Bradley-Terryベイズモデル実装
├── bt_rating.py              # 基本的なBradley-Terryモデル
├── gunicorn_config.py        # Gunicorn設定
├── requirements.txt          # Python依存パッケージ
├── templates/                # HTMLテンプレート
│   ├── index.html           # メインページ
│   ├── result_view.html     # 結果詳細ページ
│   └── results_list.html    # 結果一覧ページ
├── uploads/                  # アップロードされたCSVファイル（一時）
├── results/                  # 保存された分析結果
└── logs/                     # アプリケーションログ
```

## 技術スタック

### バックエンド
- **Flask**: Webフレームワーク
- **NumPy**: 数値計算
- **Pandas**: データ処理
- **Matplotlib**: グラフ生成
- **SciPy**: 統計計算
- **Gunicorn**: WSGIサーバー

### フロントエンド
- **HTML5/CSS3**: UI
- **JavaScript**: インタラクティブ機能
- **Jinja2**: テンプレートエンジン

## 使い方

1. トップページにアクセス
2. CSVファイルをドラッグ&ドロップまたは選択
3. 分析タイトルを入力（オプション）
4. 「分析開始」ボタンをクリック
5. 結果を確認
   - レーティングテーブルの行をクリックすると、その選手が他の選手に勝つ確率が色で表示されます
   - 対戦勝率マトリックスで全ての対戦組み合わせを確認できます
6. 「結果を保存して共有可能にする」にチェックを入れると、結果が保存され共有URLが発行されます

## 導入時に行った設定

### システムパッケージ
```bash
# Python環境のセットアップ
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Nginxのインストール（リバースプロキシ用）
sudo apt install nginx
```

### Python環境
```bash
# 仮想環境の作成
python3 -m venv venv
source venv/bin/activate

# 依存パッケージのインストール
pip install flask numpy pandas matplotlib scipy gunicorn
pip freeze > requirements.txt
```

### ディレクトリ構成
```bash
mkdir -p logs uploads results templates
```

### Nginx設定
```bash
# /etc/nginx/sites-available/rating に設定ファイルを作成
sudo nano /etc/nginx/sites-available/rating

# シンボリックリンクを作成
sudo ln -s /etc/nginx/sites-available/rating /etc/nginx/sites-enabled/

# Nginx設定をテスト
sudo nginx -t

# Nginxを再起動
sudo systemctl restart nginx
```

### サービス化（systemd）- サーバー再起動時の自動起動設定

アプリケーションをバックグラウンドで常時起動し、サーバー再起動後も自動的に起動するようにします。

#### 1. サービスファイルの作成

```bash
# /etc/systemd/system/rating-app.service を作成
sudo nano /etc/systemd/system/rating-app.service
```

以下の内容を記述（`xxx`の部分は実際のユーザー名に置き換えてください）：

```ini
[Unit]
Description=Bradley-Terry Rating Application
After=network.target

[Service]
User=xxx
WorkingDirectory=/home/xxx/rating-app
Environment="PATH=/home/xxx/rating-app/venv/bin"
ExecStart=/home/xxx/rating-app/venv/bin/gunicorn -c gunicorn_config.py app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 2. サービスの有効化と起動

```bash
# systemdを再読み込み
sudo systemctl daemon-reload

# 既存のgunicornプロセスがあれば停止
pkill -f "gunicorn.*app:app"

# サービスを有効化（自動起動設定）
sudo systemctl enable rating-app

# サービスを起動
sudo systemctl start rating-app

# 状態確認
sudo systemctl status rating-app
```

#### 3. サービスの管理コマンド

```bash
# サービスの起動
sudo systemctl start rating-app

# サービスの停止
sudo systemctl stop rating-app

# サービスの再起動
sudo systemctl restart rating-app

# サービスの状態確認
sudo systemctl status rating-app

# ログの確認
sudo journalctl -u rating-app -f

# 自動起動の無効化
sudo systemctl disable rating-app
```

これで、VPSサーバーを再起動しても自動的にアプリケーションが起動するようになります。

## トラブルシューティング

### ポートが使用中のエラー
```bash
# 既存のプロセスを確認
ps aux | grep gunicorn

# プロセスを終了
pkill -f "gunicorn.*app:app"
```

### パッケージのインストールエラー
```bash
# システムパッケージの更新
sudo apt update
sudo apt install python3-dev build-essential

# pipのアップグレード
pip install --upgrade pip
```

### ログの確認
```bash
# アプリケーションログ
tail -f logs/error.log
tail -f logs/access.log

# systemdサービスのログ
sudo journalctl -u rating-app -f
```

## ライセンス

MIT License

## 作者

Bradley-Terry レーティング分析システム

