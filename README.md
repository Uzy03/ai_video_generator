# AI Video Generator — 画像1枚から動画を生成する I2V プレイグラウンド

> **1 枚の静止画を入力すると、数秒の動画を自動生成します。**
> 熊本大学 情報融合学館オープンキャンパス展示用に作成しました。

---

## ✨ 特長

* **複数モデル対応** — PixArt-Video / LTX-Video / HunyuanVideo など、ワンクリックで切り替え可能
* **軽量フルスタック** — フロントエンドに Streamlit、バックエンドに FastAPI + WebSocket
* **Dev Container 完備** — VS Code で *Reopen in Container* を選ぶだけで環境を再現
* **Apple Silicon（M シリーズ MPS）** と **CUDA GPU** の両方を自動サポート
* モデル重み (`*.safetensors`) は初回起動時に自動ダウンロード

---

## ⚡ クイックスタート（推奨：VS Code Dev Container）

1. **前提ソフトウェア**

   * Docker Desktop ≥ 24
   * VS Code 拡張機能 **Dev Containers** (`ms-vscode-remote.remote-containers`)
2. リポジトリをクローン

```bash
git clone https://github.com/Uzy03/ai_video_generator.git
cd ai_video_generator
```

3. Dev Container を起動

   * macOS : <kbd>⌘</kbd><kbd>⇧</kbd><kbd>P</kbd> / Win・Linux : <kbd>Ctrl</kbd><kbd>Shift</kbd><kbd>P</kbd>
   * **Dev Containers: Open Folder in Container…** を実行し、フォルダを選択
   * 初回のみビルドに ≈10 分かかります
4. コンテナ内ターミナルでデモを起動

```bash
streamlit run app/quick_demo.py --server.port 8501
```

5. ブラウザで [http://localhost:8501](http://localhost:8501) を開き、`sample_img/` から画像を選択 → モデルを選び **Generate** をクリック。

> **ヒント** : `sample_img/forest.png` を入力すると、`sample_output/forest.mp4` が自動生成され、その場で再生されます。

---

## 🐍 ローカル Python 環境で実行する場合

GPU が無い場合は 256×256 / 2 FPS 程度です。仮想環境（venv / conda）の利用を推奨します。

```bash
# Python 3.10 以上を想定
python -m venv .venv
source .venv/bin/activate      # Windows は .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# モデル重みを取得（初回のみ）
bash scripts/setup_repo.sh

# デモ起動
streamlit run app/quick_demo.py
```

---

## 🖼️ サンプルアセット

| 入力 (`sample_img`) | 出力 (`sample_output`) |
| ----------------- | -------------------- |
| ![girl_with_cat](sample_img/https---qiita-image-store.s3.ap-northeast-1.amazonaws.com-0-235259-92fc9bcb-49cd-4d54-b4ee-912e9da590c1.jpeg) | [▶︎ girl_with_cat.mp4](sample_output/video_output_0_a-young-girl-bravely-and_171198_544x448x113_0.mp4)       | `forest.mp4`         |
| `city.png`        | `city.mp4`           |

> 上記は **PixArt-Video（16 フレーム／10 FPS）** で生成した例です。モデルを変えて品質や速度を比較してみてください。

---

## ⚙️ 動作概要

```text
┌────────┐ アップロード  ┌─────────────┐ WS  ┌────────┐   torch   ┌───────────┐
│ブラウザ　│ ───────────▶│ Streamlit 　│────▶│ FastAPI 　│ ───────▶ │ I2V model │
│ (JS)   │             │ フロントエンド│     │ バックエンド│          │ パイプライン │
└────────┘             └─────────────┘     └────────┘          └─────┬─────┘
      ▲  MP4 プレビュー                                 ffmpeg      │
      └────────────────────────────────────────────────────────────┘
```

1. Streamlit が UI を提供し、アップロードされた画像を WebSocket で FastAPI バックエンドへ送信。
2. バックエンドで選択したモデルを実行し、フレームを連結 → **ffmpeg** で MP4 にエンコード。
3. 生成された動画を即座にブラウザへ返し、`<video>` タグで再生します。

---

## 📂 ディレクトリ構成

```text
.
├── app/            # Streamlit UI と FastAPI バックエンド
├── external/       # サードパーティ I2V モデル（git submodule）
├── processing/     # 推論パイプライン & ユーティリティ
├── scripts/        # セットアップ／重みダウンロードスクリプト
├── sample_img/     # 入力用サンプル画像
├── sample_output/  # 生成動画（git ig
```
