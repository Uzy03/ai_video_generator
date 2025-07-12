# プロジェクトルートで
mkdir -p external
cd external

# Wan2.1
git clone https://github.com/Wan-Video/Wan2.1.git

# HunyuanVideo-I2V
git clone https://github.com/tencent/HunyuanVideo-I2V.git

# Wan2.1 の 14B モデル
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
    --local-dir ./Wan2.1/Wan2.1-I2V-14B-720P

# HunyuanVideo-I2V
huggingface-cli download tencent/HunyuanVideo-I2V \
    --local-dir ./HunyuanVideo-I2V/ckpts

cd ..

pip install -r requirements.txt
pip install -r external/HunyuanVideo-I2V/requirements.txt

tmp_req=$(mktemp)
grep -v '^flash_attn' external/Wan2.1/requirements.txt > "$tmp_req"
pip install -r "$tmp_req"
rm "$tmp_req"