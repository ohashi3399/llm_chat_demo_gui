# 自由対話のための日本語LLM用GUI
- LoRAで学習した日本語LLMの簡易デモアプリを作成しました。
- 学習の結果をGUIで確認したい時にお使い下さい。
- powered by Streamlit

# 🖥️ 動作環境
- Ubuntu20.04
- Python3.8.10
- torch==1.13.1+cu117
- transformers==4.35.1

# ⚙️ 環境構築
```
git clone https://github.com/ohashi3399/llm_chat_demo_gui.git
pip install -r requirements.txt
```

# ⚙️ 動作手順
```
streamlit run chatdemo.py
```

> **Warning**
> HugginhfaceのモデルをLoRAで学習したLLMのみを対象としています。
> ご了承下さい。

# 📖 参考
## Streamlit
- https://streamlit.io/