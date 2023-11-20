# 自由対話のための日本語LLM用GUI
- LoRAで学習した日本語LLMの簡易デモアプリを作成しました。
- 学習の結果をGUIで確認したい時にお使い下さい。
- HugginhfaceのモデルをLoRAで学習したLLMのみを対象としています。ご了承下さい。
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

https://github.com/ohashi3399/llm_chat_demo_gui/assets/87519834/91fe44b2-9d2a-4e33-9c47-679753c7a6e2


https://github.com/ohashi3399/llm_chat_demo_gui/assets/87519834/643b5cd5-9076-44ba-994e-3b5c844746c7


# 📖 参考
## Streamlit
- https://streamlit.io/
