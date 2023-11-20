import os
import time
import copy
import datetime
import numpy as np
import streamlit as st

from PIL import Image
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_selector import PromptTemplate_ja


class Chatter(object):

    def __init__(self):

        # 初期画面の表示
        self.init_page()

        # session_stateに何もない時(=初回のみ)に実行される関数
        if not st.session_state:
            self.init_only_once()

        # session_stateにmodelがない時に再実行される関数
        if "model" not in st.session_state:
            self.select_model()

        # 履歴のリセットボタンの表示
        self.set_clear_chat_button()

        # 選択した学習データセットとモデルの表示
        self.display_verbose()

        # 対話部
        self.run()

        return

    def init_only_once(self) -> None:
        out_dir = "./out"
        chatlog_dir = "./chatlog"
        os.makedirs(chatlog_dir, exist_ok=True)

        # 学習したデータセットを変数に追加して下さい
        task_names = (
            "jdd_multi_turn",
            "jwc_multi_turn",
            )

        # 学習したモデルを変数に追加して下さい
        model_cards = (
            "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2",
            "rinna/youri-7b-chat",
            "cyberagent/calm2-7b-chat",
            "stabilityai/japanese-stablelm-instruct-beta-7b",
            "stockmark/stockmark-13b-instruct",
            "pfnet/plamo-13b-instruct"
            )

        user_icon = np.array(Image.open("./datasets/icons/humation/user1.png"))
        ai_icon = np.array(Image.open("./datasets/icons/humation/ai1.png"))

        st.session_state.out_dir = out_dir
        st.session_state.model_cards = model_cards
        st.session_state.task_names = task_names
        st.session_state.chatlog_dir = chatlog_dir
        st.session_state.user_icon = user_icon
        st.session_state.ai_icon = ai_icon
        st.session_state.chat_log = ["input,output,generation_time"]
        st.session_state.date = datetime.datetime.now().strftime('%Y_%m_%d_ %H_%M_%S')
        st.session_state.like_or_not = list()
        st.session_state.eos_error = """<eos>トークンが出力されませんでした。学習時の文末に<eos>トークンが付与されていないか、学習が不十分なことが原因として考えられます。学習が不十分とは、epoch数が少ないか、学習率が極端に大きいもしくは小さすぎる、学習率のschedulerが設定されていない、等が考えられます。また、学習時のプロンプトに'出力後に<eos>トークンを必ず出力しなさい。'という文章を追加して再学習すると改善されることがあります。"""
        return

    def init_page(self) -> None:
        st.set_page_config(page_title="💬 日本語対話デモ")
        st.header("🖋️ 2. 日本語自由対話モデル")
        st.sidebar.title("📖 1. 言語モデルの選択")
        return

    def set_clear_chat_button(self) -> None:
        clear_button = st.sidebar.button("会話のリセット", key="chatlog_clear")

        if clear_button or "messages" not in st.session_state:
            st.session_state.messages = list()
        return

    def select_model(self):
        task_name = st.sidebar.selectbox(
            "学習したデータセットを選択して下さい。",
            st.session_state.task_names,
            index=None,
            placeholder="データセットは...",
            )
        if not task_name:
            st.stop()

        model_name = st.sidebar.selectbox(
            "言語モデルを選択して下さい。",
            st.session_state.model_cards,
            index=None,
            placeholder="言語モデルは...",
            )
        if not model_name:
            st.stop()

        with st.status("対話モデル起動中...", expanded=True) as status:

            adapter_path = f"{st.session_state.out_dir}/{task_name}/{model_name}"

            st.write("1/4 プロンプトフォーマットの作成")
            prompter = PromptTemplate_ja(model_name)
            prompt_format = prompter.set_prompt_format()

            st.write("2/4 トークナイザーの読み込み")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                )
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # youri-7b-chat, stablelm-7bに必要

            st.write("3/4 学習済みモデルの読み込み")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=True,
                torch_dtype="auto",
                )
            # model.resize_token_embeddings(len(tokenizer)) # youri-7b-chat, stablelm-7bに必要

            st.write("4/4 Adapterモデルの読み込み")
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                device_map="auto",
                )

            model.eval()

            print(type(model))
            status.update(
                label="対話モデル起動完了",
                state="complete",
                expanded=False
                )

        st.session_state.task_name = task_name
        st.session_state.model_name = model_name
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.adapter_path = adapter_path
        st.session_state.prompt_format = prompt_format
        return

    def display_verbose(self):
        with st.sidebar:
            verbose = list()
            verbose.append(f"## 出力先フォルダ\n- {st.session_state.chatlog_dir}")
            verbose.append(f"## 学習データセット\n- {st.session_state.task_name}")
            verbose.append(f"## 学習モデル\n- {st.session_state.model_name}")
            verbose.append(f"## 推論手法\n- 4bit量子化")
            verbose = "\n".join(verbose)
            st.markdown(verbose)
        return

    def generate_prompt(self, messages: list, eos_token: str):
        pf = st.session_state.prompt_format
        prompt = list()

        for message in messages:
            role = pf["input"] if message["from"] == "user" else pf["output"]
            line = f"{pf['sep']}{role}{message['value']}"
            if role == pf["output"]:
                line += eos_token
            prompt.append(line)
        prompt.append(f"\n{pf['output']}")

        prompt = "\n".join(prompt)
        return prompt

    def generate_response(self, prompt):
        input_ids = st.session_state.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False
            ).input_ids.cuda()

        outputs = st.session_state.model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.75,
            top_k=40,
            no_repeat_ngram_size=2,
            )
        outputs = outputs[0].tolist()

        eos_indices = [i for i, x in enumerate(outputs) if x == st.session_state.tokenizer.eos_token_id]
        assert len(eos_indices) > 0, self.eos_error
        eos_index = eos_indices[-1]

        response = st.session_state.tokenizer.decode(outputs[:eos_index])

        return response

    def extract_answer(self, response):
        sentinel = st.session_state.prompt_format["output"]
        sentinelLoc = response.rfind(sentinel)

        if sentinelLoc >= 0:
            result = response[sentinelLoc+len(sentinel):].replace("\n", "")
        else:
            result = 'エラー：指定されたプロンプトテンプレートが見つかりませんでした。プロンプトテンプレートが誤っている可能性があります。プロンプトテンプレートが選択したモデルと一致しているか確認して下さい。'
            with st.chat_message("assistant"):
                st.markdown(result)

        return result

    def reply(self) -> tuple:
        prompt = self.generate_prompt(
            st.session_state.messages,
            st.session_state.tokenizer.eos_token
            )
        answer = self.extract_answer(self.generate_response(prompt))
        return answer

    def _display_chatlog(self):
        messages = st.session_state.get("messages", [])
        for idx, message in enumerate(messages):

            if message["from"] == "gpt":
                with st.chat_message("assistant", avatar=st.session_state.ai_icon):
                    st.markdown(message["value"])
            else:
                with st.chat_message("user", avatar=st.session_state.user_icon):
                    st.markdown(message["value"])
        return

    def _dump_chatlog(self):
        model_name = copy.deepcopy(st.session_state.model_name)
        model_name = model_name.replace("/", "_")
        filename = f"{st.session_state.chatlog_dir}/{st.session_state.task_name}_{model_name}_{st.session_state.date}.csv"

        with open(filename, mode="w", encoding="utf-8") as o:
            o.write("\n".join(st.session_state.chat_log))
        return

    def run(self):

        if user_input := st.chat_input("文章を入力しましょう"):

            st.session_state.messages.append(
                {"from": "user", "value": user_input}
                )

            self._display_chatlog()

            with st.spinner("生成中 ..."):
                begin = time.perf_counter()
                answer = self.reply()
                duration = time.perf_counter() - begin

            st.session_state.messages.append(
                {"from": "gpt", "value": answer}
                )

            with st.chat_message("assistant", avatar=st.session_state.ai_icon):
                st.markdown(answer)

            st.session_state.chat_log.append(f"{user_input},{answer},{duration:02f}")

        if st.sidebar.button("会話履歴の保存", key="save"):
            self._dump_chatlog()
        return


def main():
    chatbot = Chatter()


if __name__ == "__main__":
    main()
