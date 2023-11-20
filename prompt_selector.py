class PromptTemplate_ja(object):

    def __init__(self, model_name):
        self.model_name = model_name
        return
    
    def set_prompt_format(self):
        if 'rinna' in self.model_name:
            prompt_format = rinna_prompt_format()
        if 'youri' in self.model_name:
            prompt_format = youri_prompt_format()
        if 'stockmark' in self.model_name:
            prompt_format = stockmark_prompt_format()
        if 'calm' in self.model_name:
            prompt_format = calm_prompt_format()
        if 'stablelm' in self.model_name:
            prompt_format = stablelm_prompt_format()
        if 'elyza' in self.model_name:
            prompt_format = elyza_prompt_format()
        if 'pfnet' in self.model_name:
            prompt_format = pfn_prompt_format()
        return prompt_format


def rinna_prompt_format():
    prompt = dict()
    prompt["instruction"] = "指示:"
    prompt["input"] = "ユーザー:"
    prompt["output"] = "システム:"
    prompt["eos"] = "</s>"
    prompt["sep"] = "<NL>"
    return prompt


def youri_prompt_format():
    prompt = dict()
    prompt["instruction"] = "設定:"
    prompt["input"] = "ユーザー:"
    prompt["output"] = "システム:"
    prompt["eos"] = "</s>"
    prompt["sep"] = "\n"
    return prompt


def stockmark_prompt_format():
    prompt = dict()
    prompt["instruction"] = "Instruction: "
    prompt["input"] = "Input: "
    prompt["output"] = "Output: "
    prompt["eos"] = "</s>"
    prompt["sep"] = "\n\n### "
    return prompt


def calm_prompt_format():
    prompt = dict()
    prompt["instruction"] = "INSTRUCT:"
    prompt["input"] = "USER:"
    prompt["output"] = "ASSISTANT:"
    prompt["eos"] = "<|endoftext|>"
    prompt["sep"] = "\n"
    return prompt


def stablelm_prompt_format():
    prompt = dict()
    prompt["instruction"] = "指示"
    prompt["input"] = "入力"
    prompt["output"] = "応答"
    prompt["eos"] = "<|endoftext|>"
    prompt["sep"] = "\n\n### "
    prompt["B_INST"] = "[INST]"
    prompt["E_INST"] = "[/INST]"
    prompt["B_SYS"] = "<<SYS>>\n"
    prompt["E_SYS"] = "\n<</SYS>>\n\n"
    return prompt


def elyza_prompt_format():
    prompt = dict()
    prompt["instruction"] = ""
    prompt["input"] = ""
    prompt["output"] = ""
    prompt["eos"] = "</s>"
    prompt["sep"] = "\n\n### "
    prompt["B_INST"] = "[INST]"
    prompt["E_INST"] = "[/INST]"
    prompt["B_SYS"] = "<<SYS>>\n"
    prompt["E_SYS"] = "\n<</SYS>>\n\n"
    return prompt


def pfn_prompt_format():
    prompt = dict()
    prompt["instruction"] = "指示: "
    prompt["input"] = "入力: "
    prompt["output"] = "応答: "
    prompt["eos"] = "</s>"
    prompt["sep"] = "\n\n### "
    return prompt
