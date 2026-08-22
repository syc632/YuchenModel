import os
import random
import json
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
from transformers import AutoTokenizer
random.seed(42)
def train_tokenizer():
    def read_file(file_path):
        with open(file_path,"r",encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield data["text"]



    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "data", "wiki_zh.jsonl")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"训练数据不存在: {data_path}\n"
            "请先准备 JSONL 文件，每行格式为: {\"text\": \"...\"}"
        )
    tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
    #中文不需要在每个序列前面加空格,add_prefix_space = false
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)


    #end_of_text:文本结束符,同时用作填充符和未知符
    #im_start:聊天信息开始符
    #end_start:聊天信息结束符
    special_token = ["<|endoftext|>","<|im_start|>","<|im_end|>"]


    #HuggingFace的训练器对象,驱动BPE算法的训练过程
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        min_frequency=2,
        special_tokens=special_token,
        show_progress=True,#展示训练进度
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()#初始字母表,可以避免OOV问题
    )
    data = read_file(data_path)
    #train_from_iterator方法从迭代器中读取数据进行训练
    tokenizer.train_from_iterator(data,trainer)
    tokenizer.decoder = decoders.ByteLevel()


    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2


    tokenizer_dir = os.path.join(project_dir, "BPEmodel")
    os.makedirs(tokenizer_dir,exist_ok=True)



    #保存分词器的核心配置
    tokenizer.save(os.path.join(tokenizer_dir,"tokenizer.json"))
    #保存BPE模型(生成vocab.json和merge.txt)
    tokenizer.model.save(tokenizer_dir)


    config= {
        "add_bos_token":False,
        "add_eos_token":False,
        "add_prefix_space":False,
        "added_tokens_decoder":{
            "0":{
                "content":"<|endoftext|>", #token的实际文本内容
                "lstrip":False, #解码时不在左边去除空格
                "rstrip":False, #解码时不在右边去除空格
                "single_word" :False, #不是单词语境
                "special":True #特殊token
            },
            "1":{
                "content":"<|im_start|>",
                "lstrip":False,
                "rstrip":False,
                "single_word":False,
                "special":True
            },
            "2":{
                "content":"<|im_end|>",
                "lstrip":False,
                "rstrip":False,
                "single_word":False,
                "special":True
            }
        },
        #额外的特殊token,当前为空
        "additional_special_tokens" : [],
        #在生成任务中,模型会用这个token作为序列的开始
        "bos_token":"<|im_start|>",
        #不去处理分词后的空格
        "clean_up_tokenization_space":False,
        #在生成任务中,模型会用这个token作为序列的结束
        "eos_token":"<|im_end|>",
        #是否使用旧版兼容模式
        "legacy":True,
        #模型支持的最大序列长度
        "model_max_length":32000,
        #当序列的长度不够的时候会使用padding来填充
        "pad_token":"<|endoftext|>",
        "spaces_between_special_tokens":False,
        #快速分类器
        "tokenizer_class":"PreTrainedTokenizerFast",
        "unk_token":"<|endoftext|>",
        # 注意：这个模板是 Jinja2 格式，支持条件判断、循环等复杂逻辑
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }
    with open(os.path.join(tokenizer_dir,"tokenizer_config.json"),"w",encoding="utf-8") as config_file:
        #ensure_ascii=False,允许保存中文字符(不转义为\uXXX)

        json.dump(config,config_file,ensure_ascii=False,indent=4)

        print("Tokenizer training completed and saved.")


def eval_tokenizer():
    """
    训练完后,用transformer加载并验证分词器

    验证内容:
    1.分词器加载
    2.聊天内容:验证聊天模板能否正确格式化多轮对话
    3.解编码的一致性
    4.词汇表大小
    """
    #分词器加载
    from transformers import PreTrainedTokenizerFast
    #使用Transformer的AutoTokenizer加载保存的分词器
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(project_dir, "BPEmodel")
    )


    #验证聊天内容
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    #apply_chat_template 会使用 tokenizer_config.json 中的 chat_template
    #将消息列表转换为模型输入格式
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False #tokenize=False: 直接返回格式化后的字符串，而不是 token ID 序列
                        #这样可以直观地看到格式化结果，便于验证模板是否正确
    )
    print(new_prompt)


    #验证词汇表大小和解编码一致性
    actual_vocab_size = len(tokenizer)
    print("tokenizer的实际词汇表大小:",actual_vocab_size)

    #编码测试,将文字转换为token id
    input = tokenizer(new_prompt)
    print("encoder长度:",len(input["input_ids"]))
    #输出编码后的token数量


    #解码测试,将token_id转换为文本
    output = tokenizer.decode(
        input["input_ids"],
        skip_special_tokens=False
    )
    #skip_special_tokens保留特殊的token


    print("decoder和原始的文本长度一致:",new_prompt == output)


def main():
    """
    主函数,执行分词的训练流程

    """
    train_tokenizer()
    eval_tokenizer()


if __name__ == "__main__":
    main()
