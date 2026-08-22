from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer

ROOT = Path(r"D:\Kimi")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from SFT import SFTConfig, build_model

# 1. 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(ROOT / "BPEmodel")
model = build_model(tokenizer, SFTConfig())

state_dict = torch.load(
    ROOT / "train/sft_model/sft_weight_512.pth",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state_dict, strict=True)

# 当前模型内部存在 BF16 类型兼容问题，测评先使用 float32
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device=device, dtype=torch.float32).eval()

# 2. 修改这里测试不同问题
messages = [
    {"role": "user", "content": "告诉我你叫什么"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

input_ids = tokenizer(
    prompt,
    return_tensors="pt",
).input_ids.to(device)

generated = input_ids

max_new_tokens = 100
new_tokens = []

with torch.inference_mode():
    # 第一次处理完整 prompt，同时建立缓存
    output = model(input_ids=input_ids)
    cache = output.past_key_values

    for _ in range(max_new_tokens):
        next_id = output.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        new_tokens.append(next_id)

        if next_id.item() == tokenizer.eos_token_id:
            break

        # 后续每次只输入新生成的一个 token
        output = model(
            input_ids=next_id,
            cache=cache,
            logits_to_keep=1,
        )
        cache = output.past_key_values

answer_ids = torch.cat(new_tokens, dim=1)
print(tokenizer.decode(answer_ids[0], skip_special_tokens=True))

answer_ids = generated[0, input_ids.shape[1]:]
print(tokenizer.decode(answer_ids, skip_special_tokens=True))