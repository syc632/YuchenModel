"""逐位置展示教师如何为学生采样轨迹打分。

默认模型是一对共享 Qwen tokenizer 的本地或 Hugging Face 模型，因此可以直接
比较相同 token ID 上的 top-k 概率分布。该脚本只做推理与可视化，不会反向传播、
更新权重或写入检查点。

示例：
    python train/Post_train/OPD_score_demo.py
    python train/Post_train/OPD_score_demo.py --prompt "解释牛顿第二定律" --max-new-tokens 12
    python train/Post_train/OPD_score_demo.py --student D:/models/Qwen3-0.6B --teacher D:/models/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class PositionState:
    """一个蒸馏位置的完整、可打印状态。"""

    index: int
    response_prefix_ids: list[int]
    response_prefix_text: str
    sampled_token_id: int
    sampled_token_text: str
    student_sample_prob: float
    teacher_sample_prob: float
    topk_kl: float
    teacher_token_ids: list[int]
    candidate_texts: list[str]
    teacher_topk_probs: list[float]
    student_topk_probs: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="展示一条学生采样回答上的逐位置 OPD 打分状态。")
    parser.add_argument("--student", default="Qwen/Qwen3-0.6B", help="学生模型名或本地目录")
    parser.add_argument("--teacher", default="Qwen/Qwen3-1.7B", help="教师模型名或本地目录")
    parser.add_argument("--prompt", default="你是谁？", help="供学生采样的提示词")
    parser.add_argument("--max-new-tokens", type=int, default=12, help="学生最多采样多少个新 token")
    parser.add_argument("--top-k", type=int, default=8, help="每个位置展示的教师 top-k 数量")
    parser.add_argument("--temperature", type=float, default=1.0, help="学生采样温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="学生 nucleus sampling 的 top-p")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="模型运行设备")
    parser.add_argument("--trust-remote-code", action="store_true", help="仅在模型明确要求时启用")
    return parser.parse_args()


def resolve_device(option: str) -> torch.device:
    if option == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(option)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 未检测到可用 CUDA。")
    return device


def ensure_shared_vocab(student_tokenizer: Any, teacher_tokenizer: Any) -> None:
    """该演示直接按 token ID 比较概率，故师生必须使用同一词表。"""

    same_vocab = student_tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
    same_special_ids = (
        student_tokenizer.bos_token_id == teacher_tokenizer.bos_token_id
        and student_tokenizer.eos_token_id == teacher_tokenizer.eos_token_id
        and student_tokenizer.pad_token_id == teacher_tokenizer.pad_token_id
    )
    if not same_vocab or not same_special_ids:
        raise ValueError(
            "该可视化脚本要求师生共享完全相同的 tokenizer/词表。"
            "若词表不同，应使用文本投影 top-k KL，而不能直接比较 token ID。"
        )


def sample_response(
    model: Any,
    inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int | None,
) -> torch.Tensor:
    """让学生采样一条回答；该阶段不建立计算图。"""

    generation_args: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": eos_token_id,
    }
    if temperature > 0:
        generation_args.update({"temperature": temperature, "top_p": top_p})
    with torch.inference_mode():
        return model.generate(**inputs, **generation_args)


def contextual_extension(tokenizer: Any, prefix_ids: list[int], token_id: int) -> str:
    """获取 token 在当前前缀下真正新增的文本，避免孤立解码 BPE token。"""

    prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    full = tokenizer.decode(
        [*prefix_ids, token_id],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if full.startswith(prefix):
        return full[len(prefix) :]
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def token_probability(logits: torch.Tensor, token_id: int) -> float:
    """不构造完整 softmax 概率张量，直接计算某一 token 的概率。"""

    log_prob = logits[token_id].float() - torch.logsumexp(logits.float(), dim=-1)
    return log_prob.exp().item()


def collect_position_states(
    generated_ids: torch.Tensor,
    prompt_length: int,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tokenizer: Any,
    top_k: int,
) -> list[PositionState]:
    """收集回答每个位置的学生采样、教师 top-k 与局部 KL 状态。"""

    sequence = generated_ids[0].tolist()
    response_ids = sequence[prompt_length:]
    states: list[PositionState] = []

    for response_index, sampled_id in enumerate(response_ids):
        # 序列位置 p 的 logits 用于预测 p+1，因此第一个回答 token 的预测位置为 prompt_length - 1。
        logit_position = prompt_length + response_index - 1
        student_next = student_logits[logit_position]
        teacher_next = teacher_logits[logit_position]
        teacher_values, teacher_ids = torch.topk(teacher_next.float(), k=min(top_k, teacher_next.numel()))

        teacher_topk_log_probs = F.log_softmax(teacher_values, dim=-1)
        teacher_topk_probs = teacher_topk_log_probs.exp()
        student_topk_values = student_next.float()[teacher_ids]
        student_topk_log_probs = F.log_softmax(student_topk_values, dim=-1)
        student_topk_probs = student_topk_log_probs.exp()
        topk_kl = F.kl_div(student_topk_log_probs, teacher_topk_probs, reduction="sum").item()

        response_prefix_ids = response_ids[:response_index]
        response_prefix_text = tokenizer.decode(
            response_prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        teacher_ids_list = teacher_ids.cpu().tolist()
        states.append(
            PositionState(
                index=response_index,
                response_prefix_ids=response_prefix_ids,
                response_prefix_text=response_prefix_text,
                sampled_token_id=sampled_id,
                sampled_token_text=contextual_extension(tokenizer, response_prefix_ids, sampled_id),
                student_sample_prob=token_probability(student_next, sampled_id),
                teacher_sample_prob=token_probability(teacher_next, sampled_id),
                topk_kl=topk_kl,
                teacher_token_ids=teacher_ids_list,
                candidate_texts=[
                    contextual_extension(tokenizer, response_prefix_ids, token_id)
                    for token_id in teacher_ids_list
                ],
                teacher_topk_probs=teacher_topk_probs.cpu().tolist(),
                student_topk_probs=student_topk_probs.cpu().tolist(),
            )
        )
    return states


def print_position_state(state: PositionState) -> None:
    """清晰打印一个蒸馏位置的完整状态。"""

    print("\n" + "=" * 88)
    print(f"蒸馏位置: {state.index}")
    print(f"学生回答前缀文本: {state.response_prefix_text!r}")
    print(f"学生回答前缀 token IDs: {state.response_prefix_ids}")
    print(
        "学生实际采样: "
        f"id={state.sampled_token_id}, 文本={state.sampled_token_text!r}, "
        f"学生全词表概率={state.student_sample_prob:.6f}, "
        f"教师全词表概率={state.teacher_sample_prob:.6f}"
    )
    print(f"教师 top-k 子集上的 KL(教师 || 学生): {state.topk_kl:.6f}")
    print("教师候选（子集内概率已重新归一化）：")
    print(f"{'排名':<6}{'教师 token ID':<16}{'候选文本':<24}{'教师概率':<18}{'学生概率'}")
    for rank, (token_id, text, teacher_prob, student_prob) in enumerate(
        zip(
            state.teacher_token_ids,
            state.candidate_texts,
            state.teacher_topk_probs,
            state.student_topk_probs,
        ),
        start=1,
    ):
        print(f"{rank:<6}{token_id:<16}{text!r:<24}{teacher_prob:<18.6f}{student_prob:.6f}")


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0 or args.top_k < 2:
        raise ValueError("--max-new-tokens 必须大于 0，且 --top-k 至少为 2。")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("--temperature 必须非负，且 --top-p 必须在 (0, 1]。")

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"设备: {device}；精度: {dtype}")
    print(f"加载学生模型: {args.student}")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=args.trust_remote_code)
    student = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
    ).to(device).eval()

    print(f"加载教师模型: {args.teacher}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=args.trust_remote_code)
    ensure_shared_vocab(student_tokenizer, teacher_tokenizer)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
    ).to(device).eval()

    inputs = student_tokenizer(args.prompt, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = sample_response(
        student,
        inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=student_tokenizer.eos_token_id,
    )
    response_ids = generated_ids[0, prompt_length:].tolist()
    response_text = student_tokenizer.decode(response_ids, skip_special_tokens=True)
    print("\n" + "=" * 88)
    print(f"Prompt: {args.prompt!r}")
    print(f"学生完整采样 token IDs: {response_ids}")
    print(f"学生完整采样回答: {response_text!r}")

    # 两个模型都在同一条“prompt + 学生回答”轨迹上前向；没有梯度，也不会更新权重。
    with torch.inference_mode():
        student_logits = student(input_ids=generated_ids).logits[0]
        teacher_logits = teacher(input_ids=generated_ids).logits[0]
    states = collect_position_states(
        generated_ids,
        prompt_length,
        student_logits,
        teacher_logits,
        student_tokenizer,
        args.top_k,
    )
    for state in states:
        print_position_state(state)

    if states:
        mean_kl = sum(state.topk_kl for state in states) / len(states)
        print("\n" + "=" * 88)
        print(f"完成：共展示 {len(states)} 个蒸馏位置；平均 top-k KL = {mean_kl:.6f}")
    else:
        print("学生未生成可用于蒸馏的 token。")


if __name__ == "__main__":
    main()
