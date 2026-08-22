from pathlib import Path
import sys

import torch

# Allow both `python test/mla.py` and execution from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.MLA import Config, MLA

def test_kv_cache_matches_full_attention():
    torch.manual_seed(0)

    cfg = Config()
    model = MLA(cfg).eval()  # 关闭 dropout
    x = torch.randn(2, 16, cfg.d_model)

    with torch.no_grad():
        # 一次性计算完整序列
        full_output, _ = model(x)

        # 每次输入一个 token，并持续传递 cache
        kv_cache = None
        cached_outputs = []

        for t in range(x.size(1)):
            current_x = x[:, t:t + 1, :]
            output, kv_cache = model(current_x, cache=kv_cache)
            cached_outputs.append(output)

        cached_output = torch.cat(cached_outputs, dim=1)

    # 浮点计算顺序不同，允许小误差
    torch.testing.assert_close(
        cached_output,
        full_output,
        rtol=1e-3,
        atol=1e-3,
    )

    # cache 最终应保存完整序列
    # kv_latent: (B, L, C_kv); k_rope: (B, L, H, R)
    assert kv_cache[0].size(1) == x.size(1)
    assert kv_cache[1].size(1) == x.size(1)
    print("没问题")

if __name__ =="__main__":
    test_kv_cache_matches_full_attention()
