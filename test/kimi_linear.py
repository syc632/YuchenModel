from pathlib import Path
import sys

import torch

#同时支持在项目根目录运行和直接运行当前文件
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from model.KimiLinear import Config,KDALayer,KimiDeltaAttention,KimiLinear


def make_config():
    return Config(
        d_model=32,
        n_head=2,
        head_dim=8,
        gate_rank=8,
        n_layer=4,
        expan=2,
        kv_latent=12,
        q_latent=12,
        dropout=0.0,
    )


def test_kda_cache_matches_full_sequence():
    torch.manual_seed(0)
    model = KimiDeltaAttention(32,2,8,4,8).eval()
    x = torch.randn(2,11,32)
    padding_mask = torch.ones(2,11,dtype=torch.bool)

    with torch.no_grad():
        full_output,full_cache = model(x,padding_mask=padding_mask)

        cache = None
        cached_outputs = []
        for t in range(x.size(1)):
            output,cache = model(
                x[:,t:t+1],cache,padding_mask[:,t:t+1]
            )
            cached_outputs.append(output)
        cached_output = torch.cat(cached_outputs,dim=1)

    torch.testing.assert_close(cached_output,full_output,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(cache["state"],full_cache["state"],rtol=1e-5,atol=1e-5)


def test_kda_layer_has_the_same_simple_interface_as_delta_layer():
    model = KDALayer(32,2,expan=2,conv_size=4,head_dim=8).eval()
    x = torch.randn(1,6,32)
    output,cache = model(x)

    assert output.shape==x.shape
    assert cache["state"].shape==(1,2,8,8)


def test_padding_does_not_update_kda_state():
    torch.manual_seed(1)
    model = KimiDeltaAttention(32,2,8,4,8).eval()
    x = torch.randn(1,8,32)
    padding_mask = torch.tensor([[1,1,1,1,1,0,0,0]],dtype=torch.bool)

    with torch.no_grad():
        output,padded_cache = model(x,padding_mask=padding_mask)
        prefix_output,prefix_cache = model(
            x[:,:5],padding_mask=padding_mask[:,:5]
        )

    torch.testing.assert_close(output[:,:5],prefix_output,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(padded_cache["state"],prefix_cache["state"],rtol=1e-5,atol=1e-5)
    assert torch.count_nonzero(output[:,5:])==0


def test_kimi_linear_hybrid_cache_matches_full_sequence():
    torch.manual_seed(2)
    cfg = make_config()
    model = KimiLinear(cfg).eval()
    x = torch.randn(2,9,cfg.d_model)
    padding_mask = torch.ones(2,9,dtype=torch.bool)

    with torch.no_grad():
        full_output,_ = model(x,padding_mask=padding_mask)

        caches = None
        cached_outputs = []
        for t in range(x.size(1)):
            output,caches = model(
                x[:,t:t+1],caches,padding_mask[:,t:t+1]
            )
            cached_outputs.append(output)
        cached_output = torch.cat(cached_outputs,dim=1)

    torch.testing.assert_close(cached_output,full_output,rtol=1e-4,atol=1e-5)
    assert [layer.use_mla for layer in model.layers]==[False,False,False,True]
    assert caches[0]["state"].shape==(2,cfg.n_head,cfg.head_dim,cfg.head_dim)
    assert caches[3]["kv_latent"].shape==(2,9,cfg.kv_latent)


if __name__ == "__main__":
    test_kda_cache_matches_full_sequence()
    test_kda_layer_has_the_same_simple_interface_as_delta_layer()
    test_padding_does_not_update_kda_state()
    test_kimi_linear_hybrid_cache_matches_full_sequence()
    print("没问题")
