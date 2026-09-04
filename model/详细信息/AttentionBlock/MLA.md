 **MLA（Multi-head Latent Attention）** 理解成一句话：

> **不直接缓存每个 Head 的 K/V，而是先把 K/V 压缩成一个共享 latent，推理时再从 latent 中恢复各 Head 需要的信息，从而大幅减少 KV Cache。**

下面按照一条 token 从输入到输出的完整路径来看。

![Image|331](https://images.openai.com/static-rsc-4/3jMr6cH8YAEWKRmTsfvM4Zwn7eXghbKVGj163l8Atuut7NTT8_UIUdnWVq1P5pLdE2ShEVxtBpPxLcPzZ_D5qkvE1RE9v8nS6w-T3fnhnJq0RA6R4vN6vs4obB9LOvVj-vXo5ML4gD9_DZ6xb7IZCx4_P48tUeAAVwTP9S_CSJ-qYRyiEkSTov-8LMSVzDCx?purpose=fullsize)

![Image|302](https://images.openai.com/static-rsc-4/__CXP6DEVa9dYdvQl_SAMKy48niiQPJtQ1wyb_Eds3XngG6OawCNCGjIGuqa1aUrO54LcQoDS8vMo8fZmUK13xl0BetyoNok-E6pN3oKKJ23Ihhlg-URSwa3lDijpnpbpvfFzKXBU7kkH7BLPAYyriV_8ygHcgmZ2UqQtaTOk5N77ztMkvUZw_7odlFbO3zc?purpose=fullsize)

![Image|310](https://images.openai.com/static-rsc-4/GQqjb4_b2YZgaxZ8XRlIIVa6S07mXuZSJELI_mJbgXgsp2x5ADO5vA7ACbbQV2MBhnL4Yp3URsuFaXSqRN-qEyEZRlbRRfIQQ33TseVRdsM6eLcAKu7B8dETvrYwNBkjrTvb-nHZukqWoSdS-mUzw696_2NVvg9CBKFpAaY5cuxFK-0lX051-BuUCaqvp8fr?purpose=fullsize)

## 1. 输入隐藏状态

假设 Transformer 当前层输入：
$$
[  
\mathbf h_t\in\mathbb R^{d}  
]

例如：

[  
d=4096  
]
$$
普通 MHA 会直接：
$$
[  
h_t\rightarrow Q,K,V  
]
$$
MLA 则把 **Q 路径** 和 **KV 路径** 分开处理。

---

# 2. Q 路径：先压缩，再升维

MLA 首先把 Query 压缩成一个低维 latent：
$$ 
\boxed{  c_t^Q=W^{DQ}h_t  } 
$$
其中：
$$  h_t\in\mathbb R^{d}  \rightarrow  c_t^Q\in\mathbb R^{d_c'}  $$
然后再升维：

$$ 
q_t^C=W^{UQ}c_t^Q  
$$

得到所有 Heads 的 content query：

$$ 
q_t^C  \rightarrow  q_{t,1}^C,\cdots,q_{t,H}^C  $$

也就是：

```text
h_t
 │
 ▼
Down Projection
 │
 ▼
c_t^Q
 │
 ▼
Up Projection
 │
 ▼
所有 Heads 的 q^C
 │
 ▼
Split Heads
 │
 ├── q_1^C
 ├── q_2^C
 ├── ...
 └── q_H^C
```

这里的 (C) 可以理解成 **Content**。

---

# 3. Q 的 RoPE 路径

MLA 不直接对整个 (q^C) 使用 RoPE。

而是另外生成：

$$
q_t^R=W^{QR}c_t^Q  
$$

然后分成多个 Head：

$$
q_t^R  
\rightarrow  
[q_{t,1}^R,\cdots,q_{t,H}^R]  
$$

再分别进行 RoPE：

$$
\tilde q_{t,i}^{R}

\operatorname{RoPE}(q_{t,i}^{R},t)  
$$

因此一个 Head 最终的 Query 是：

$$ 
\boxed{  
q_{t,i}

[q_{t,i}^{C};\tilde q_{t,i}^{R}]  
}  
$$

注意这里：

$$
\underbrace{q^C}_{不做RoPE}  
\quad+\quad  
\underbrace{q^R}_{做RoPE}  
$$

这就是 MLA 的 **Decoupled RoPE**。

---

# 4. KV 路径：这是 MLA 最关键的部分

输入同样是：

$$
h_t\in\mathbb R^d  
$$

但不会直接生成：

$$  
K_1,V_1,K_2,V_2,\cdots,K_H,V_H  
$$

而是先压缩：
$$ 
\boxed{  
c_t^{KV}

W^{DKV}h_t  
}  
$$

得到一个低维 latent：

$$
c_t^{KV}\in\mathbb R^{d_c}  
$$

例如：

```text
4096维 h_t
       │
       ▼
   Down Projection
       │
       ▼
512维 c_t^KV
```

这个：

$$
\boxed{c_t^{KV}}  
$$

就是 MLA 节省 KV Cache 的核心。

---

# 5. 从 KV latent 恢复 Content K/V

然后：

$$ 
k_t^C=W^{UK}c_t^{KV}  
$$

$$
v_t^C=W^{UV}c_t^{KV}  
$$

得到所有 Heads 的 K/V。

再 reshape：

$$  
k_t^C  
\rightarrow  
[k_{t,1}^C,\cdots,k_{t,H}^C]  
$$

$$  
v_t^C  
\rightarrow  
[v_{t,1}^C,\cdots,v_{t,H}^C]  
$$

因此：

```text
             c_t^KV
             /    \
            /      \
         W^UK      W^UV
          │          │
          ▼          ▼
      所有 K^C     所有 V^C
          │          │
      Split Head  Split Head
       / | ...     / | ...
     K1 K2 KH    V1 V2 VH
```

---

# 6. K 的 RoPE 路径

这里是 MLA 和普通 MHA 很不一样的地方。

另外从输入产生一个 RoPE Key：

$$ 
k_t^R=W^{KR}h_t  
$$

然后：

$$
\tilde k_t^R

\operatorname{RoPE}(k_t^R,t)  
$$

关键在于：

$$ 
\boxed{k_t^R\text{ 是所有 Heads 共享的}}  
$$

所以：

```text
Head 1 → [k_1^C ; k^R]
Head 2 → [k_2^C ; k^R]
Head 3 → [k_3^C ; k^R]
...
Head H → [k_H^C ; k^R]
```

而 Query 的 RoPE 部分是不共享的：

```text
Head 1 → [q_1^C ; q_1^R]
Head 2 → [q_2^C ; q_2^R]
...
Head H → [q_H^C ; q_H^R]
```

---

# 7. 拼接最终 Q/K

因此第 (i) 个 Head：

$$  
\boxed{  
q_{t,i}
=
[q_{t,i}^{C};  
\operatorname{RoPE}(q_{t,i}^{R})]  
}  
$$

而：

$$
\boxed{  
k_{t,i}
=
[k_{t,i}^{C};  
\operatorname{RoPE}(k_t^{R})]  
}  
$$

Value：
$$
\boxed{  
v_{t,i}=v_{t,i}^{C}  
}  
$$

可以画成：

```text
             Head i

Query                       Key
─────                       ────

q_i^C                       k_i^C
  │                           │
  │     不做 RoPE             │
  │                           │
  ▼                           ▼
┌────────┐                  ┌────────┐
│ q_i^C  │                  │ k_i^C  │
├────────┤                  ├────────┤
│RoPE(q_i^R)│               │RoPE(k^R)│ ← Heads共享
└────────┘                  └────────┘
     │                           │
     └──────────┬────────────────┘
                ▼
             Q_i K_i^T
```

---

# 8. 正常计算 Attention

到这里之后，就和普通 Multi-Head Attention 很像了。

第 (i) 个 Head：

$$
S_i=  
\frac{Q_iK_i^T}{\sqrt{d_h}}  
$$

然后：

$$
A_i=\operatorname{Softmax}(S_i)  
$$

再：

$$
O_i=A_iV_i  
$$

所有 Head 拼接：

$$
O=  
\operatorname{Concat}(O_1,\cdots,O_H)  
$$

最后经过输出投影：

$$
\boxed{  
Y=W^OO  
}  
$$

---

# 9. 真正关键：KV Cache 缓存什么？

这才是理解 MLA 的核心。

普通 MHA 推理的时候需要保存：

$$ 
K_1,V_1,K_2,V_2,\cdots,K_H,V_H  
$$

每来一个 token 都要缓存所有 Head 的 K/V。

因此缓存量大致：

$$
\boxed{2Hd_h}  
$$

---

MLA 不需要保存完整的：
$$
K_1,V_1,\cdots,K_H,V_H  
$$

而主要保存：

$$
\boxed{  
c_t^{KV}  
}  
$$

以及位置相关的：

$$
\boxed{  
k_t^R  
}  
$$

所以每个 token 的缓存规模变成：

$$
\boxed{  
d_c+d_h^R  
}  
$$

这就是为什么 MLA 可以显著降低 KV Cache。

---

# 10. 把整个 MLA 串起来

完整流程可以压缩成下面这张图：

```text
                         h_t
                    ┌─────┴─────┐
                    │           │
                 Q Path       KV Path
                    │           │
              Down Project   Down Project
                    │           │
                    ▼           ▼
                  c^Q         c^KV  ← ★ KV Cache核心
                    │          / \
              ┌─────┴───┐     /   \
              │         │   W^UK  W^UV
              ▼         ▼     │      │
            q^C        q^R    ▼      ▼
              │         │    k^C    v^C
         Split Head Split Head │      │
              │         │      │      │
              │       RoPE     │      │
              │         │      │      │
              ▼         ▼      ▼      ▼
Head i:      q_i^C   q_i^R   k_i^C   v_i^C
               \       /        │
                \     /         │
                 CONCAT         │
                    │           │
                    q_i         │
                                │
h_t ───────→ W^KR ──→ k^R ─→ RoPE
                         │
                         │ 所有Heads共享
                         ▼
                   [k_i^C ; k^R]
                         │
                         k_i

                    q_i · k_i^T
                         │
                      Softmax
                         │
                         × v_i
                         │
                      Head_i
                         │
              Concatenate Heads
                         │
                        W^O
                         │
                         ▼
                       Output
```

## 最需要抓住的 4 点

1. **Q 和 KV 都可以经过低秩压缩**：  
    $$
    h\rightarrow c^Q,\qquad h\rightarrow c^{KV}  
    $$
    
2. **KV latent 是 MLA 的核心**：  
    $$
    c^{KV}\rightarrow K^C,V^C  
    $$ 
    推理时主要缓存 (c^{KV})，而不是完整的多头 K/V。
    
3. **RoPE 与 Content 解耦**：  
    $$
    Q=[Q^C;Q^R],\qquad K=[K^C;K^R]  
    $$ 
    只有 (Q^R,K^R) 做 RoPE。
    
4. **K 的 RoPE 部分跨 Head 共享**，从而避免为了位置编码重新引入巨大的 KV Cache。
    

所以你可以把 MLA 的设计思想记成：

$$
\boxed{  
\text{MLA}

\text{KV低秩压缩}  
+  
\text{Decoupled RoPE}  
+  
\text{Multi-Head Attention}  
}  
$$

而它最终解决的核心工程问题就是：

$$ 
\boxed{\text{在尽量保持 MHA 表达能力的同时，大幅压缩 KV Cache}}  $$