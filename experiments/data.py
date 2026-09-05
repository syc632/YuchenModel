"""Document-level dedup/split before tokenization; mmap token blocks without packing across documents."""
import hashlib
from contextlib import closing
import json
from pathlib import Path
import random
import sqlite3
import unicodedata

import numpy as np
import torch
from transformers import AutoTokenizer

from .config import digest, file_hash, read_json, write_json


def prepare(source, tokenizer_path, output, seed=2026):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"数据目录已存在，拒绝覆盖: {output}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if any(x is None for x in (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)):
        raise ValueError("tokenizer必须有pad/bos/eos")
    output.mkdir(parents=True)
    database = output/"dedup.sqlite"
    counts = {"input_documents": 0, "empty_documents": 0}
    try:
        with closing(sqlite3.connect(database)) as conn:
            conn.execute("CREATE TABLE docs (hash TEXT PRIMARY KEY, text TEXT NOT NULL)")
            with source.open() as f:
                for line_number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row.get("text"), str):
                        raise ValueError(f"第{line_number}行需要字符串text")
                    text = unicodedata.normalize("NFC", row["text"].replace("\r\n", "\n")).strip()
                    counts["input_documents"] += 1
                    if not text:
                        counts["empty_documents"] += 1
                        continue
                    h = hashlib.sha256(text.encode()).hexdigest()
                    conn.execute("INSERT OR IGNORE INTO docs VALUES (?, ?)", (h, text))
            conn.commit()
            hashes = [row[0] for row in conn.execute("SELECT hash FROM docs ORDER BY hash")]
            if len(hashes) < 100:
                raise ValueError("98/1/1文档切分至少需要100篇去重非空文档")
            random.Random(seed).shuffle(hashes)
            heldout = max(1, len(hashes)//100)
            splits = {"validation": hashes[:heldout], "test": hashes[heldout:2*heldout], "train": hashes[2*heldout:]}
            token_counts, doc_counts = {}, {}
            for split, ids in splits.items():
                offset, index = 0, []
                with (output/f"{split}.bin").open("wb") as binary:
                    for h in ids:
                        text = conn.execute("SELECT text FROM docs WHERE hash=?", (h,)).fetchone()[0]
                        tokens = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
                        values = np.asarray(tokens, dtype="<i4")
                        binary.write(values.tobytes())
                        index.append({"hash": h, "offset": offset, "length": len(tokens)})
                        offset += len(tokens)
                write_json(output/f"{split}.index.json", index)
                token_counts[split], doc_counts[split] = offset, len(ids)
        tokenizer.save_pretrained(output/"tokenizer")
        files = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p != database}
        metadata = {"version": 1, "data_seed": seed, "source_sha256": file_hash(source),
                    "normalization": "NFC, CRLF to LF, strip; exact document SHA256 dedup",
                    "document_counts": doc_counts, "token_counts": token_counts, **counts,
                    "vocab_size": len(tokenizer), "pad_token_id": tokenizer.pad_token_id,
                    "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
                    "files": files}
        metadata["fingerprint"] = digest(metadata)
        write_json(output/"manifest.json", metadata)
    finally:
        # 中间去重表仅用于准备数据，不保留原始文本的额外副本。
        database.unlink(missing_ok=True)
    return metadata


def verify_data(path):
    path = Path(path)
    metadata = read_json(path/"manifest.json")
    if digest({k: v for k, v in metadata.items() if k != "fingerprint"}) != metadata["fingerprint"]:
        raise ValueError("数据manifest指纹不匹配")
    for name, expected in metadata["files"].items():
        if file_hash(path/name) != expected:
            raise ValueError(f"准备数据或tokenizer被修改: {name}")
    return metadata


class Blocks:
    def __init__(self, path, split, seq_len, long_only=False):
        if seq_len < 2:
            raise ValueError("seq_len>=2")
        path = Path(path)
        self.values = np.memmap(path/f"{split}.bin", dtype="<i4", mode="r")
        self.blocks = []
        for doc in read_json(path/f"{split}.index.json"):
            if long_only and doc["length"] < seq_len:
                continue
            for start in range(0, doc["length"]-1, seq_len-1):
                length = min(seq_len, doc["length"]-start)
                if long_only and length < seq_len:
                    continue
                self.blocks.append((doc["offset"]+start, length))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        start, length = self.blocks[index]
        return torch.from_numpy(np.array(self.values[start:start+length], dtype=np.int64))


def collate(rows, pad, seq_len):
    ids = torch.full((len(rows), seq_len), pad, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = row
        mask[i, :len(row)] = True
    labels = ids.masked_fill(~mask, -100)
    labels[:, 0] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def target_count(batch):
    return int(batch["labels"][:, 1:].ne(-100).sum())


def cap_targets(batch, maximum):
    labels = batch["labels"].clone()
    valid = labels[:, 1:].ne(-100)
    keep = valid.reshape(-1).long().cumsum(0).reshape_as(valid) <= maximum
    labels[:, 1:].masked_fill_(~keep, -100)
    return {**batch, "labels": labels}


class BatchStream:
    """Order depends only on data_seed/epoch, independently of model RNG."""
    def __init__(self, dataset, batch_size, pad, seed, epoch=0, position=0):
        if not len(dataset):
            raise ValueError("空训练集")
        self.dataset, self.batch_size, self.pad, self.seed = dataset, batch_size, pad, seed
        self.epoch, self.position = epoch, position
        self._order()

    def _order(self):
        self.order = list(range(len(self.dataset)))
        random.Random(self.seed+self.epoch).shuffle(self.order)

    def next(self):
        rows = []
        for _ in range(self.batch_size):
            if self.position == len(self.order):
                self.epoch += 1
                self.position = 0
                self._order()
            rows.append(self.dataset[self.order[self.position]])
            self.position += 1
        return collate(rows, self.pad, self.dataset.seq_len)

    def state_dict(self):
        return {"epoch": self.epoch, "position": self.position}


def batches(dataset, batch_size, pad):
    for start in range(0, len(dataset), batch_size):
        yield collate([dataset[i] for i in range(start, min(len(dataset), start+batch_size))], pad, dataset.seq_len)
