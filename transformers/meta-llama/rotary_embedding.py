class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预缓存 sin/cos，便于 torch.jit.trace
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    @staticmethod
    def rotate_half(x):
        """将最后一维一分为二并旋转: (x1, x2) -> (-x2, x1)"""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len=None):
        """
        参数:
            x: [bs, num_attention_heads, seq_len, head_dim]
            seq_len: 可选，默认取 x 的 seq_len 维度
        返回:
            (cos, sin): 形状均为 [1, 1, seq_len, dim]
        """
        if seq_len is None:
            seq_len = x.size(-2)

        # 需要时扩展缓存
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)

        cos = self.cos_cached[:, :, :seq_len, ...].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[:, :, :seq_len, ...].to(device=x.device, dtype=x.dtype)
        return cos, sin

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        """
        对 q,k 应用 RoPE。
        形状:
            q,k: [bs, num_heads, seq_len, head_dim]
            cos,sin: [1, 1, seq_len, dim]
            position_ids: [bs, seq_len]
        返回:
            (q_embed, k_embed): 与 q,k 同形状
        """
        # cos/sin 前两维为 1，压缩后按 position_ids 采样
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

        q_embed = (q * cos) + (LlamaRotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (LlamaRotaryEmbedding.rotate_half(k) * sin)
        return q_embed, k_embed
