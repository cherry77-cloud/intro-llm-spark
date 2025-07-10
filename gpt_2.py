📝 文本输入 "Hello world!"
    ↓
🔪 正则分词 ["Hello", " world", "!"]  
    ↓
🔄 逐token处理：
    ├─ UTF-8编码: "Hello" → [72,101,108,108,111]
    ├─ 字节映射: bytes → unicode安全字符
    ├─ BPE合并: 按优先级迭代合并字符对
    └─ ID转换: BPE tokens → 数字ID
    ↓
🎯 输出 token IDs [15496, 995, 0]

解码时完全逆向操作：IDs → BPE → bytes → UTF-8 → 原文本


class Encoder:
   """GPT-2 字节级 BPE 编码器
   核心功能：将任意文本转换为模型可处理的 token ID 序列
   工作原理：文本 → 正则分词 → 字节级 Unicode 映射 → BPE 合并 → token ID
   """
   
   def __init__(self, encoder, bpe_merges, errors='replace'):
       """初始化编码器
       
       工作流程：
           1. 构建双向映射表：token ↔ ID, byte ↔ unicode
           2. 建立 BPE 优先级排序：早期合并规则优先级更高
           3. 编译正则表达式：用于智能分词
           4. 初始化缓存：避免重复 BPE 计算
       """
       self.encoder = encoder                    # token → ID 映射
       self.decoder = {v:k for k,v in encoder.items()}  # ID → token 映射
       self.byte_encoder = bytes_to_unicode()    # byte → unicode 映射
       self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}  # 反向映射
       self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))  # 合并优先级
       self.cache = {}                          # BPE 结果缓存
       # 正则模式：处理缩写、字母、数字、标点、空白
       self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

   def bpe(self, token):
       """对单个 token 执行字节对编码
       
       算法流程：
           1. 检查缓存，避免重复计算
           2. 将 token 转为字符元组
           3. 迭代合并：
              - 找出所有相邻字符对
              - 选择优先级最高的对进行合并
              - 更新 token，重新计算字符对
              - 重复直到无法继续合并
           4. 缓存结果并返回
       """

   def encode(self, text):
       """将文本编码为 token ID 序列
       """

   def decode(self, tokens):
       """将 token ID 序列解码为文本
       
       逆向工作流程：
           输入: [15496, 995, 0, 20204]
           1. ID 转 token: [15496, 995, 0, 20204] → ["Hello", "Ġworld", "!", "Ġ你好"]
           2. 拼接 BPE tokens: "HelloĠworld!Ġ你好"
           3. Unicode 转字节: 每个字符 → 对应字节值
           4. UTF-8 解码: 字节数组 → "Hello world! 你好"


def top_k_logits(logits, k):
    if k == 0:
        return logits
    
    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)    # 核心操作：获取前 k 个最大的 logit 值
        min_values = values[:, -1, tf.newaxis]  # 获取第 k 大的值作为截断阈值
        return tf.where(
            logits < min_values,           # 条件：logit值 < 第k大的值
            tf.ones_like(logits) * -1e10,
            logits,                       
        )
    
    # 动态条件执行：避免不必要的计算
    return tf.cond(
        tf.equal(k, 0),     # 如果k=0
        lambda: logits,     # 直接返回原logits
        lambda: _top_k(),   # 否则执行top-k截断
    )


def top_p_logits(logits, p):
    """Nucleus sampling - 动态截断策略"""
    batch, _ = logits.shape.as_list()
    
    # 步骤1：按概率从大到小排序
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    
    # 步骤2：计算排序后的累积概率分布
    cumulative_probs = tf.cumsum(
        tf.nn.softmax(sorted_logits, axis=-1), axis=-1
    )
    
    # 步骤3：找到累积概率刚好≤p的最后一个位置
    indices = tf.stack([
        tf.range(0, batch),  # 批次索引 [0, 1, 2, ..., batch_size-1]
        # 关键计算：有多少个token的累积概率≤p
        tf.maximum(
            tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 
            0
        ),
    ], axis=-1)
    
    # 步骤4：获取截断阈值
    min_values = tf.gather_nd(sorted_logits, indices)
    
    # 步骤5：应用截断
    return tf.where(
        logits < min_values,           # 原始logit < 阈值
        tf.ones_like(logits) * -1e10,  # 设为负无穷
        logits,                        # 保持原值
    )

# 温度缩放（Temperature Scaling）
# ──────────────────────────────────────────────────────────────────────────────────
# 通过除以温度参数T来调节概率分布的"尖锐程度"
#   • T > 1.0 → logits缩小 → 概率分布更平滑 → 增加随机性和创造力
#   • T < 1.0 → logits放大 → 概率分布更尖锐 → 趋向确定性选择  
#   • T = 1.0 → 保持原始分布不变
# 数学原理：P(token_i) = exp(logit_i/T) / Σ exp(logit_j/T)
logits = next_outputs['logits'][:, -1, :] / tf.to_float(temperature)

# ② Top-K 截断采样（Top-K Sampling）
# ──────────────────────────────────────────────────────────────────────────────────
# 只保留概率最高的前k个候选token，其余全部屏蔽：
#   • 找出前k个最大的logit值，以第k大值为阈值
#   • 将小于阈值的位置设为-1e10（相当于概率≈0）
#   • k=0表示不进行截断，保留所有候选
# 作用：防止模型选择明显不合理的低概率词汇
logits = top_k_logits(logits, k=top_k)

# ③ Top-P/核采样（Nucleus Sampling）  
# ──────────────────────────────────────────────────────────────────────────────────
# 动态选择累积概率刚好达到p的最小token集合：
#   • 将token按概率从高到低排序
#   • 找到使累积概率 ≥ p 的最小前缀集合
#   • 只保留这个"核心候选集"，其余设为-1e10
#   • p=1.0表示不进行截断
# 优势：根据概率分布形状自适应调整候选数量
logits = top_p_logits(logits, p=top_p)

# 多项式随机采样（Multinomial Sampling）
# ──────────────────────────────────────────────────────────────────────────────────
# 从经过筛选的候选池中按概率随机选择一个token：
#   • TensorFlow内部先对logits进行softmax归一化
#   • 然后按多项式分布随机抽样num_samples=1个token
#   • 每次调用结果可能不同，体现真正的随机性
# 替代方案：tf.argmax实现贪心解码（总是选概率最高的）
samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)


def body(past, prev, output):
    # 【自回归核心】：基于当前序列预测下一个token
    next_outputs = step(hparams, prev, past=past)
    logits = next_outputs['logits'][:, -1, :]  # 只取最后位置的预测
    
    # ... 采样策略 ...
    
    # 【关键的自回归状态更新】：
    return [
        updated_past,                              # 历史状态累积
        samples,                                   # 当前输出 → 下一步输入
        tf.concat([output, samples], axis=1)       # 序列逐步构建
    ]

_, _, tokens = tf.while_loop(
    cond=cond,                       # 循环条件
    body=body,                       # 每次迭代执行的自回归步骤
    maximum_iterations=length-1,     # 生成length-1个新token
    loop_vars=[past, prev, output],  # 循环变量：历史、当前、输出
)


def multihead_attn(q, k, v):
    # 计算注意力分数：每个位置对其他位置的"关注度"
    w = tf.matmul(q, k, transpose_b=True)  # [batch, heads, seq_len, seq_len]
    w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))  # 缩放防止梯度消失
    
    # 【关键】因果掩码：确保只能看到"过去"，不能看到"未来"
    w = mask_attn_weights(w)  # 上三角设为-∞，下三角保留
    w = softmax(w)            # 转为概率分布
    
    # 加权聚合：基于注意力权重整合信息
    a = tf.matmul(w, v)       # 最终的上下文表示
    return a

def mask_attn_weights(w):
    # 生成因果掩码：位置i只能关注位置≤i的信息
    b = attention_mask(nd, ns, dtype=w.dtype)  # 下三角矩阵
    w = w*b - tf.cast(1e10, w.dtype)*(1-b)    # 未来位置设为极小值
    return w


def attn(x, scope, n_state, *, past, hparams):
    # ① 计算当前步的Q、K、V
    c = conv1d(x, 'c_attn', n_state*3)
    q, k, v = map(split_heads, tf.split(c, 3, axis=2))
    
    # 【核心优化】保存当前的K、V用于下次缓存
    present = tf.stack([k, v], axis=1)  # 新的缓存状态
    
    # 【效率关键】如果有历史缓存，直接拼接使用
    if past is not None:
        pk, pv = tf.unstack(past, axis=1)  # 取出缓存的K、V
        k = tf.concat([pk, k], axis=-2)    # 历史K + 当前K
        v = tf.concat([pv, v], axis=-2)    # 历史V + 当前V

    a = multihead_attn(q, k, v)
    a = merge_heads(a)
    a = conv1d(a, 'c_proj', n_state)
    return a, present
