## intro-llm-spark

推理/采样 (Inference/Rollout)  
logits → softmax → argmax → index  从概率分布中选择一个 token 来生成文本
训练 (Training/Optimization)
logits → log_softmax → gather → log_prob 计算出生成已经存在的 token 序列的对数概率
