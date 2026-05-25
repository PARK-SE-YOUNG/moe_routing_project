import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    """기존 Baseline 방식: 가중치 기반 Top-k Softmax 라우터"""
    def __init__(self, input_dim, num_experts, top_k=1):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: [batch_size, input_dim] (토큰들의 임베딩 벡터)
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # 상위 K개의 전문가 선택
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 확률 재정규화
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_indices

class RLRouter(nn.Module):
    """강화학습 코딩 담당자가 구현할 Constrained RFT 라우터의 껍데기"""
    def __init__(self, input_dim, num_experts):
        super().__init__()
        # TODO: 강화학습 에이전트의 Policy 네트워크가 이 자리에 들어옵니다.
        self.policy = nn.Linear(input_dim, num_experts) 

    def forward(self, x):
        # 지금은 에러가 나지 않도록 임시로 Softmax 형태로 반환 구조만 맞춰둡니다.
        logits = self.policy(x)
        probs = F.softmax(logits, dim=-1)
        indices = torch.argmax(probs, dim=-1, keepdim=True)
        return probs, indices