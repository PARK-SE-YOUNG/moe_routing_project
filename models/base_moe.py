import torch
import torch.nn as nn
from models.router import TopKRouter

class DummyExpert(nn.Module):
    """보고서의 FFNN 1~4 역할을 하는 경량 전문가 네트워크"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

class SparseMoELayer(nn.Module):
    """전체 MoE 레이어 구조"""
    def __init__(self, d_model, d_ff, num_experts=4, top_k=1):
        super().__init__()
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([DummyExpert(d_model, d_ff) for _ in range(num_experts)])
        
    def forward(self, x):
        # x 차원: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # [batch_size * seq_len, d_model]
        
        # 1. 라우터를 통해 어느 전문가로 갈지와 가중치 계산
        weights, indices = self.router(x_flat)
        
        # 2. 결과 출력을 담을 텐서 초기화
        output = torch.zeros_like(x_flat)
        
        # 3. 각 토큰을 선택된 전문가에게 전달하여 연산 (동적 라우팅 구현)
        for i, expert in enumerate(self.experts):
            # 현재 전문가(i)가 선택된 토큰들의 마스크 생성
            mask = (indices == i).any(dim=-1)
            if mask.any():
                token_inputs = x_flat[mask]
                expert_outputs = expert(token_inputs)
                # 가중치를 곱해서 결과에 누적
                output[mask] += expert_outputs * weights[mask]
                
        return output.view(batch_size, seq_len, d_model), indices