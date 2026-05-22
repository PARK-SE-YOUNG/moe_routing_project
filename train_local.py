
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import torch
from models.base_moe import SparseMoELayer
from datasets.dummy_loader import get_local_dataloader

def run_baseline_test():
    print("=== [Baseline] 로컬 파이프라인 테스트 시작 ===")
    
    # 하이퍼파라미터 세팅
    d_model = 64
    d_ff = 128
    num_experts = 4
    
    # 1. 모델 및 데이터 로더 선언
    model = SparseMoELayer(d_model, d_ff, num_experts=num_experts, top_k=1)
    dataloader = get_local_dataloader(batch_size=8)
    
    # 2. 메트릭 측정을 위한 변수 초기화
    start_time = time.time()
    expert_counts = torch.zeros(num_experts)
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # 추론 수행
            outputs, indices = model(batch)
            
            # 메트릭 1: 각 전문가별로 처리한 토큰 개수 누적 (Expert Load)
            for i in range(num_experts):
                expert_counts[i] += (indices == i).sum().item()
                
    latency = time.time() - start_time
    
    # 3. 결과 출력 (실험/보고서 담당자에게 제공할 데이터)
    print("\n=== 실험 결과 (Metrics) ===")
    print(f"Total Inference Time (Latency): {latency:.4f} seconds")
    for i in range(num_experts):
        print(f" - Expert {i+1}이 처리한 토큰 수: {int(expert_counts[i].item())}개")
        
    print("\n[성공] 로컬 Baseline 구조가 정상적으로 작동합니다. 팀원들에게 공유 가능합니다.")

if __name__ == "__main__":
    run_baseline_test()