## 📌 Hugging Face Transformers 오픈소스 기여 (Issue #35290)

### 🔹 1. 개요
- Hugging Face의 `transformers` 라이브러리에 Pull Request 제출
- GPT-2 모델에서 발생하던 **어텐션 마스크(Attention Mask) 관련 버그 수정**
- 최신 PyTorch SDPA(Scaled Dot Product Attention) 연산 적용 과정에서 발생한 **차원 변형 문제 해결**
- [Issue 링크](https://github.com/huggingface/transformers/issues/35290)

### 🔹 2. 문제 정의
- GPT-2 모델에서 `attention_mask`를 4D 텐서로 전달했지만, 내부적으로 2D로 변형되는 문제가 발생
- 모델이 `_prepare_4d_causal_attention_mask_for_sdpa()` 함수를 호출하면서 차원 조건에 의해 `to_4d()` 함수가 실행 됨
- 이로 인해 **출력 차원이 예상과 다르게 변형되어, 모델이 오류를 발생**시킴

### 🔹 3. 해결 과정
   1️⃣ **이슈 분석**
   - 실제 실행 환경에서 **`attention_mask`의 차원이 4D → 2D로 변형되는 과정**을 디버깅  
   - GPT2Model 내부 `forward()`에서 `attention_mask.view(batch_size, -1)` 연산을 수행하면서 문제가 발생  
   - `_prepare_4d_causal_attention_mask_for_sdpa()` 내부 `to_4d()` 함수가 잘못 호출되어 `attention_mask` 차원이 의도와 다르게 변형되는 부분 발견  

   2️⃣ **해결 방법**
   - `attention_mask.view(batch_size, -1)`가 실행되기 전, `attention_mask.dim() == 2`인지 확인하는 조건 추가  
   - `attention_mask`가 4D인 경우, 변형 없이 그대로 사용하도록 수정  

   3️⃣ **테스트 & 검증**
   - 변경된 코드에서 기존 오류가 해결되는지 검증  
   - GPT-2 모델의 attention mask 연산이 정상적으로 수행됨을 확인  

### 🔹 4. 결과 및 성과
   ✅ PR 제출했지만, 동일한 문제를 먼저 해결한 PR이 이미 머지된 상태
   ✅ GPT-2 모델의 어텐션 마스크 처리 로직이 정상적으로 동작하도록 개선  
   ✅ 오픈소스 코드 구조를 더 깊게 이해하는 기회가 되었으며, PR 리뷰 과정에서 **더 나은 코드 스타일을 학습**  

### 🔹 5. 배운 점 & 적용할 수 있는 부분
- **대규모 오픈소스 프로젝트에서 Pull Request 기여하는 과정**을 직접 경험  
- **Attention Mask와 PyTorch SDPA 연산의 내부 처리 방식**을 깊게 이해  
- **PR 리뷰 피드백을 반영하면서 코드 품질을 높이는 방법 학습**  
