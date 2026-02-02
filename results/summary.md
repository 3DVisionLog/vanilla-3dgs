### 01_densify_test
3dgs-e1 500->258->5482
3dgs-e2 500->301->3968
densify를 끝까지 안하고 중간까지만 하니까 더 안정화된 모습

### 02_coordinate
카메라 좌표계가 LHS로 되어있어 불투명하게 속이 비친 모습으로 렌더링 됨. RHS로 수정.
추가로 train loop 속에서 아래 코드를 진행했는데
    c2w = datas[img_i]["c2w"].clone().to(device)
    c2w[0:3, 1:3] *= -1    
처음 7번 이미지가 뽑혔을 때는 -1이 곱해져서 정상적으로 +Z (OpenCV 좌표계)가 되는데....
학습이 진행되다가 나중에 똑같은 7번 이미지가 다시 뽑히면 이미 -1이 곱해져 있던 값에 다시 -1을 곱하게 됨..!!!
datas[img_i]["c2w"]가 리스트에 저장된 텐서의 메모리 주소를 가리켜서 원본 데이터를 건드는게 문제임
    c2w = datas[img_i]["c2w"].clone().to(device)
    c2w[0:3, 1:3] *= -1
이렇게 수정했다고 한다~

### 03_opacity_freqyency
opacity 초기화를 densify 끝날 때 한번만 했더니 렌더링하니까 불투명하고 그런게 많이 보였음
opacity 초기화를 1000 iter 마다 1번으로 변경함

### 04_rasterizer_opimization
numba 기반 rasterizer를 3dgs 공식 구현과 같은 c++/cu 기반으로 변경
학습 시간은 3:16:32 -> 18:04 ㄷㄷㄷㄷ

### 05_densify_start
0.1*iters 부터 시작하던 densify 과정을 일괄적으로 500 step에서 시작하는 것을 변경
간격도 100 iter로 조정!
3dgs-e6에 비해 점 개수 2만->8만으로 늘고 최종 학습 시간은 50분

### 06_ply_render
ply 저장 시 SH 값이 발산하는 문제 발생!!!! superspl.at 기준으로.. SH 밴드 값을 0을 하면 그래도 색 잘보임
뷰어에선 color가 0.5 + C0 * dc 이렇게 linear하게 표현 되는데 내 모델은 sigmoid(C0 * dc) 이렇게 표현하고 있었음
sigmoid가 적용된게 문제인 것 같아서 forward에서의 sigmoid를 제거하고 loss 계산 전 clamp를 처리하도록 하니까 고쳐짐

### 07_model_init
3dgs 공식 구현처럼 모델 초기화 과정에서 bin을 읽어 학습에 활용