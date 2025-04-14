import re

# 임계값 설정 tau x
# delta1_threshold_arr = [0.946, 0.979, 0.984, 0.754, 0.978]
# abs_relative_difference_threshold_arr = [0.076,0.044,0.050,0.250,0.043]

# 임계값 설정 tau 3
# delta1_threshold_arr = [0.95, 0.984, 0.983, 0.759, 0.983]
# abs_relative_difference_threshold_arr = [0.076,0.04,0.052,0.225,0.041]
data_checkpoint = []
# data_name_arr = ['kitti_eigen_test','nyu_test', 'eth3d', 'diode', 'scannet']


delta1_threshold_arr = [0.938]
abs_relative_difference_threshold_arr = [0.081]
data_name_arr = ['kitti_eigen_test']
for name, delta1_threshold, abs_relative_difference_threshold, in zip(data_name_arr, delta1_threshold_arr,abs_relative_difference_threshold_arr):
    # 파일 경로 설정 tau x
    # file_path = f'/home/wodon326/project/AsymKD_VIT_Adapter_large/output/{name}/eval_metrics-bfm-ddp.txt'

    # 파일 경로 설정 tau 3
    file_path = f'output/{name}/eval_metrics-Efficient_BriGeS_residual-ddp.txt'
    # 체크포인트를 저장할 리스트
    checkpoints = []

    # 파일을 열고 한 줄씩 읽으면서 처리합니다.
    with open(file_path, 'r') as file:
        current_checkpoint = None
        delta1_acc = None
        abs_relative_difference = None
        
        for line in file:
            # 체크포인트를 추출합니다.
            if 'AsymKD' in line:
                current_checkpoint = re.search(r'(\d+)_AsymKD_new_loss', line)
                if current_checkpoint:
                    current_checkpoint = current_checkpoint.group(1)
            
            # delta1_acc 값을 추출합니다.
            if 'delta1_acc' in line:
                delta1_acc = float(re.search(r'delta1_acc\s*:\s*(\d+\.\d+)', line).group(1))
            
            # abs_relative_difference 값을 추출합니다.
            if 'abs_relative_difference' in line:
                abs_relative_difference = float(re.search(r'abs_relative_difference\s*:\s*(\d+\.\d+)', line).group(1))
            
            # 두 값이 모두 추출되었으면 조건에 맞는지 확인하고, 맞으면 리스트에 추가합니다.
            if delta1_acc is not None and abs_relative_difference is not None:
                if delta1_acc >= (delta1_threshold) and abs_relative_difference <= (abs_relative_difference_threshold):
                    # print(delta1_acc,abs_relative_difference)
                    checkpoints.append(current_checkpoint)
                # 다음 체크포인트를 위해 초기화합니다.
                delta1_acc = None
                abs_relative_difference = None

    data_checkpoint.append(checkpoints)
    # print(f'{name} : {len(checkpoints)} : {checkpoints}')
    print(f'{name} : {len(checkpoints)}')
    print(checkpoints)


common_elements = set(data_checkpoint[0])

# 나머지 배열들과의 교집합 구하기
for array in data_checkpoint[1:]:
    common_elements = common_elements.intersection(array)

# 결과 출력
print("교집합:", common_elements)