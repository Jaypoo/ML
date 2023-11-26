import os
import cv2
import numpy as np

def load_data(data_dir='./Dataset'):
    # 클래스 레이블 정의
    class_labels = ['Mild_Demented', 'Non_Demented', 'Moderate_Demented', 'Very_Mild_Demented']

    # 데이터와 레이블을 저장할 리스트 초기화
    data = []
    labels = []

    for i, class_label in enumerate(class_labels):
        class_dir = os.path.join(data_dir, class_label)

        # 클래스 폴더 내의 파일 목록을 읽기
        file_list = os.listdir(class_dir)

        for file in file_list:
            # 파일 경로
            file_path = os.path.join(class_dir, file)

            # 이미지 읽기
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 흑백 이미지

            # 이미지 크기를 모델에 맞게 조절
            img = cv2.resize(img, (128, 128))

            # 이미지를 0에서 1 사이의 값으로 정규화
            img = img.astype(np.float32) / 255.0

            # 데이터와 레이블에 추가
            data.append(img)
            labels.append(i)

    # NumPy 배열로 변환
    data = np.array(data)
    labels = np.array(labels)

    # 데이터를 무작위로 섞은 후, 훈련 세트와 테스트 세트로 분할
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    split_idx = int(0.8 * len(data)) # 8:2 로 분할
    x_train, x_test = data[:split_idx], data[split_idx:]
    t_train, t_test = labels[:split_idx], labels[split_idx:]

    # 4차원 데이터로 변환
    x_train = x_train.reshape(-1, 1, 128, 128)
    x_test = x_test.reshape(-1, 1, 128, 128)

    return (x_train, t_train), (x_test, t_test)
