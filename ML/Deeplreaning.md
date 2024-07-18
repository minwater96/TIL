# TensorFlow 수업 정리

### TensorFlow 설치

```bash
pip install tensorflow
```

### TensorFlow commend

```python
import tensorflow as tf
from tensorflow import keras
```

- 데이터 불러오기

```python
# 데이터 불러오기
(train_input, train_target), (test_input, test_target) = \
keras.datasets.fashion_mnist.load_data()
```

- 데이터 확인 및 차원축소

```python
# 이미지 파일 확인
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize=(10, 10))

for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')

plt.show()

# 데이터 범주 확인
import numpy as np

np.unique(train_target, return_counts=True)

# (흑백)이미지 데이터 차원축소
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28) # 2차원 사진 1차원으로 바꾸기
```

## DNN 모델 만들기

Deep Neural Network (DNN) 모델을 만드는 기본 코드

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 모델 나누기
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2)

# 모델 정의
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,)) #(뉴런 개수, 활성함수='', 입력값 크기())
model = keras.Sequential([dense])
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model 학습
model.fit(train_scaled, train_target, epochs=5)

# 성능검증
model.evaluate(val_scaled, val_target)
```

## Hidden layer 추가하기

```python
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784, ), name='hidden') # 은닉층(hidden layer) 만들기
dense2 = keras.layers.Dense(10, activation='softmax', name='output') # 출력층
model = keras.Sequential([dense1, dense2])

# 함수를 활용한 layer 만들기
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784, )))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model 학습
model.fit(train_scaled, train_target, epochs=5)
```

## Optimizer, activation(relu) 추가

```python
# activation add
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28))) #입력층
model.add(keras.layers.Dense(100, activation='relu')) #은닉층
model.add(keras.layers.Dense(10, activation='softmax'))

# optimizer add
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

# Dropout 추가 및 Callback 설정

```python
# dropout add

# 모델생성 자동화 함수
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))

    if a_layer:
        model.add(a_layer)

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
    
# 은닉층을 추가하고, dropout 30% 실행
model = model_fn(keras.layers.Dropout(0.3))
model.summary()
```

```python
# callback 설정

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# callback 설정
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    save_best_only=True #내가 동작했던 모델 중 가장 best model을 저장해달라는 명령어
)

history = model.fit(train_scaled,
                    train_target,
                    epochs=20,
                    verbose=0,
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb]
                    )
```

- 위 코드는 Dropout layer를 추가해 모델의 overfitting을 방지하고, EarlyStopping callback을 설정해 validation loss가 5 epoch 동안 개선되지 않으면 학습을 중단합니다.