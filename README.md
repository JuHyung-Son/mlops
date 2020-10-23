---
# theme: gaia
# _class: lead
paginate: true
# backgroundColor: #fff
# backgroundImage: url('https://marp.app/assets/hero-background.jpg')
marp: true
---

![bg left:40% 80%](https://avatars3.githubusercontent.com/u/20140126?s=460&v=4)

# **발표자**

손주형,
네이버 웹툰 ML 엔지니어 :computer:

**관심사**
ml, cloud, tensorflow, onnx :thumbsup:

[github](https://github.com/jusonn)
[linkedIn](https://www.linkedin.com/in/juhyung-son-897b28130/)

---
# mlops로 ml 자동화하기

1. mlops
    - mlops란
    - 이전까지의 ml

---
# mlops로 ml 자동화하기

1. mlops
    - mlops란
    - 이전까지의 ml
2. mlops 컴포넌트
    - 데이터
        - 주입
        - 검증
        - 전처리
    - 모델
        - 학습
        - 검증
    - 서빙
---
# mlops로 ml 자동화하기

1. mlops
    - mlops란
    - 이전까지의 ml
2. mlops 컴포넌트
    - 데이터
        - 주입
        - 검증
        - 전처리
    - 모델
        - 학습
        - 검증
    - 서빙
3. 파이프라인
___
![bg right:47% 80%](assets/mlpipe.png)
# mlops로 ml 자동화하기

1. mlops
    - mlops란
    - 이전까지의 ml
2. mlops 컴포넌트
    - 데이터
        - 주입
        - 검증
        - 전처리
    - 모델
        - 학습
        - 검증
    - 서빙
3. 파이프라인
---
# mlops란
> MLOps is a practice for collaboration and communication between data scientists and operations professionals to help **manage production ML lifecycle.**

![](assets/lifecycle.png)

___

# mlops란
> MLOps is a practice for collaboration and communication between data scientists and operations professionals to help **manage production ML lifecycle.**

**ml + ops**
- mlops, ml 파이프라인으로 불리는 중
- 다양한 오픈소스 (tfx, kubeflow, mlflow...)
- 최근들어 중요성이 부각됨
    - 연구, 비즈니스 모델 검증 위주 ➡️ 서비스 적용, 고도화
---
# mlops란
- 연구, 비즈니스 모델 검증 위주 ➡️ 서비스 적용, 고도화
![](assets/debt.png)
[Hidden Technical Debt in Machine Learning Systems, 2015 NIPS](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)

---
# mlops란
- 연구, 비즈니스 모델 검증 위주 ➡️ 서비스 적용, 고도화
![](assets/mlops2.png)

---
# 이전까지의 ml
1. 데이터 준비

```bash
$ ssh ~~~~
$ mv /data/root/dataset ./
```

```bash
$ python preprocessing.py

Processing: 100%|██████████████████████| 352M/352M [80:14<00:00, 30.2it/s]
```

---
# 이전까지의 ml
2. 학습

```bash
$ nvidia-smi
```

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python train.py --data_path ./data --epoch 100 --save_path ./saved_model

training...
epoch 1: 100%|██████████████████████| 1000/1000 [00:14<00:00, 30it/s]
epoch 2:  42%|█████████▎            | 428/1000 [00:14<00:19, 28it/s]
```

---
# 이전까지의 ml
3. 모델 테스트


```bash
$ python test.py --data_path ./test_data --model_path ./saved_model

testing...
model 14: accuracy 90%
```

```bash
$ mv saved_model/14 serving_model/
```

--- 
# mlops에서는?
![bg right:65% 80%](assets/model-life-cycle.png)
- 데이터
    - 주입
    - 검증
    - 전처리
- 모델
    - 학습
    - 검증
- 서빙

---
# mlops에서는?

![bg right:46% 80%](assets/dag.png)
프로젝트에 따라 필요한 컴포넌트만 사용

---
# TFX

- 텐서플로우 생태계 구성 중 프로덕션을 위한 [플랫폼](https://www.tensorflow.org/tfx?hl=ko)
- ML 파이프라인을 구성하는 컴포넌트 제공
- MetadataStore(DB), tfdv, tft 등 tf 라이브러리 포함
- jupyter notebook을 통한 interactive 파이프라인 기능 제공
- 정식 릴리즈 0.24

---
# TFX
- ML 파이프라인을 구성하는 컴포넌트를 제공
![](assets/components.png)

---
# TFX
- 컴포넌트는 MetadataStore(DB)를 통해 통신
- MetadataStore를 이용해 모델 혹은 컴포넌트 성능 비교 가능
![](assets/inside-tfx.png)
---
# TFX
**비슷한 오픈소스** 

[AeroSolve](https://github.com/airbnb/aerosolve) (airbnb)
[Railyard](https://stripe.com/blog/railyard-training-models) (Stripe)
[Luigi](https://github.com/spotify/luigi) (spotify)
[Michelangelo](https://eng.uber.com/michelangelo-machine-learning-platform/) (Uber)
[Metaflow](https://metaflow.org/) (Netflix)

---
# 파이프라인

ml 컴포넌트를 실행, 모니터링
- [airflow](https://airflow.apache.org/)
- [apach beam](https://beam.apache.org/)
- [kubeflow](https://www.kubeflow.org/)
- [mlflow](https://mlflow.org/)

---

1. 컴포넌트 구성
2. 파이프라인 실행

---
# 데이터 주입 컴포넌트
> 데이터 준비하고 파이프라인에 주입
- ml 프로젝트를 시작하는 첫 단계
- 외부의 데이터를 파이프라인에 넣는 역할
    - csv, parquet, avro, tfrecords 등 지원하나 **tfrecords 권장**
    - gcp storage bigquery, aws s3 지원

---
# 데이터 주입 컴포넌트
> 데이터 준비하고 파이프라인에 주입

- 데이터 주입
    - 데이터 읽기
    - 분리 (학습, 검증)
    - 파이프라인에 tfrecords로 저장

**인풋**
csv, parquet, avro, tfrecords, ...

**아웃풋**
ExampleGen
    - 데이터셋 수집, 분할하는 파이프라인 초기 입력 컴포넌트
    
--- 
# 데이터 검증 컴포넌트
> 데이터가 현재 모델에 적합한 데이터인지 검증
- not ML 시스템
    - 데이터 포맷, 타입이 올바른지에 관심
    
--- 
# 데이터 검증 컴포넌트
> 데이터가 현재 모델에 적합한 데이터인지 검증
- ML 시스템
    - 데이터 포맷, 타입이 올바른지 + alpha

--- 
# 데이터 검증 컴포넌트
> 데이터가 현재 모델에 적합한 데이터인지 검증
- ML 시스템
    - 데이터 포맷, 타입이 올바른지 + alpha

**garbage in, garbage out**
![](assets/garbage.jpg)

--- 
# 데이터 검증 컴포넌트
> 데이터가 현재 모델에 적합한 데이터인지 검증
- ML 시스템
    - 데이터 포맷, 타입이 올바른지 + alpha
**garbage in, garbage out**
![](assets/garbage.jpg)
- 새로 들어온 데이터가 모델의 성능을 향상 시킬 수 있는 데이터인지 확인
    - 데이터 분포, 이미지 도메인, 자연어처리

---
# 데이터 검증 컴포넌트
> 데이터가 현재 모델에 적합한 데이터인지 검증

**TFDV**
-> 정형 데이터에 특화된 데이터 검증 라이브러리

- missing, empty, anomaly 데이터 탐지
- 학습, 검증 데이터 비교
- 과거, 현재 데이터 비교
- 효과적인, 불필요한 피쳐 찾기

텍스트, 이미지 데이터의 경우엔??
=> 눈으로 본다... 아직 빈약:cry: (딱히 방법도 없음)

---
# 데이터 전처리 컴포넌트
> 검증된 데이터를 모델이 사용할 수 있는 형태로 처리

**:star:Training-Serving Skew:star:**

---
# 데이터 전처리 컴포넌트
> 검증된 데이터를 모델이 사용할 수 있는 형태로 처리

![bg right:38% 100%](assets/skew.png)
**Training-Serving Skew**
- 학습할 때와 서빙할 때 데이터를 전처리하는 방법이 다름.
    - 학습시에는 pandas 등으로 데이터를 전치리 해놓음
    - 서빙시에는 api를 통해 실시간으로 처리
    
---
# 데이터 전처리 컴포넌트
> 검증된 데이터를 모델이 사용할 수 있는 형태로 처리

**Training-Serving Skew**
- 학습할 때와 서빙할 때 데이터를 전처리하는 방법이 다름.

**해결 방법 1**
- 전처리 서버 추가
    - 서빙서버(trt, tf serving)의 성능을 받쳐줄 수 있어야함 :angry::angry:
        - C++로 수년간 개발되어 온 서빙서버..
    - 비용 증가
        - 전처리 코드가 분리되어 있음
        - 모델마다 다른 전치리 방법 
    - 운영 비용 증가

---
# 데이터 전처리 컴포넌트
> 검증된 데이터를 모델이 사용할 수 있는 형태로 처리

**Training-Serving Skew**
- 학습할 때와 서빙할 때 데이터를 전처리하는 방법이 다름.

**해결 방법 2**
- 백엔드 서버에 구현
    - 백엔드 서버에서 전처리 진행 후 서빙 요청
        - 쉽다.
    - 백엔드 서버 성능에 영향을 끼침.
    - 비용 증가
        - 코드가 분리되어 있음

---
# 데이터 전처리 컴포넌트
> 검증된 데이터를 모델이 사용할 수 있는 형태로 처리

**Training-Serving Skew**
- 학습할 때와 서빙할 때 데이터를 전처리하는 방법이 다름.

**해결 방법 3**
- 전처리를 프레임워크 함수로 구현
    - `torch.nn`, `tft` 등..
        - 모델 저장시 전처리 그래프도 함께 저장 가능
    - 관리 비용 감소
        - 서빙서버만 사용
        - 전처리 함수 통합
    - 구현 난이도 증가
    - torch의 경우 전처리를 위한 함수가 많지 않음.
 
---
# 데이터 전처리 컴포넌트
> 검증된 데이터를 모델이 사용할 수 있는 형태로 처리
![bg right:36% 100%](assets/skew2.png)

**[TFT](https://www.tensorflow.org/tfx/transform/get_started)**
- 동일한 코드로 학습, 서빙에 사용
- tf.image, tf.text, tf.audio, tfa 등등 사용가능
- 전처리 그래프 생성 후 모델 그래프 앞단에 전처리 그래프를 붙이는 방식

```python
model.ftf_layer = tf_transform_output.transform_features_layer()

# 전처리
transformed_features = model.ftf_layer(input)

# 인퍼런스
outputs = model(transformed_features)
```

**but**
- tf 함수를 이용해 전처리 코드 짜야함.

---
# 학습 컴포넌트
> 학습 후 결과물을 저장

- TFX의 Trainer 컴포넌트 사용
    - `run_fn()` 함수를 구현 해놓으면 컴포넌트가 가져다 사용
    - `run_fn()`:
        - 데이터를 읽고
        - 원래처럼 학습을 하고
        - 결과를 저장
        
        하는 함수

---
# 학습 컴포넌트
> 학습 후 결과물을 저장

```python
def run_fn(fn_args):

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)                              

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, batch_size)                 
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, batch_size)

    model = get_model()

    model.fit(
        ...
    )

    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )

```

---
# 모델 검증, 분석 컴포넌트
> 학습된 모델을 분석하고 배포중인 모델과 성능을 비교 후 대체

- Metric을 이용해 모델 분석

---
# 모델 검증, 분석 컴포넌트
> 학습된 모델을 분석하고 배포중인 모델과 성능을 비교 후 대체


**loss vs metric**

- loss: 학습시 모델이 보는 지표
    - 모델이 학습할 방향을 제시

---
# 모델 검증, 분석 컴포넌트
> 학습된 모델을 분석하고 배포중인 모델과 성능을 비교 후 대체

![bg right:50% 80%](assets/lossmetric.png)

**loss vs metric**

- loss: 학습시 모델이 보는 지표
    - 모델이 학습할 방향을 제시

---
# 모델 검증, 분석 컴포넌트
> 학습된 모델을 분석하고 배포중인 모델과 성능을 비교 후 대체

**loss vs metric**

- loss: 학습시 모델이 보는 지표
    - 모델이 학습할 방향을 제시
- metric: 사람이 보는 성능 지표
    - 해석이 쉽고 좀 더 설명이 쉬움

---
# 모델 검증, 분석 컴포넌트
> 학습된 모델을 분석하고 배포중인 모델의 성능을 비교 후 대체
![bg right:50% 80%](assets/roccurve.png)

**loss vs metric**

- loss: 학습시 모델이 보는 성능 지표
    - 모델이 학습할 방향을 제시
- metric: 사람이 보는 성능 지표
    - 해석이 쉽고 좀 더 설명이 가능

---
# 모델 검증, 분석 컴포넌트
> 학습된 모델을 분석하고 배포중인 모델의 성능을 비교 후 대체

- Metric을 이용해 모델 분석

#### Classification
- accuracy, precision, recall ...

#### Regression
- MAE, MSE ...

#### GAN
- :eye::eye:

---
# 모델 검증, 분석 컴포넌트
> 학습된 모델을 분석하고 배포중인 모델의 성능을 비교 후 대체

모델 분석
- [TFMA](https://www.tensorflow.org/tfx/model_analysis/get_started)

XAI
- [WhatIfTool](https://www.tensorflow.org/tensorboard/what_if_tool)

--- 
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**딥러닝 모델 서빙**
- 연산량 많음
- GPU 가속 가능
- 배치 단위 서빙 가능

--- 
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**파이썬 웹 서버(flask, fastAPI, django) 기반**

- 가벼운 것에 쉽게 적용 가능, 개발 쉬움 :thumbsup:
- 성능 많이 떨어짐 :cry:
- 구현해야할 기능이 너무 많음 :cry:
    - gpu 스케쥴링
    - 배치 인퍼런스
    - 모델 버전 관리
    - etc

--- 
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**오픈소스 서빙 서버(triton, tf serving, kf serving)**
- 간편함
- 다양한 프레임워크 지원
- 동시 서빙
- 배치 단위의 서빙
- 모델 관리
- 멀티 gpu
- http/grpc 프로토콜 제공
- metric 수집
- etc

--- 
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

- [TF Serving](https://www.tensorflow.org/tfx/guide/serving)
- [Triton Serving Server](https://github.com/triton-inference-server/server)
- [onnx runtime](https://microsoft.github.io/onnxruntime/)
- [tvm](https://tvm.apache.org/)

--- 
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**[TF Serving](https://github.com/tensorflow/serving)**

**Saved Model**
- 모델의 변수, 상수와 signature를 갖고 있는 디렉토리 
```python
saved_model_path = model.save('saved_models/1', save_format='tf', signatures=signatures)
```

--- 
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

- saved_model.pb: 모델 그래프 구조가 저장된 binary pb 파일
- variables: 모델 그래프의 변수들이 저장된 폴더
- assets: 모델에 필요한 추가적인 파일들 ex) vocab

```bash
$ tree saved_models/                                                                             
saved_models/
└── 1
    ├── assets
    │   └── saved_model.json
    ├── saved_model.pb
    └── variables
        ├── checkpoint
        ├── variables.data-00000-of-00001
        └── variables.index

3 directories, 5 files
```

--- 
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**Model Signatures**

- 모델 그래프의 인풋, 아웃풋, 서빙 방법을 확인
- model signatures 설정으로 모델 아웃풋 변경 가능
- `predict`, `classify`, `regress` 타입 제공
    - 모델이 어떤 문제냐에 따라 방법을 다르게 사용.
    - 잘 모르겠다면 predict만 사용하면 됨

--- 
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**Model Signatures**
- predict
    - savedModel의 디폴트 설정, 가장 유연한 방법
    - 아웃풋을 추가하는 것이 가능 (ex. attention 레이어 아웃풋 추가) 
```json
signature_def: {
  key  : "prediction_signature"                                                                                                                                                                 
  value: {
    inputs: {
      key  : "inputs"
      value: {
        name: "sentence:0"
        dtype: DT_STRING
        tensor_shape: ...
      },
      ...
    }
    outputs: {
      key  : "scores"
      value: {
        name: "y:0"
        dtype: ...
        tensor_shape: ...
      }
    }
    method_name: "tensorflow/serving/predict"
  }
}
```

---
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**Model Signatures**

- classify
    - 하나의 인풋과 2개의 아웃풋(클래스, 스코어)을 제공하는 방법

```json
signature_def: {
  key  : "classification_signature"                                                                                                                                                                  
  value: {
    inputs: {
      key  : "inputs"
      value: {
        name: "sentence:0"
        dtype: DT_STRING
        tensor_shape: ...
      }
    }
    outputs: {
      key  : "classes"
      value: {
        name: "y_classes:0"
        dtype: DT_UINT16
        tensor_shape: ...
      }
    }
    outputs: {
      key  : "scores"
      value: {
        name: "y:0"
        dtype: DT_FLOAT
        tensor_shape: ...
      }
    }
    method_name: "tensorflow/serving/classify"
  }
}
```

---
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**Model Signatures**

- regress
    - 하나의 인풋과 하나의 아웃풋을 제공하는 방법

```json
signature_def: {                                                                                                                                  
  key  : "regression_signature"
  value: {
    inputs: {
      key  : "inputs"
      value: {
        name: "input_tensor_0"
        dtype: ...
        tensor_shape: ...
      }
    }
    outputs: {
      key  : "outputs"
      value: {
        name: "y_outputs_0"
        dtype: DT_FLOAT
        tensor_shape: ...
      }
    }
    method_name: "tensorflow/serving/regress"
  }
}
```

--- 
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**[signature 설정](https://www.tensorflow.org/guide/saved_model#%EB%82%B4%EB%B3%B4%EB%82%BC_%EC%8B%9C%EA%B7%B8%EB%8B%88%EC%B2%98_%EC%A7%80%EC%A0%95%ED%95%98%EA%B8%B0)**
- tf serving 이 어떤 함수를 사용할지 정함
    - `tf.keras.Model`은 자동으로 signatures를 정해줌
- 다수의 signature를 정할 수도 있음
```python
signatures = {
    'serving_default':                                                                                 
        _get_serve_tf_examples_fn(
            model,
            tf_transform_output).get_concrete_function( # tf 함수 불러오기
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')
            )
}
model.save(fn_args.serving_model_dir,
            save_format='tf', signatures=signatures)
```

---

```python
def get_serve_tf_examples_fn(model, tf_transform_output):

    model.tft_layer = tf_transform_output.transform_features_layer() 1

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec) 2

        transformed_features = model.tft_layer(parsed_features) 3
        outputs = model(transformed_features) 4
        return {'outputs': outputs}

    return serve_tf_examples_fn
```

--- 
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**cli 로 model inspect**
`$ pip install tensorflow-serving-api`
```bash
$ saved_model_cli show --dir saved_models/ \
        --tag_set serve --signature_def serving_default

The given SavedModel SignatureDef contains the following input(s):
  inputs['examples'] tensor_info:
      dtype: DT_STRING
      shape: (-1)
      name: serving_default_examples:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['outputs'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: StatefulPartitionedCall_1:0
Method name is: tensorflow/serving/predict
```

--- 
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**서빙서버 실행**
`$ pip install tensorflow-serving-api`:x:

`$ docker pull tensorflow/serving:latest-gpu`:thumbsup:
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2 docker run -p 8500:8500 \ 
             -p 8501:8501 \
             --mount type=bind,source=/tmp/models,target=/models/my_model \ 
             -e MODEL_NAME=my_model \ 
             -e MODEL_BASE_PATH=/models/my_model \
             -t tensorflow/serving
```

---
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

**tf serving**            

- tfx 파이프라인으로 관리X
- 모델 검증 단계에서 검증된 모델을 Pusher가 정해놓은 디렉토리에 업로드 
![](assets/tfx-serving.png)    

---
# 서빙 컴포넌트
> 클라이언트에게 모델 아웃풋을 제공

![](assets/tfserving.png)

---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자

- tfx가 모든 기능을 제공하지는 않음 (아직 버전 0.23..)
- 비정형 데이터를 다루는데에는 아직 부족
- 커스텀 컴포넌트 생성 방법
    1. 파이썬 함수 기반
    2. 컨테이너 기반
    3. 기본 컴포넌트

---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자

- tfx가 모든 기능을 제공하지는 않음 (아직 버전 0.23..)
- 비정형 데이터를 다루는데에는 아직 부족
- [커스텀 컴포넌트 생성 방법](https://www.tensorflow.org/tfx/guide/understanding_custom_components)
    1. 파이썬 함수 기반
    2. 컨테이너 기반
    3. 기본 컴포넌트 구현
- ex
    - 데이터 주입(비정형 데이터, custom db)
    - 분석 결과, 배포 노티
    - [slack api](https://github.com/tensorflow/tfx/tree/master/tfx/examples/custom_components/slack)
    - 등등
---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자

**이미지 로드 -> TFRecord 데이터 주입 컴포넌트**
![](assets/cuscomp.png)
 
---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자

기본 컴포넌트 구현
1. from scratch
2. inherit

---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자

**[from scratch](tensorflow.org/tfx/guide/custom_component?hl=en)**

1. component spec
    - 컴포넌트 실행에 필요한 것들을 정의
        - 인풋, 아웃풋, 파라미터
2. custom executor
    - 컴포넌트 내부의 프로세스를 정의
3. custom driver
    - 컴포넌트의 인풋, 아웃풋을 metadatastore에 등록
    - 데이터 주입 컴포넌트 종류가 아니라면 쓸 일이 거의 없음

---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자

**component spec**
```python
from tfx.types.component_spec import ChannelParameter                                              
from tfx.types.component_spec import ExecutionParameter
from tfx.types import standard_artifacts

class ImageIngestComponentSpec(types.ComponenetSpec):

    PARAMETERS = {
        'name': ExecutionParameter(type=Text)
    }
    INPUTS = {
        # 'input'으로 input path 를 넣는다.
        'input': ChannelParameter(type=standard_artifacts.ExternalArtifact)
    }
    OUTPUTS = {
        # 'examples'에 tfrecords를 저장한다.
        'examples': ChannelParameter(type=standard_artifacts.Examples)
    }
```

---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자


**component excutor**

```python
from tfx.components.base import base_executor

class Executor(base_executor.BaseExecutor):
    
    def Do(self,                                                                                                             
input_dict: Dicr[Text, List[types.Artifact]],
output_dict: Dict[Text, List[types.Artifact]], exec_properties: Dict[Text, Any]) -> None:
        self._log_startup(input_dict, output_dict, exec_properties)

        input_base_url = artifact_utils.get_single_url(input_dict['input'])
        image_files = tf.io.gfile.listdir(input_base_url)
        splits = get_splits(image_files)

        for split_name, images in splits:
            output_dir = artifact_utils.get_split_url(
                output_dict['examples'], split_name
            )

            tfrecord_filename = os.path.join(output_dir, 'images.tfrecord')
            options = tf.io.TFRecordOptions(compression_type=None)
            writer = tf.io.TFRecordWriter(tfrecord_filename, options=options)
            for image in images:
                convert_image_to_TFExample(image, tf_writer, input_base_url)
```

---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자

**component driver**

```python
class ImageIngestDriver(base_driver.BaseDriver):
  def resolve_input_artifacts(
      self,
      input_channels: Dict[Text, types.Channel],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo) -> Dict[Text, List[types.Artifact]]:
    del driver_args 
    del pipeline_info

    input_dict = channel_utils.unwrap_channel_dict(input_channels) 
    for input_list in input_dict.values():
        for single_input in input_list:
            self._metadata_handler.publish_artifacts([single_input]) 
            absl.logging.debug("Registered input: {}".format(single_input))
            absl.logging.debug("single_input.mlmd_artifact "
                               "{}".format(single_input.mlmd_artifact)) 
    return input_dict
```

---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자


**component**

```python
from tfx.components.base import base_component
from tfx import types
from tfx.types import channel_utils

class ImageIngestComponent(base_component.BaseComponent):
    SPEC_CLASS = ImageIngestComponentSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(ImageIngestExecutor)
    DRIVER_CLASS = ImageIngestDriver

    def __init__(self, input, output_data=None, name=None):
        if not output_data:
            examples_artifact = standard_artifacts.Examples()
            examples_artifact.split_names = artifact_utils.encode_split_names(['train', 'eval'])
            
            output_data = channel_utils.as_channel([examples_artifact])

        spec = ImageIngestComponentSpec(input=input, examples=output_data, name=name)
        super(ImageIngestComponent, self).__init__(spec=spec)
```

---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자

**component 사용**

```python
from tfx.utils.dsl_utils import external_input
from image_ingestion_component.component import ImageIngestComponent

context = InteractiveContext()
image_file_path = "/path/to/files"
examples = external_input(dataimage_file_path_root)
example_gen = ImageIngestComponent(input=examples,
                                   name=u'ImageIngestComponent')
context.run(example_gen)

statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
context.run(statistics_gen)

context.show(statistics_gen.outputs['statistics'])
```

---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자

**inherit**

```python
class ImageExampleGenExecutor(BaseExampleGenExecutor):

    @beam.ptransform_fn
    def image_to_example(...):
        ...

    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        return image_to_example
```

---
# 커스텀 컴포넌트
> 필요한 컴포넌트를 직접 만들자

**inherit 사용**

```python
from tfx.components import FileBasedExampleGen
from tfx.utils.dsl_utils import external_input
from image_ingestion_component.executor import ImageExampleGenExecutor                                    

input_config = example_gen_pb2.Input(splits=[
    example_gen_pb2.Input.Split(name='images',
                                pattern='sub-directory/if/needed/*.jpg'),
])
output = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(
            name='train', hash_buckets=4),
        example_gen_pb2.SplitConfig.Split(
            name='eval', hash_buckets=1)
    ])
)
example_gen = FileBasedExampleGen(
    input=external_input("/path/to/images/"),
    input_config=input_config,
    output_config=output,
    custom_executor_spec=executor_spec.ExecutorClassSpec(
        ImageExampleGenExecutor)
)
```

---
# 파이프라인
> [kubeflow](https://www.kubeflow.org/docs/about/kubeflow/)

- 쿠버네티스 기반 ML 툴킷
    - [KFServing](https://github.com/kubeflow/kfserving)
    - [Katib](https://github.com/kubeflow/katib)
    - [TFJob](https://www.kubeflow.org/docs/components/training/tftraining/)
    - [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/)
- v1.1
- 구글 내부 프로젝트로 시작

---
# 파이프라인
> [kubeflow pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/)
![bg right:45% 80%](assets/kf.png)
- 내부는 [argo](https://github.com/argoproj/argo)로 동작
1. 파이썬 스크립트 작성
2. argo conf 생성
3. 파이프라인에 전달

---
# 파이프라인
> [kubeflow pipelines](https://www.kubeflow.org/docs/about/kubeflow/)
![bg right:45% 80%](assets/kf.png)

k8s, argo... 아직 잘 모르겠다면,

tfx 스크립트만..!


---
# 파이프라인
> [kubeflow pipelines](https://www.kubeflow.org/docs/about/kubeflow/)

![bg right:45% 80%](assets/kf.png)

- [tfx_py_script.py](https://gist.github.com/jusonn/24c3628de8a5c63e179ac109c56e6e05)

```bash
INFO:absl:Pipeline root set to: /tfx-data/output
INFO:absl:Adding upstream dependencies for component CsvExampleGen
INFO:absl:Adding upstream dependencies for component ResolverNode_latest_blessed_model_resolver
INFO:absl:Adding upstream dependencies for component StatisticsGen
INFO:absl:   ->  Component: CsvExampleGen
INFO:absl:Adding upstream dependencies for component SchemaGen
INFO:absl:   ->  Component: StatisticsGen
INFO:absl:Adding upstream dependencies for component ExampleValidator
INFO:absl:   ->  Component: SchemaGen
INFO:absl:   ->  Component: StatisticsGen
INFO:absl:Adding upstream dependencies for component Transform
INFO:absl:   ->  Component: CsvExampleGen
INFO:absl:   ->  Component: SchemaGen
INFO:absl:Adding upstream dependencies for component Trainer
INFO:absl:   ->  Component: SchemaGen
INFO:absl:   ->  Component: Transform
INFO:absl:Adding upstream dependencies for component Evaluator
INFO:absl:   ->  Component: CsvExampleGen
INFO:absl:   ->  Component: ResolverNode_latest_blessed_model_resolver
INFO:absl:   ->  Component: Trainer
INFO:absl:Adding upstream dependencies for component Pusher
INFO:absl:   ->  Component: Evaluator
INFO:absl:   ->  Component: Trainer
```

[argo_conf.yaml](https://gist.github.com/jusonn/be0e5c4600af3af28ffa8a4eb764c297)