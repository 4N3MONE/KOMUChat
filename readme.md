# KOMUChat : 인공지능 학습을 위한 온라인 커뮤니티 대화 데이터셋

## 개요
에펨코리아 + 인스티즈에서 수집한 약 3만건의 문장 페어 데이터

## 속성 정보
Q : 커뮤니티 게시글의 제목을 활용한 질문 문장

A : 게시글에 달린 첫 번째 댓글을 활용한 답변 문장

tag : 커뮤니티 게시판 내 세부 태그/탭

src : 데이터 출처 사이트

## 데이터 예시
| tag | Q | A | src |
|-----------|-----------|-----------|-----------|
| breakup     | 헤어졌읍니다      |   힘내 더 좋은 인연 나타날거야    | fmkorea      |
| talk      | 드라마만 주구장창 보는중      | 뭐 보고 있으신가요      | fmkorea      |
| wedding      | 결혼까지 생각하는 상태일수록 더 깐깐하게 따지게 돼?     | 아무래도 그런 거 같아     | instiz     |
| flirt      | ISTJ 여자가 남자한테 관심있으면 어떻게 해?     | 연락 자주 하고 취향 많이 물어보고 자주는 아니어도 꾸준히 만나려 함     | instiz     |
| love       | #청자#들아 연애는 갑자기 시작한다고 생각해?    | 나는 진자 신기한 게 연애에 대한 기대가 떨어지면 하고 있더랔ㅋㅋㅋ      | instiz     |


## getting start with KoBART

*** requirements.txt를 참고 부탁드립니다***  
KoBART 폴더로 이동하기  
```cd KOMUChat/kobart```  

### let's train!
```sh bart_comuchat_job.sh ```  
.py 파일로 직접 돌리는 것도 가능합니다.  
```python kobart_chit_chat.py ```  

한 커뮤니티만의 데이터로 학습을 진행하고 싶다면(예: 인스티즈)..  
```sh bart_comu_ins_job.sh```  

### let's chat!
```python chat.py```

## getting start with KoGPT
*** requirements.txt를 참고 부탁드립니다. ***  
KoGPT폴더로 이동해주세요.  
```cd KOMUChat/GPT_ForTraining```  

### let's train!
```sh comuchat_gpt_job.sh ```  
.py을 사용하여 직접 학습하는 것도 가능합니다.
```python train_torch.py ```  

한 커뮤니티만의 데이터로 학습을 진행하고 싶다면(예: 인스티즈)..   
```sh comu_ins_gpt_job.sh```  

### let's chat!
.py파일의 --chat 인자를 변경해주면 됩니다.
```python train_torch.py --chat True ```

## getting start with T5
KoGPT폴더로 이동해주세요  
```cd KOMUChat/pko_t5```  
가상환경을 구축하여 학습을 진행할 수도 있습니다.  
```conda env create -n t5 python=3.8 -f requirements.txt```  

### let's train!
하이퍼파라미터 조정하기  
```python train.py --num_train_epochs 5 --train_batch_size 16 ...```   

한 커뮤니티만의 데이터로 학습을 진행하고 싶다면..   
```python train.py --train_path ../data/comuchat_ins_train.csv --test_path  ../data/comuchat_ins_valid.csv```

### let's chat!
```python chat.py```  
채팅을 멈추고 싶다면 'false'를 입력하면 됩니다.  

## 인용
```
@article{ART002972994,
author={유용상 and 정민화 and 이승민 and 송민},
title={KOMUChat : 인공지능 학습을 위한 온라인 커뮤니티 대화 데이터셋 연구},
journal={지능정보연구},
issn={2288-4866},
year={2023},
volume={29},
number={2},
pages={219-240}
}
```
