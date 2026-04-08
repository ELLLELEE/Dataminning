# 02_data 내용 정리
### - 데이터린?
- > objectdhk attribute의 구성체
  >
- Attribute
- > variable
  >
  > feature
- Object
- > smaple
  >
  > instance
  >
- Attribute value
- > 같은 att라도 다른 값으로 저장될 수 있음 -> 키를 미터나 센치로 표현가능
  >
  > 다른 arr라도 같은 세트로 묶일 수 있음
  >
- Att type
  1) 카테고리(dicrete) -> 구분가능
     - Nominal -> ID number
     - Ordinal -> 순서를 가지는 특징 -> 학점, 키
  2) Continuous -> 실제 숫자
     - Interval -> 벨류간에 차이가 의미 있음 -> 온도 * 기준값 0 존재 X
     - Ratio -> 비율이 의미를 가짐 -> 절대 온도 * 기준값 0이 존재
    
  - comment
  - > 실제 데이터들은 노이즈가 많음
    >
    > 그래서 우리는 의미있는 데이터를 특성에 맞게 결정해야함

  ### - 데이터 타입
  - record data
    - 메트릭스 데이터 -> 행렬형태로 실제 연속적인 숫ㅈ ㅏ들어감
    - document -> 등장횟수를 보여줌
    - transaction -> 아이템들의 세트 -> 장바구니 이론
  - graph data -> 웹 페이지간의 관계
  - ordered data -> 순서가 중요한 데이터
 
  ### - 데이터 질
  - 문제 있는 데이터
    - 타입이 맞지않는 데이터
    - 중복 데이터
    - 놓친 데이터 , 잃어버린 데이터, 기록안된 데이터
      - Missing completly at random(MCAR) -> 빵꾸가 랜덤으로 생김
      - Missing at random(MAR) -> 빵꾸가 특정 att과 관련 있음
      - Missing not at random(MNAR) -> 빵꾸가 이유가 있게 랜덤 -> 이유 찾아야됨
      - > 기록 안된 데이터를 삭제
        >
        > 임의의 값으로 채우기
        >
- noisy
- outlier(이상치)
- > case1) 노이즈
  >
  > caes2) 우리의 목표
