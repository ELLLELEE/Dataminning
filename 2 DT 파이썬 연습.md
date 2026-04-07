# 2 DT 파이썬 연습
- print("Keys of iris_dataset:\n", iris_dataset.keys())
- print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
- > #### {:.2f} -> 소수점 둘째자리까지 표시
  > 
  > #### .format() -> {} 자리에 값을 넣어주는 함수

## DT 실습
- 트리 모델을 먼저 임포트
  
  from sklearn.datasets import load_breast_cancer

  from sklearn.tree import DecisionTreeClassifier

  from sklearn import tree

- 데이터 스플릿
- >  cancer = load_breast_cancer()
  > 
  > X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, stratify=cancer.target, random_state=42

- 데이터 핏팅
- > clf = DecisionTreeClassifier(random_state=0)
  > 
  > clf.fit(X_train, y_train)
#### * 여기서 fit은 꼭 train만하기!!!!!!!!!!!!!!!

- 데이터 예측
- > y_train_hat = clf.predict(X_train)
  > 
  > y_test_hat = clf.predict(X_test)

- 데이터 평가
- >from sklearn.metrics import accuracy_score

- 트리 시각화 1
- > clf.fit(X_train, y_train)
  >
  > dt_clf_model_text = tree.export_text(clf)
  >
  > print(dt_clf_model_text)

- 트리 시각화 2
- > import matplotlib.pyplot as plt
  >
  > fig = plt.figure(figsize=(15, 8))
  > 
  > #### plt.figure -> 그래프 그릴 도화지 생성
  > 
  > #### figsize -> 도화지 크기 설정
  >
  > tree.plot_tree(clf,feature_names=cancer.feature_names,
                  class_names=["malignant", "benign"],
                  filled=True)
  >
  > #### tree.plot_tree(clf, ) -> 결청 트리 모델을 그림으로 시각화
  >
  > #### feature_names=cancer.feature_names -> 각 노드에 사용된 특징 이름 표시
  >
  > #### class_names=["malignant", "benign"] -> 결과 클래스 이름 표시
  >
  > #### filled=True -> 노드에 색 추
  
