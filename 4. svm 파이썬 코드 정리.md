# 서포트 백터 머신 파이썬 연습

    from sklearn.datasets import make_blobs 
  
    X,y = make_blobs(centers=4, random_state=8) 
  
    y = y%2 
  
    mglearn.discrete_scatter(X[:,0], X[:,1], y) 
    
    plt.xlabel("Feature 0") 
  
    plt.ylabel("Feature 1")

- 가짜 데이터를 만들 때 make_blobs 가져옴
- make_blob -> 입력값은 X, 정답 라벨은 y에 저장해
- centers = 4 ->클러스터 4개 만들어
- y = y%2 -> y의 클래스를 0,1로 만들어주는 과정
- mglearn.discrete_scatter ->  산점도 그리는 코드
- X[:,0], X[:,1] -> X의 첫번째 특징이랑 두번째 특징 뽑아내기


## 결정경계 만들기

    mglearn.plots.plot_2d_separator(linear_svm, X)

## 3D 환경 만들기


    # add the squared first feature
    X_new = np.hstack([X, X[:, 1:] ** 2])


    from mpl_toolkits.mplot3d import Axes3D, axes3d
    
    figure = plt.figure()
    
    # visualize in 3D
    ax = Axes3D(figure, elev=-152, azim=-26, auto_add_to_figure=False)
    figure.add_axes(ax)
    # plot first all the points with y==0, then all with y == 1
    
    mask = y == 0
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
           
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.cm2, s=60, edgecolor='k')
           
    ax.set_xlabel("feature0")
    ax.set_ylabel("feature1")
    ax.set_zlabel("feature1 ** 2")

- 기존 X에 새로운 특징 하나 추가
- > hstack을 이용해 옆으로 붙임
- 그래프 그리기 위한 라이브러리 가져오기
- figure = plt.figure() -> 그래프 생성
- ax = Axes3D(figure, elev=-152, azim=-26, auto_add_to_figure=False) -> 3d 좌표축 생성 , elev, azim -> 보는각도 설정
- figure.add_axes(ax) -> 만든 3d 좌표축을 figure에 추가
- mask = y == 0 -> 클래스가 0인애들 고르기
- ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b') -> 파란색 점으로 3d에 표현


## 3D 환경에서 svm 평면 구하기

    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
  
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50) 
  
    XX, YY = np.meshgrid(xx, yy) 
  
    ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2] 
  
    ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3) 
  
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60, edgecolor='k') 
  
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60, edgecolor='k')

- xx -> feature0의 범위를 기준으로 최소~최대 사이를 50개의 점으로 나눔 -> x축용 좌표
- yy -> feature1의 범위를 기준으로 최소~최대 사이를 50개의 점으로 나눔 -> y축용 좌표
- > 이렇게 하는 이유는 x,y범위를 촘촘하게 쪼개서 평면으로 만들기 위함
- XX, YY = np.meshgrid(xx, yy) -> 2D 격자 생성
-  ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
-  > 원래 식 -> coef[0] * XX = coed[1] * YY + coef[2] * ZZ + intercept = 0 -> 평면 방정식
   > 
   > coef[0] -> w1  coef[1] -> w2 coef[2] -> w3

- 3차원에서 SVM이 만든 결정경계(평면)를 직접 계산해서 시각화한 코드

## 3D -> 2D

    ZZ = YY ** 2
    
    dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    
    plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
                cmap=mglearn.cm2, alpha=0.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    
    plt.xlabel("Feature 0")
    
    plt.ylabel("Feature 1")

- ZZ를 특징 다시 바꿔주기
- dec -> (x,y,z) 좌표들을 모델에 넣어서 결정함수 값 계산
- ravel() -> 2D gird -> 1줄로 펼짐
- np.c[] -> (x,y,z) 붙여서 입력 형태 맞춤
- plt.contourf -> 결과를 다시 grid형태로 바꿔서
- level -> 0을 기준으로 클래스 나누기


## SVC 

    from sklearn.svm import SVC

    X, y = mglearn.tools.make_handcrafted_dataset()
    
    svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
    
    mglearn.plots.plot_2d_separator(svm, X, eps=.5)
    
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    
    # plot support vectors
    
    sv = svm.support_vectors_
    
    # class labels of support vectors are given by the sign of the dual coefficients

    
    sv_labels = svm.dual_coef_.ravel() > 0  
    
    mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
  

- svc 가져오기
- svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
- > RBF 비선형 경계로 만들 수 잇는 커널 사용
  > 
  > C = 10 -> 오분류를 얼마나 엄격하게 줄일지
  >
  > gamma = 0.1 -> 각 데이터 점 영향 범위를 정하는 값

- sv = svm.support_vectors_ -> SVM의 서포트 벡터만 따로 꺼내기
- sv_labels = svm.dual_coef_.ravel() > 0 -> 클래스 나누기


## 실제 데이터로 실습 + 스케일링  
    
    cancer = load_breast_cancer()
    
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    svc = SVC(C=100)
    
    svc.fit(X_train, y_train)
    
    y_train_hat = svc.predict(X_train)
    y_test_hat = svc.predict(X_test)

    print("Accuracy on training set: {:.2f}".format(accuracy_score(y_train, y_train_hat)))
    print("Accuracy on test set: {:.2f}".format(accuracy_score(y_test, y_test_hat)))
    print(confusion_matrix(y_test, y_test_hat))

1)   수동으로 스케일링
   
    min_on_training = X_train.min(axis=0)
    
    range_on_training = (X_train - min_on_training).max(axis=0)


    X_train_scaled = (X_train - min_on_training) / range_on_training
    print("Minimum for each feature\n", X_train_scaled.min(axis=0))
    print("Maximum for each feature\n", X_train_scaled.max(axis=0))

2)   패키기 가져와서 스케일링
   
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
