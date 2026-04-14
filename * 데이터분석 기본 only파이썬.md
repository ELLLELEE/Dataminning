- 기본 파이썬, 라이브러리 활용 경험이 없으신 분은 (최소한) 아래 사이트의 코드를 반드시 연습하세요. 기초입니다. 
    - https://cs231n.github.io/python-numpy-tutorial/
- 본 수업에서 파이썬 기본을 모두 다룰 수 없습니다. 부족하신 분들은 개인적으로 익숙해지는 과정을 거치세요. 
    - numpy
    - pandas
    - matplotlib



# Numpy
- 행렬 연산을 쉽게 해주는 패키지 (matlab처럼)

import numpy as np

data = np.random.randn(2,3)
data

data * 10

data + data

data.shape

a = np.array([1, 2, 3])   # Create a rank 1 array
print(a[0], a[1], a[2])   # index starts from '0'

a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"

a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(5)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

## array indexing

## slicing

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
a

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
b

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"

b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"

a



# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"

## integer array indexing

import numpy as np

# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])
b

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])

## boolean indexing

a = np.array([[1,2], [3, 4], [5, 6]])
a

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 4 5 6]"



# Pandas

- from : https://www.kaggle.com/learn/pandas
- 엑셀처럼 데이터 테이블을 다루는데 용이한 패키지

import pandas as pd

pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
# Series가 모여 DataFrame을 이룰 때 Series의 name이 DataFrame의 column name역할

# dictionary로부터 생성
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])

## Read files

reviews = pd.read_csv("winemag-data-130k-v2.csv")

reviews.head()

reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)

reviews.head()

## indexing

reviews.country

reviews['country']

reviews['country'][0]

index-based selection: selecting data based on its numerical position in the data

reviews.iloc[0]

reviews.iloc[:, 0]

reviews.iloc[:3, 0]

reviews.iloc[-5:]

label-based selection use 'index'

reviews.loc[0, 'country']

reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]

## conditional selection

reviews.country == 'Italy'

reviews.loc[reviews.country == 'Italy']

reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]

## assigning data

reviews['index_backwards'] = range(len(reviews), 0, -1)
reviews.head()

## summary function

reviews.taster_name.describe()

reviews.points.describe()

reviews.taster_name.value_counts()



## missing data

pd.isnull(reviews.country)

reviews[pd.isnull(reviews.country)]

reviews.region_2.fillna("Unknown")

reviews

reviews.region_2.fillna("Unknown", inplace=True)

reviews

## combining (join)

data_A = {'key': [1,2,3], 'name': ['Jane', 'John', 'Peter']}
dataframe_A = pd.DataFrame(data_A, columns = ['key', 'name'])

data_B = {'key': [2,3,4], 'age': [18, 15, 20]}
dataframe_B = pd.DataFrame(data_B, columns = ['key', 'age'])

print(dataframe_A)
print(dataframe_B)

df_INNER_JOIN = pd.merge(dataframe_A, dataframe_B, left_on='key', right_on='key', how='inner')
print(df_INNER_JOIN)

df_LEFT_JOIN = pd.merge(dataframe_A, dataframe_B, left_on='key', right_on='key', how='left')
print(df_LEFT_JOIN)

df_RIGHT_JOIN = pd.merge(dataframe_A, dataframe_B, left_on='key', right_on='key', how='right')
print(df_RIGHT_JOIN)

df_OUTER_JOIN = pd.merge(dataframe_A, dataframe_B, left_on='key', right_on='key', how='outer')
print(df_OUTER_JOIN)



