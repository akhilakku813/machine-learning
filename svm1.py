# use standard dataset load iris
#Training......
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target

#create a model

ml=SVC()
ml=ml.fit(x,y)

#Testing......

result=ml.predict([[2.3,6.2,4.1,4.2]])
print(result)
