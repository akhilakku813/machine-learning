# use our sample data
#Training......
import numpy as np
from sklearn.svm import SVC
x=np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
y=np.array([1,1,2,2])

#create a model

ml=SVC()
ml=ml.fit(x,y)

#Testing......

result=ml.predict([[-0.8,-1]])
print(result)
