from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pickle 
data=load_iris()
x,y=data.data,data.target
model=LogisticRegression(max_iter=200)
model.fit(x,y)
with open("model.pkl","wb") as f:
    pickle.dump(model,f)
print("model saved")    