# Extra analyses for the paper: per-class recall, 20-split sensitivity, depth and k sensitivity, full tree rules.
# Run from the project folder: python analysis_extra.py
import os, pandas as pd, numpy as np, json, sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score
cols=['buying','maint','doors','persons','lug_boot','safety','class']
df=pd.read_csv('data/car.data',header=None,names=cols); f=cols[:6]; co=['unacc','acc','good','vgood']
cats=[['low','med','high','vhigh']]*2+[['2','3','4','5more'],['2','4','more'],['small','med','big'],['low','med','high']]
X=pd.DataFrame(OrdinalEncoder(categories=cats).fit_transform(df[f]),columns=f).astype(int)
y=df['class'].map({c:i for i,c in enumerate(co)})
def mk(): return {'Majority baseline':DummyClassifier(strategy='most_frequent'),
 'Logistic Regression':LogisticRegression(max_iter=1000),'Decision Tree':DecisionTreeClassifier(max_depth=6,random_state=42),
 'KNN (k=5)':KNeighborsClassifier(5),'Naive Bayes':CategoricalNB(min_categories=[4,4,4,3,3,3])}
out={'versions':{'python':'3.12','scikit-learn':sklearn.__version__,'pandas':pd.__version__}}
# per-class recall on the main split
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pc={}; cms={}
for n,m in mk().items():
    m.fit(Xtr,ytr); p=m.predict(Xte)
    pc[n]=[float(v) for v in recall_score(yte,p,average=None,labels=[0,1,2,3],zero_division=0)]
    cms[n]=confusion_matrix(yte,p,labels=[0,1,2,3]).tolist()
out['per_class_recall']=pc; out['cms']=cms; out['test_counts']=np.bincount(yte,minlength=4).tolist()
# seed sensitivity: 20 different stratified splits
seeds=range(20); sens={n:[] for n in mk()}
for sd in seeds:
    a,b,c,d=train_test_split(X,y,test_size=0.2,random_state=sd,stratify=y)
    for n,m in mk().items():
        m.fit(a,c); sens[n].append(f1_score(d,m.predict(b),average='macro',zero_division=0))
out['seed_sens']=sens
# depth sensitivity on the test split (fixed depth, for transparency)
dep={}
for d_ in range(2,11):
    t=DecisionTreeClassifier(max_depth=d_,random_state=42).fit(Xtr,ytr)
    dep[d_]=float(f1_score(yte,t.predict(Xte),average='macro'))
out['depth_test']=dep
# k sensitivity for KNN (repeated CV on train)
kk={}
cv=StratifiedKFold(5,shuffle=True,random_state=42)
for k in [1,3,5,7,9,15]:
    kk[k]=float(cross_val_score(KNeighborsClassifier(k),Xtr,ytr,cv=cv,scoring='f1_macro').mean())
out['knn_k_cv']=kk
# full depth-6 tree rules
t6=DecisionTreeClassifier(max_depth=6,random_state=42).fit(Xtr,ytr)
out['rules']=export_text(t6,feature_names=f,class_names=co)
out['n_leaves']=int(t6.get_n_leaves())
os.makedirs('results', exist_ok=True)
json.dump(out,open('results/extra_analysis.json','w'),default=float)
print('per-class recall:',{k:[round(x,3) for x in v] for k,v in pc.items()})
print('test counts',out['test_counts'])
for n,v in sens.items(): print('seed',n,round(np.mean(v),3),round(np.std(v,ddof=1),3),round(min(v),3),round(max(v),3))
print('depth test',{k:round(v,3) for k,v in dep.items()})
print('knn k cv',{k:round(v,3) for k,v in kk.items()})
print('leaves',out['n_leaves'], sklearn.__version__, pd.__version__)
