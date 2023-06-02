import scipy.optimize
import numpy
import numpy.linalg
import sklearn.datasets

labelToN={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
nToLabel=['Iris-setosa','Iris-versicolor','Iris-virginica']
attributeToN={'Sepal-length':0,'Sepal-width':1,'Petal-length':2,'Petal-width':3}
nToAttribute=['Sepal-length','Sepal-width','Petal-length','Petal-width']

def vcol(v):
    return v.reshape((v.size,1))

def vrow(v):
    return v.reshape((1,v.size))

def load_iris_binary():
    attributes,label = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    attributes = attributes[:, label != 0] #remove 1 class since problem must be binary
    label = label[label != 0] #remove label of 1 class since problem must be binary
    label[label == 2] = 0 #remap label 2 to 1
    return attributes,label

def split_db_2tol(D,L,seed=0):

    nTrain=int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx=numpy.random.permutation(D.shape[1])
    idxTrain=idx[:nTrain]
    idxTest=idx[nTrain:]

    trainData=D[:,idxTrain]
    testData=D[:,idxTest]
    trainLabel=L[idxTrain]
    testLabel=L[idxTest]

    return (trainData,trainLabel),(testData,testLabel)

def logreg_solve(trainData,trainLabel,l):
   
    logreg_obj=logreg_obj_wrap(trainData,trainLabel,l)#function
    wbopt=numpy.zeros((trainData.shape[0]+1,))#params (to modify)
    (wbopt,Jwbmin,_)=scipy.optimize.fmin_l_bfgs_b(logreg_obj,wbopt,approx_grad=True)#optimize paramiters
    print("lambda: %f, Jwbmin: %f" % (l,Jwbmin))#check min result
    return (wbopt[0:-1], wbopt[-1])# return w,b

def logreg_obj_wrap(trainData,trainLabel,l):#useful closure for defining at runtime parameters that we don't vary in order to maximize
    
    n=trainData.shape[1]#variables that doesnt change between calls
    zi=-(2*trainLabel-1)
    def logreg_obj(wb):#wb=4+1 for iris dataset
        w, b = wb[0:-1], wb[-1]#unpacking from 1-d array
        Jwb=0.0
        
        logpost=vrow(w)@trainData+vcol(b)
        for i in range(n):
            Jwb += numpy.logaddexp(0,zi[i]*(logpost[0,i]))
        Jwb/=n
        Jwb+=l/2*(numpy.linalg.norm(w)**2)
        return Jwb
    
    return logreg_obj

def inferClass_logReg(w,b,testData,testLabel):
    #calculate score
    scores=vrow(w)@testData+b
    #infer class
    predictedLabel=numpy.where(scores>0,1,0)
    
    #calculate accuracy
    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])

    return (predictedLabel,acc)

def logreg_mc_obj_wrap(trainData,trainLabel,l):
    
    na=trainData.shape[0]#variables that doesnt change between calls
    nc=len(nToLabel)
    n=trainData.shape[1]

    def logreg_mc_obj(Wb):#Wb=4+4+4+1+1+1 for iris dataset

        W=numpy.zeros((na,nc))#unpacking from 1-d array
        b=numpy.zeros((nc,1))
        for c in range(nc):
            W[:,c] = Wb[na*c:na*(c+1)]
            b[c,0] = Wb[nc*na+c]
        
        S=numpy.zeros((nc,n))
        for i in range(n):
            S[:,i]=(W.T@vcol(trainData[:,i])+b).reshape((nc,))
        Ylog=S-(S.max(axis=0)+numpy.log((numpy.exp(S-S.max(axis=0))).sum(axis=0)))
        T=numpy.eye(nc)[trainLabel].T
        Jwb=l/2*(W*W).sum()
        Jwb-=(T*Ylog).sum()/n
        
        return Jwb
    
    return logreg_mc_obj


def logreg_mc_solve(trainData,trainLabel,l):

    na=trainData.shape[0]
    nc=len(nToLabel)
   
    logreg_obj=logreg_mc_obj_wrap(trainData,trainLabel,l)#function
    Wbopt=numpy.random.random((trainData.shape[0]*nc+1*nc,))#params (to modify)
    (Wbopt,Jwbmin,_)=scipy.optimize.fmin_l_bfgs_b(logreg_obj,Wbopt,approx_grad=True)#optimize paramiters
    print("lambda: %f, Jwbmin: %f" % (l,Jwbmin))#check min result
    
    W=numpy.zeros((na,nc))#unpacking from 1-d array
    b=numpy.zeros((nc,1))
    for i in range(nc):
        W[:,i] = Wbopt[na*i:na*(i+1)]
        b[i,0] = Wbopt[nc*na+i]
    
    return (W, b)




def inferClass_logReg_multiclass(W,b,testData,testLabel):
    n=testData.shape[1]
    nc=len(nToLabel)
    scores=W.T@testData+vcol(b)
    
    predictedLabel=numpy.zeros((n,))

    for i in range(n):
        max=0
        maxV=0.0
        for c in range(nc):
            if (scores[c,i]>maxV):
                maxV=scores[c,i]
                max=c
        
        predictedLabel[i]=max

    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])

    return (predictedLabel,acc)

if __name__ == "__main__":
    
    #binary logistic regression
    attributes , label = load_iris_binary()
    (trainData,trainLabel),(testData,testLabel) = split_db_2tol(attributes,label)

    (w,b)=logreg_solve(trainData,trainLabel,10**-6)
    (_,acc)=inferClass_logReg(w,b,testData,testLabel)
    print("accuracy:%f"%acc)
    (w,b)=logreg_solve(trainData,trainLabel,10**-3)
    (_,acc)=inferClass_logReg(w,b,testData,testLabel)
    print("accuracy:%f"%acc)
    (w,b)=logreg_solve(trainData,trainLabel,10**-1)
    (_,acc)=inferClass_logReg(w,b,testData,testLabel)
    print("accuracy:%f"%acc)
    (w,b)=logreg_solve(trainData,trainLabel,1.0)
    (_,acc)=inferClass_logReg(w,b,testData,testLabel)
    print("accuracy:%f"%acc)

    #multiclass logistic regression
    attributes,label = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    (trainData,trainLabel),(testData,testLabel) = split_db_2tol(attributes,label)
    (W,b)=logreg_mc_solve(trainData,trainLabel,10**-6)
    (_,acc)=inferClass_logReg_multiclass(W,b,testData,testLabel)
    print("accuracy:%f"%acc)
    (W,b)=logreg_mc_solve(trainData,trainLabel,10**-3)
    (_,acc)=inferClass_logReg_multiclass(W,b,testData,testLabel)
    print("accuracy:%f"%acc)
    (W,b)=logreg_mc_solve(trainData,trainLabel,10**-1)
    (_,acc)=inferClass_logReg_multiclass(W,b,testData,testLabel)
    print("accuracy:%f"%acc)
    (W,b)=logreg_mc_solve(trainData,trainLabel,1.0)
    (_,acc)=inferClass_logReg_multiclass(W,b,testData,testLabel)
    print("accuracy:%f"%acc)
    