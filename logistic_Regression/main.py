import scipy.optimize
import numpy
import numpy.linalg
import sklearn.datasets

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
    
    def logreg_obj(wb):#wb=4+1 for iris dataset
        w, b = wb[0:-1], wb[-1]
        Jwb=l/2*(numpy.linalg.norm(w)**2)
        n=trainData.shape[1]
        for i in range(n):
            Jwb += numpy.logaddexp(0,-(2*trainLabel[i]-1)*(vrow(w)@vcol(trainData[:,i])+b))/n
        return Jwb[0][0]
    
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



    