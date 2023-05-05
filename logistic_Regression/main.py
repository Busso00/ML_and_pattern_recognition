import scipy.optimize
import numpy
import numpy.linalg
import sklearn.datasets

def vcol(v):
    return v.reshape((v.size,1))

def vrow(v):
    return v.reshape((1,v.size))

def numerical_solve1():
    def f(values):
        y = values[0]
        z = values[1]
        return (y+3)**2 + numpy.sin(y) + (z+1)**2
    
    values=numpy.zeros((2,))

    print(scipy.optimize.fmin_l_bfgs_b(f,values,approx_grad=True))

def numerical_solve2():
    def f(values):
        y = values[0]
        z = values[1]
        return (y+3)**2 + numpy.sin(y) + (z+1)**2,numpy.array([ 2*(y+3) + numpy.cos(y) , 2*(z+1)] )
    
    values=numpy.zeros((2,))

    print(scipy.optimize.fmin_l_bfgs_b(f,values))

def load_iris_binary():
    D,L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] #remove 1 class since problem must be binary
    L = L[L != 0] #remove label of 1 class since problem must be binary
    L[L == 2] = 0 #remap label 2 to 1
    return D,L

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

    logreg_obj=logreg_obj_wrap(trainData,trainLabel,l)
    wbopt=numpy.zeros((trainData.shape[0]+1,))
    (wbopt,Jwbmin,_)=scipy.optimize.fmin_l_bfgs_b(logreg_obj,wbopt,approx_grad=True,factr=1000.0)
    print("lambda: %f, Jwbmin: %f" % (l,Jwbmin))
    return (wbopt[0:-1], wbopt[-1])



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
    #tested
    #print(numerical_solve1())
    #print(numerical_solve2())
    
    #logistic regression biary 
    D,L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2tol(D,L)

    (w,b)=logreg_solve(DTR,LTR,10**-6)
    (_,acc)=inferClass_logReg(w,b,DTE,LTE)
    print("accuracy:%f"%acc)
    (w,b)=logreg_solve(DTR,LTR,10**-3)
    (_,acc)=inferClass_logReg(w,b,DTE,LTE)
    print("accuracy:%f"%acc)
    (w,b)=logreg_solve(DTR,LTR,10**-1)
    (_,acc)=inferClass_logReg(w,b,DTE,LTE)
    print("accuracy:%f"%acc)
    (w,b)=logreg_solve(DTR,LTR,1.0)
    (_,acc)=inferClass_logReg(w,b,DTE,LTE)
    print("accuracy:%f"%acc)



    