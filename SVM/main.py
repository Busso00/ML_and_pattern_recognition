import scipy.optimize
import numpy
import numpy.linalg
import sklearn.datasets

labelToN={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
nToLabel=['Iris-setosa','Iris-versicolor','Iris-virginica']
attributeToN={'Sepal-length':0,'Sepal-width':1,'Petal-length':2,'Petal-width':3}
nToAttribute=['Sepal-length','Sepal-width','Petal-length','Petal-width']

PC = [1/2,1/2]
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

    trainDataV=D[:,idxTrain]
    testDataV=D[:,idxTest]
    trainLabelV=L[idxTrain]
    testLabelV=L[idxTest]

    return (trainDataV,trainLabelV),(testDataV,testLabelV)

def SVMPrimalSolve(trainDataV,trainLabelV,K,C):
    #C, K are hyperparameters of the model
    N_ATTRS=trainDataV.shape[0]

    SVMobj=SVMPrimalObjWrap(trainDataV,trainLabelV,K,C)#function
    wbopt=numpy.zeros((N_ATTRS+1,))#params (to modify)
    (wbopt,Jwbmin,_)=scipy.optimize.fmin_l_bfgs_b(SVMobj,wbopt,approx_grad=True,factr=1.0)#optimize paramiters
    print("C: %f, K: %f, primal loss: %f" % (C,K,Jwbmin))#check min result
    return (wbopt[0:-1], wbopt[-1])# return w,b


def SVMPrimalObjWrap(trainDataV,trainLabelV,K,C):#useful closure for defining at runtime parameters that we don't vary in order to maximize
    #C, K are hyperparameters of the model
    N_ATTRS=trainDataV.shape[0]
    N_RECORDS=trainDataV.shape[1]

    expandedDataV=numpy.zeros((N_ATTRS+1,N_RECORDS))
    expandedDataV[0:-1,:]=trainDataV
    expandedDataV[-1,:]=numpy.ones((N_RECORDS,))*K
    zV=2*trainLabelV-1
    #no use of H
    
    def SVMPrimalObj(wb):
        regterm=(numpy.linalg.norm(wb)**2)/2#is actually objective function if class are perfectly separable (I assume they are not)
        hingelossV=numpy.maximum(numpy.zeros(expandedDataV.shape[1]),1-zV*(vrow(wb)@expandedDataV))
        return regterm+C*numpy.sum(hingelossV)
    
    return SVMPrimalObj


def SVMDualSolve(trainDataV,trainLabelV,K,C):
    #C, K are hyperparameters of the model
    N_RECORDS=trainDataV.shape[1]

    SVMobj=SVMDualWrap(trainDataV,trainLabelV,K)
    alfaVopt=numpy.zeros((N_RECORDS,))
    alfaLimits=[]
    for _ in range(N_RECORDS):
        alfaLimits.append((0,C))
    (alfaVopt,Jalfamin,_)=scipy.optimize.fmin_l_bfgs_b(SVMobj[0],alfaVopt,fprime=SVMobj[1],bounds=alfaLimits,factr=1.0)#optimize paramiters
    
    return (alfaVopt,Jalfamin)# return w,b

def SVMDualWrap(trainDataV,trainLabelV,K):#useful closure for defining at runtime parameters that we don't vary in order to maximize
    N_ATTRS=trainDataV.shape[0]
    N_RECORDS=trainDataV.shape[1]

    expandedDataV=numpy.zeros((N_ATTRS+1,N_RECORDS))
    expandedDataV[0:-1,:]=trainDataV
    expandedDataV[-1,:]=numpy.ones((N_RECORDS,))*K
    G=expandedDataV.T@expandedDataV
    zV=2*trainLabelV-1
    H=G*vrow(zV)*vcol(zV)

    def SVMDualObj(alfaV):
        
        return vrow(alfaV)@H@vcol(alfaV)/2 - alfaV.sum()
    
    def SVMDualGradient(alfaV):

        return (H@vcol(alfaV) - 1 ).ravel()
    
    return (SVMDualObj,SVMDualGradient)

def recoverPrimal(trainDataV,trainLabelV,alfaV,K,C):
    N_ATTRS=trainDataV.shape[0]
    N_RECORDS=trainDataV.shape[1]

    expandedDataV=numpy.zeros((N_ATTRS+1,N_RECORDS))
    expandedDataV[0:-1,:]=trainDataV
    expandedDataV[-1,:]=numpy.ones((N_RECORDS,))*K
    zV=2*trainLabelV-1
    wb = numpy.sum(alfaV*zV*expandedDataV,axis=1)
    regterm=(numpy.linalg.norm(wb)**2)/2
    hingelossV=numpy.maximum(numpy.zeros(expandedDataV.shape[1]),1-zV*(vrow(wb)@expandedDataV))
    Jw = regterm+C*numpy.sum(hingelossV)
    return (wb,Jw)

def inferClassLinearSVM(wb,testDataV,testLabelV,K):
    N_RECORDS=testDataV.shape[1]

    w,b = wb[0:-1],wb[-1]
    scoreV=vrow(w)@testDataV+b*K
    predictedLabelV=numpy.where(scoreV>0,1,0)
    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (scoreV,predictedLabelV,acc)

def polinKernelSVMSolve(trainDataV,trainLabelV,C,eps=0.0,degree=2,const=0.0):
    #C, eps, const are hyperparameters of the model
    N_RECORDS=trainDataV.shape[1]

    SVMobj=polinKernelSVMWrap(trainDataV,trainLabelV,eps=eps,degree=degree,const=const)#function
    alfaVopt=numpy.zeros((N_RECORDS,))#params (to modify)
    alfaLimits=[]
    for i in range(N_RECORDS):
        alfaLimits.append((0,C))
    (alfaVopt,Jalfamin,_)=scipy.optimize.fmin_l_bfgs_b(SVMobj[0],alfaVopt,fprime=SVMobj[1],bounds=alfaLimits,factr=1.0)#optimize paramiters
    
    return (alfaVopt,Jalfamin)# return alfaV,loss (dual) , also JAlfamin

def polinKernelSVMWrap(trainDataV,trainLabelV,eps=0,degree=2,const=0):#useful closure for defining at runtime parameters that we don't vary in order to maximize
    #eps, const, degree are hyperparameters of the model

    kernelM=(trainDataV.T@trainDataV + const)**degree + eps
    zV=2*trainLabelV-1
    H=kernelM*vrow(zV)*vcol(zV)

    def SVMObj(alfaV):
        
        return vrow(alfaV)@H@vcol(alfaV)/2 - alfaV.sum()
    
    def SVMGradient(alfaV):

        return (H@vcol(alfaV) - 1 ).ravel()
    
    return (SVMObj,SVMGradient)

def inferClassPolinSVM(trainDataV,trainLabelV,testDataV,testLabelV,alfaV,eps=0.0,degree=2,const=0.0):
    N_RECORDS=testDataV.shape[1]

    zV=2*trainLabelV-1
    
    scoreV = (vcol(zV)*vcol(alfaV)*((trainDataV.T@testDataV + const)**degree + eps)).sum(axis=0)
    
    predictedLabelV=numpy.where(scoreV>0,1,0)
    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (scoreV,predictedLabelV,acc)


def expKernelSVMWrap(trainDataV,trainLabelV,eps=0.0,gamma=1.0):#useful closure for defining at runtime parameters that we don't vary in order to maximize
    N_RECORDS=trainDataV.shape[1]
    
    kernelM=numpy.zeros((N_RECORDS,N_RECORDS))
    for row in range(N_RECORDS):
        #modify kernel
        kernelM[row,:]=numpy.exp(-gamma*numpy.linalg.norm(trainDataV-vcol(trainDataV[:,row]),axis=0)**2)+eps

    zV=2*trainLabelV-1
    H=kernelM*vrow(zV)*vcol(zV)

    def SVMObj(alfaV):
        
        return vrow(alfaV)@H@vcol(alfaV)/2 - alfaV.sum()
    
    def SVMGradient(alfaV):

        return (H@vcol(alfaV) - 1 ).ravel()
    
    return (SVMObj,SVMGradient)
    

def expKernelSVMSolve(trainDataV,trainLabelV,C,eps=0.0,gamma=0.0,p0=PC[0]):
    #C, eps, const are hyperparameters of the model
    N_RECORDS=trainDataV.shape[1]
    p1=1-p0
    SVMobj=expKernelSVMWrap(trainDataV,trainLabelV,eps=eps,gamma=gamma)#function
    alfaVopt=numpy.zeros((N_RECORDS,))#params (to modify)
    alfaLimits=[]
    rebalanceTermV=numpy.where(trainLabelV==0,p0/PC[0],p1/PC[1])

    for i in range(N_RECORDS):
        alfaLimits.append((0,C*rebalanceTermV[i]))
    (alfaVopt,Jalfamin,_)=scipy.optimize.fmin_l_bfgs_b(SVMobj[0],alfaVopt,fprime=SVMobj[1],bounds=alfaLimits,factr=1.0)#optimize paramiters
    
    return (alfaVopt,Jalfamin)# return alfaV,loss (dual) , also JAlfamin

def inferClassExpSVM(trainDataV,trainLabelV,testDataV,testLabelV,alfaV,eps=0.0,gamma=1.0):
    N_RECORDS=testDataV.shape[1]

    zV=2*trainLabelV-1

    scoreV = numpy.zeros((N_RECORDS,))

    for t in range(N_RECORDS):
        #modify kernel
        kernelVT=numpy.exp(-gamma*numpy.linalg.norm(trainDataV-vcol(testDataV[:,t]),axis=0)**2)+eps
        scoreV[t] = (vrow(zV)*vrow(alfaV)*kernelVT).sum(axis=1)
    
    predictedLabelV=numpy.where(scoreV>0,1,0)
    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (scoreV,predictedLabelV,acc)

def test_SVM_primal(trainDataV, trainLabelV):

    SVMPrimalSolve(trainDataV, trainLabelV,1,0.1)
    SVMPrimalSolve(trainDataV, trainLabelV,1,1.0)
    SVMPrimalSolve(trainDataV, trainLabelV,1,10.0)
    SVMPrimalSolve(trainDataV, trainLabelV,10,0.1)
    SVMPrimalSolve(trainDataV, trainLabelV,10,1.0)
    SVMPrimalSolve(trainDataV, trainLabelV,10,10.0)


def test_SVM_primal_dual(trainDataV, trainLabelV, testDataV, testLabelV):

    (alfaV,Jwbp)=SVMDualSolve(trainDataV, trainLabelV,1,0.1)
    (wb,loss)=recoverPrimal(trainDataV,trainLabelV,alfaV,1,0.1)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (1,0.1,Jwbp,loss,loss+Jwbp))
    (_,_,acc)=inferClassLinearSVM(wb,testDataV,testLabelV,1)
    print("error rate: %f" % (1-acc))

    (alfaV,Jwbp)=SVMDualSolve(trainDataV, trainLabelV,1,1.0)
    (wb,loss)=recoverPrimal(trainDataV,trainLabelV,alfaV,1,1.0)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (1,1,Jwbp,loss,loss+Jwbp))
    (_,_,acc)=inferClassLinearSVM(wb,testDataV,testLabelV,1)
    print("error rate: %f" % (1-acc))

    (alfaV,Jwbp)=SVMDualSolve(trainDataV, trainLabelV,1,10.0)
    (wb,loss)=recoverPrimal(trainDataV,trainLabelV,alfaV,1,10.0)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (1,10,Jwbp,loss,loss+Jwbp))
    (_,_,acc)=inferClassLinearSVM(wb,testDataV,testLabelV,1)
    print("error rate: %f" % (1-acc))

    (alfaV,Jwbp)=SVMDualSolve(trainDataV, trainLabelV,10,0.1)
    (wb,loss)=recoverPrimal(trainDataV,trainLabelV,alfaV,10,0.1)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (10,0.1,Jwbp,loss,loss+Jwbp))
    (_,_,acc)=inferClassLinearSVM(wb,testDataV,testLabelV,10)
    print("error rate: %f" % (1-acc))

    (alfaV,Jwbp)=SVMDualSolve(trainDataV, trainLabelV,10,1.0)
    (wb,loss)=recoverPrimal(trainDataV,trainLabelV,alfaV,10,1.0)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (10,1,Jwbp,loss,loss+Jwbp))
    (_,_,acc)=inferClassLinearSVM(wb,testDataV,testLabelV,10)
    print("error rate: %f" % (1-acc))

    (alfaV,Jwbp)=SVMDualSolve(trainDataV, trainLabelV,10,10.0)
    (wb,loss)=recoverPrimal(trainDataV,trainLabelV,alfaV,10,10.0)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (10,10,Jwbp,loss,loss+Jwbp))
    (_,_,acc)=inferClassLinearSVM(wb,testDataV,testLabelV,10)
    print("error rate: %f" % (1-acc))

def test_SVM_kernel(trainDataV, trainLabelV, testDataV, testLabelV):
    #polinomials --------------------------------

    (alfaV,loss)=polinKernelSVMSolve(trainDataV, trainLabelV, 1.0, eps=0.0, degree=2, const=0.0)
    print("eps: 0.0, const:0.0, dual loss: %f" % loss)
    (_,_,acc)=inferClassPolinSVM(trainDataV,trainLabelV,testDataV,testLabelV, alfaV, eps=0.0, degree=2, const=0.0)
    print("error rate: %f" % (1-acc))

    (alfaV,loss)=polinKernelSVMSolve(trainDataV, trainLabelV, 1.0, eps=1.0, degree=2, const=0.0)
    print("eps: 1.0, const:0.0, dual loss: %f" % loss)
    (_,_,acc)=inferClassPolinSVM(trainDataV,trainLabelV,testDataV,testLabelV, alfaV, eps=1.0, degree=2, const=0.0)
    print("error rate: %f" % (1-acc))

    (alfaV,loss)=polinKernelSVMSolve(trainDataV, trainLabelV, 1.0, eps=0.0, degree=2, const=1.0)
    print("eps: 0.0, const:1.0, dual loss: %f" % loss)
    (_,_,acc)=inferClassPolinSVM(trainDataV,trainLabelV,testDataV,testLabelV, alfaV, eps=0.0, degree=2, const=1.0)
    print("error rate: %f" % (1-acc))

    (alfaV,loss)=polinKernelSVMSolve(trainDataV, trainLabelV, 1.0, eps=1.0, degree=2, const=1.0)
    print("eps: 1.0, const:1.0, dual loss: %f" % loss)
    (_,_,acc)=inferClassPolinSVM(trainDataV,trainLabelV,testDataV,testLabelV, alfaV, eps=1.0, degree=2, const=1.0)
    print("error rate: %f" % (1-acc))
    #exponentials ----------------------------------

    (alfaV,loss)=expKernelSVMSolve(trainDataV, trainLabelV, 1.0, eps=0.0, gamma=1.0)
    print("eps: 0.0, gamma: 1.0, dual loss: %f" % loss)
    (_,_,acc)=inferClassExpSVM(trainDataV,trainLabelV,testDataV,testLabelV, alfaV, eps=0.0, gamma=1.0)
    print("error rate: %f" % (1-acc))

    (alfaV,loss)=expKernelSVMSolve(trainDataV, trainLabelV, 1.0, eps=0.0, gamma=10.0)
    print("eps: 0.0, gamma: 10.0, dual loss: %f" % loss)
    (_,_,acc)=inferClassExpSVM(trainDataV,trainLabelV,testDataV,testLabelV, alfaV, eps=0.0, gamma=10.0)
    print("error rate: %f" % (1-acc))

    (alfaV,loss)=expKernelSVMSolve(trainDataV, trainLabelV, 1.0, eps=1.0, gamma=1.0)
    print("eps: 1.0, gamma: 1.0, dual loss: %f" % loss)
    (_,_,acc)=inferClassExpSVM(trainDataV,trainLabelV,testDataV,testLabelV, alfaV, eps=1.0, gamma=1.0)
    print("error rate: %f" % (1-acc))

    (alfaV,loss)=expKernelSVMSolve(trainDataV, trainLabelV, 1.0, eps=1.0, gamma=10.0)
    print("eps: 1.0, gamma: 10.0, dual loss: %f" % loss)
    (_,_,acc)=inferClassExpSVM(trainDataV,trainLabelV,testDataV,testLabelV, alfaV, eps=1.0, gamma=10.0)
    print("error rate: %f" % (1-acc))

if __name__ == "__main__":
    
    attributes , label = load_iris_binary()
    (trainDataV,trainLabelV),(testDataV,testLabelV) = split_db_2tol(attributes,label)

    test_SVM_primal(trainDataV,trainLabelV)
    test_SVM_primal_dual(trainDataV,trainLabelV,testDataV,testLabelV)
    test_SVM_kernel(trainDataV,trainLabelV,testDataV,testLabelV)
    