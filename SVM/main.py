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

def SVM_primal_solve(trainData,trainLabel,K,C):
   
    SVM_obj=SVM_primal_obj_wrap(trainData,trainLabel,K,C)#function
    wbopt=numpy.zeros((trainData.shape[0]+1,))#params (to modify)
    (wbopt,Jwbmin,_)=scipy.optimize.fmin_l_bfgs_b(SVM_obj,wbopt,approx_grad=True,factr=1.0)#optimize paramiters
    print("C: %f, K: %f, primal loss: %f" % (C,K,Jwbmin))#check min result
    return (wbopt[0:-1], wbopt[-1])# return w,b


def SVM_primal_obj_wrap(trainData,trainLabel,K,C):#useful closure for defining at runtime parameters that we don't vary in order to maximize
    
    expandedData=numpy.zeros((trainData.shape[0]+1,trainData.shape[1]))
    expandedData[0:-1,:]=trainData
    expandedData[-1,:]=numpy.ones((trainData.shape[1],))*K
    G=expandedData.T@expandedData
    zi=2*trainLabel-1
    H=G*vrow(zi)*vcol(zi)
    
    def SVM_primal_obj(wb):#wb=4+1 for iris dataset
        
        regterm=(numpy.linalg.norm(wb)**2)/2
        hingelosses=numpy.maximum(numpy.zeros(expandedData.shape[1]),1-zi*(vrow(wb)@expandedData))
        return regterm+C*numpy.sum(hingelosses)
    
    return SVM_primal_obj


def SVM_dual_solve(trainData,trainLabel,K,C):
   
    SVM_obj=SVM_dual_wrap(trainData,trainLabel,K,C)#function
    alfaVopt=numpy.zeros((trainData.shape[1],))#params (to modify)
    alfaLimits=[]
    for i in range(trainData.shape[1]):
        alfaLimits.append((0,C))
    (alfaVopt,Jwbmin,_)=scipy.optimize.fmin_l_bfgs_b(SVM_obj[0],alfaVopt,fprime=SVM_obj[1],bounds=alfaLimits,factr=1.0)#optimize paramiters
    
    return (alfaVopt,Jwbmin)# return w,b

def SVM_dual_wrap(trainData,trainLabel,K,C):#useful closure for defining at runtime parameters that we don't vary in order to maximize
    
    expandedData=numpy.zeros((trainData.shape[0]+1,trainData.shape[1]))
    expandedData[0:-1,:]=trainData
    expandedData[-1,:]=numpy.ones((trainData.shape[1],))*K
    G=expandedData.T@expandedData
    zi=2*trainLabel-1
    H=G*vrow(zi)*vcol(zi)

    def SVM_dual_obj(alfaV):#wb=4+1 for iris dataset
        
        return vrow(alfaV)@H@vcol(alfaV)/2 - alfaV.sum()
    
    def SVM_dual_gradient(alfaV):

        return (H@vcol(alfaV) - 1 ).reshape((alfaV.shape[0],))
    
    return (SVM_dual_obj,SVM_dual_gradient)

def recover_primal(trainData,trainLabel,alfaV,K,C):
    expandedData=numpy.zeros((trainData.shape[0]+1,trainData.shape[1]))
    expandedData[0:-1,:]=trainData
    expandedData[-1,:]=numpy.ones((trainData.shape[1],))*K
    zi=2*trainLabel-1
    wb = numpy.sum(alfaV*zi*expandedData,axis=1)
    regterm=(numpy.linalg.norm(wb)**2)/2
    hingelosses=numpy.maximum(numpy.zeros(expandedData.shape[1]),1-zi*(vrow(wb)@expandedData))
    Jw = regterm+C*numpy.sum(hingelosses)
    return (wb,Jw)

def infer_class_SVM(wb,testData,testLabel,K):

    w,b = wb[0:-1],wb[-1]
    scoreV=vrow(w)@testData+b*K
    predictedLabel=numpy.where(scoreV>0,1,0)
    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])

    return (scoreV,predictedLabel,acc)

def test_SVM_primal(trainData, trainLabel):

    SVM_primal_solve(trainData, trainLabel,1,0.1)
    SVM_primal_solve(trainData, trainLabel,1,1.0)
    SVM_primal_solve(trainData, trainLabel,1,10.0)
    SVM_primal_solve(trainData, trainLabel,10,0.1)
    SVM_primal_solve(trainData, trainLabel,10,1.0)
    SVM_primal_solve(trainData, trainLabel,10,10.0)


def test_SVM_primal_dual(trainData, trainLabel, testData, testLabel):

    (alfaV,Jwp)=SVM_dual_solve(trainData, trainLabel,1,0.1)
    (wb,Jwd)=recover_primal(trainData,trainLabel,alfaV,1,0.1)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (1,0.1,Jwp,Jwd,Jwd+Jwp))
    (_,_,acc)=infer_class_SVM(wb,testData,testLabel,1)
    print("error rate: %f" % (1-acc))

    (alfaV,Jwp)=SVM_dual_solve(trainData, trainLabel,1,1.0)
    (wb,Jwd)=recover_primal(trainData,trainLabel,alfaV,1,1.0)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (1,1,Jwp,Jwd,Jwd+Jwp))
    (_,_,acc)=infer_class_SVM(wb,testData,testLabel,1)
    print("error rate: %f" % (1-acc))

    (alfaV,Jwp)=SVM_dual_solve(trainData, trainLabel,1,10.0)
    (wb,Jwd)=recover_primal(trainData,trainLabel,alfaV,1,10.0)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (1,10,Jwp,Jwd,Jwd+Jwp))
    (_,_,acc)=infer_class_SVM(wb,testData,testLabel,1)
    print("error rate: %f" % (1-acc))

    (alfaV,Jwp)=SVM_dual_solve(trainData, trainLabel,10,0.1)
    (wb,Jwd)=recover_primal(trainData,trainLabel,alfaV,10,0.1)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (10,0.1,Jwp,Jwd,Jwd+Jwp))
    (_,_,acc)=infer_class_SVM(wb,testData,testLabel,10)
    print("error rate: %f" % (1-acc))

    (alfaV,Jwp)=SVM_dual_solve(trainData, trainLabel,10,1.0)
    (wb,Jwd)=recover_primal(trainData,trainLabel,alfaV,10,1.0)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (10,1,Jwp,Jwd,Jwd+Jwp))
    (_,_,acc)=infer_class_SVM(wb,testData,testLabel,10)
    print("error rate: %f" % (1-acc))

    (alfaV,Jwp)=SVM_dual_solve(trainData, trainLabel,10,10.0)
    (wb,Jwd)=recover_primal(trainData,trainLabel,alfaV,10,10.0)
    print("C: %f, K: %f, prial loss: %f,dual loss: %f, gap %f " % (10,10,Jwp,Jwd,Jwd+Jwp))
    (_,_,acc)=infer_class_SVM(wb,testData,testLabel,10)
    print("error rate: %f" % (1-acc))



if __name__ == "__main__":
    
    attributes , label = load_iris_binary()
    (trainData,trainLabel),(testData,testLabel) = split_db_2tol(attributes,label)

    test_SVM_primal(trainData,trainLabel)
    test_SVM_primal_dual(trainData,trainLabel,testData,testLabel)

    