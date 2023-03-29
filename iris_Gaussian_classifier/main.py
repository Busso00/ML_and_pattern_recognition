import numpy
import matplotlib.pyplot as plt
import numpy.linalg
import scipy.linalg

labelToN={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
nToLabel=['Iris-setosa','Iris-versicolor','Iris-virginica']
attributeToN={'Sepal-length':0,'Sepal-width':1,'Petal-length':2,'Petal-width':3}
nToAttribute=['Sepal-length','Sepal-width','Petal-length','Petal-width']
PC=[1/3,1/3,1/3]
FILENAME="iris.csv"
ERROR_RATE_ON_SINGLE_FOLD=False

def vcol(v):
    return v.reshape((v.size,1))

def vrow(v):
    return v.reshape((1,v.size))

class DataList:
    def __init__(self):
        self.dsAttributes=[]
        self.dsLabel=[]

class DataArray:
    def __init__(self,listAttr,listLabel):
        self.dsAttributes=numpy.vstack(listAttr).T
        self.dsLabel=numpy.array(listLabel,dtype=numpy.int32)

def load(filename):
    try:
        f=open(filename,'r')
    except:
        print("error opening Iris Dataset")
        exit(-1)
    
    labeledData=DataList()
    for line in f:
        try:
            record=line.split(',')
            attributes=numpy.array([float(i) for i in record[0:-1]])
            label=labelToN[record[-1].strip()]
            labeledData.dsAttributes.append(attributes)
            labeledData.dsLabel.append(label)
        except:
            print("error parsing line")

    labeledData=DataArray(labeledData.dsAttributes,labeledData.dsLabel)
    return labeledData

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

def GAU_ND_logpdf(data,mu,C):#nd
    GAU_ND_log_y=numpy.zeros((data.shape[1],))
    M=data.shape[0]
    SIGMA=numpy.linalg.det(C)
    centered_data=data-mu
    
    GAU_ND_log_y+=-M/2*numpy.log(2*numpy.pi)-1/2*numpy.log(SIGMA)
    for i in range(data.shape[1]):    
        GAU_ND_log_y[i]-=1/2*(centered_data[:,i].T@numpy.linalg.inv(C))@centered_data[:,i]

    return GAU_ND_log_y

def MVG(trainData,trainLabel):
    #mvg parameters
    muc=numpy.zeros((trainData.shape[0],1,len(nToLabel)))
    for i in range(len(nToLabel)):
        muc[:,0,i]+=trainData[:,trainLabel==i].mean(axis=1)
    Cc=numpy.zeros((trainData.shape[0],trainData.shape[0],len(nToLabel)))
    for i in range(len(nToLabel)):
        Cc[:,:,i]+=numpy.cov(trainData[:,trainLabel==i],bias=True)

    return (muc,Cc)

def MVG_NaiveBayes(trainData,trainLabel):
    (muc,Cc)=MVG(trainData,trainLabel)
    for i in range(len(nToLabel)):
        Cc[:,:,i]*=numpy.eye(trainData.shape[0],trainData.shape[0])
    return (muc,Cc)

def within_class_covariance_M(data,label):
    N=label.shape[0]
    Sw=numpy.zeros((data.shape[0],data.shape[0]))
    for c in range(len(nToLabel)):
        elementOfC=data[:,label==c]
        nc=elementOfC.shape[1]
        Sw+=(numpy.cov(elementOfC,bias=True)*nc)/N
    return Sw

def MVG_tied(trainData,trainLabel):
    muc=numpy.zeros((trainData.shape[0],1,len(nToLabel)))
    for i in range(len(nToLabel)):
        muc[:,0,i]+=trainData[:,trainLabel==i].mean(axis=1)
    Cc=within_class_covariance_M(trainData,trainLabel)
    return (muc,Cc)
    

def ND_GAU_pdf(data,mu,C):
    Pdata=numpy.zeros((data.shape[1],))
    N=data.shape[1]
    Pdata+=numpy.power(numpy.pi,-N/2.0)
    Pdata*=numpy.power(numpy.linalg.det(C),-0.5)
    centered_data=data-mu
    for i in range(data.shape[1]):    
        Pdata[i]*=numpy.exp(-0.5*(centered_data[:,i]).T@numpy.linalg.inv(C)@(centered_data[:,i]))

    return Pdata

def inferClass(testData,testLabel,muc,Cc):
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]+=ND_GAU_pdf(testData,muc[:,:,i],Cc[:,:,i])
    SJoint=S*vcol(numpy.array(PC))
    SMarg=vrow(SJoint.sum(axis=0))    
    SPost=SJoint/SMarg
    predictedLabel=numpy.argmax(SPost,axis=0)

    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])
    if ERROR_RATE_ON_SINGLE_FOLD:
        print("error rate: %f"%(1-acc))
    return (predictedLabel,acc)

def inferClassLog(testData,testLabel,muc,Cc):
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]+=GAU_ND_logpdf(testData,muc[:,:,i],Cc[:,:,i])
    SJoint=S+numpy.log(vcol(numpy.array(PC))) #use broadcasting *(4,1)->*(4,50)
    l=SJoint.argmax(axis=0)
    SPost=SJoint-(l+numpy.log((numpy.exp(SJoint-l).sum(axis=0))))
    predictedLabel=numpy.argmax(SPost,axis=0)
    
    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])
    if ERROR_RATE_ON_SINGLE_FOLD:
        print("error rate: %f"%(1-acc))
    return (predictedLabel,acc)

def inferClass_tied(testData,testLabel,muc,Cc):
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]+=ND_GAU_pdf(testData,muc[:,:,i],Cc)
    SJoint=S*vcol(numpy.array(PC))
    SMarg=vrow(SJoint.sum(axis=0))    
    SPost=SJoint/SMarg
    predictedLabel=numpy.argmax(SPost,axis=0)

    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])
    if ERROR_RATE_ON_SINGLE_FOLD:
        print("error rate: %f"%(1-acc))
    return (predictedLabel,acc)
            
def KFold(D,L,k,seed=0,type=0):##type: 0 = MVG, 1 = Naive-Bayesian, 2 = tied Covariance
    nFold=int(D.shape[1]/k)
    numpy.random.seed(seed)
    idx=numpy.random.permutation(D.shape[1])
    acc=0.0
    for i in range(k):
        idxTrain=numpy.zeros(D.shape[1]-nFold,dtype=numpy.int32)
        idxTrain[:i*nFold]=idx[:i*nFold]
        idxTrain[i*nFold:]=idx[(i+1)*nFold:]
        idxTest=idx[i*nFold:(i+1)*nFold]
        trainData=D[:,idxTrain]
        testData=D[:,idxTest]
        trainLabel=L[idxTrain]
        testLabel=L[idxTest]
        if ERROR_RATE_ON_SINGLE_FOLD:
            print("training set:%d"%(i+1))
        match type:
            case 0:
                (muc,Cc)=MVG(trainData,trainLabel)#calculate parameters here
                (_,partialAcc)=inferClass(testData,testLabel,muc,Cc)#check accuracy here
            case 1:
                (muc,Cc)=MVG_NaiveBayes(trainData,trainLabel)
                (_,partialAcc)=inferClass(testData,testLabel,muc,Cc)
            case 2:
                (muc,Cc)=MVG_tied(trainData,trainLabel)
                (_,partialAcc)=inferClass_tied(testData,testLabel,muc,Cc)
        acc+=partialAcc

    acc/=float(k)
    print("total accuracy:%f\n"%acc)
    

if __name__=='__main__':
    labeledData=load(FILENAME)
    #(trainData,trainLabel),(testData,testLabel)=split_db_2tol(labeledData.dsAttributes,labeledData.dsLabel)
    #(muc,Cc)=MVG(trainData,trainLabel)
    #inferClass(testData,testLabel,muc,Cc)
    #inferClassLog(testData,testLabel,muc,Cc)
    print("leave one out MVG:")
    KFold(labeledData.dsAttributes,labeledData.dsLabel,150,seed=0,type=0)
    print("leave one out Naive-Bayes:")
    KFold(labeledData.dsAttributes,labeledData.dsLabel,150,seed=0,type=1)
    print("leave one out tied covariance:")
    KFold(labeledData.dsAttributes,labeledData.dsLabel,150,seed=0,type=2)