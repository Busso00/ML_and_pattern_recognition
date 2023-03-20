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
    
    print("class means")
    for i in range(len(nToLabel)):
        print("mu%d"%i)
        print(muc[:,:,i])
    print("class covariance matrix")
    for i in range(len(nToLabel)):
        print("C%d"%i)
        print(Cc[:,:,i])

    return (muc,Cc)

def NDGaussian(data,mu,C):
    Pdata=numpy.zeros((data.shape[1],))
    N=data.shape[1]
    Pdata+=numpy.power(numpy.pi,-N/2.0)
    Pdata*=numpy.power(numpy.linalg.det(C),-0.5)
    centered_data=data-mu
    for i in range(data.shape[1]):    
        Pdata[i]*=numpy.exp(-0.5*(centered_data[:,i]).T@numpy.linalg.inv(C)@(centered_data[:,i]))

    return Pdata

def inferClass(testData,testLabel):
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]+=NDGaussian(testData,muc[:,:,i],Cc[:,:,i])
    SJoint=S*numpy.array(PC).reshape((len(nToLabel),1))
    SMarg=SJoint.sum(axis=0).reshape((1,testData.shape[1]))    
    SPost=SJoint/SMarg
    PredictedLabel=numpy.argmax(SPost,axis=0)

    A=PredictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])
    print("error rate: %f"%(1-acc))

def inferClassLog(testData,testLabel):
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]+=GAU_ND_logpdf(testData,muc[:,:,i],Cc[:,:,i])
    SJoint=S+numpy.log(numpy.array(PC).reshape((len(nToLabel),1))) #use broadcasting *(4,1)->*(4,50)
    l=SJoint.argmax(axis=0)
    SPost=SJoint-(l+numpy.log((numpy.exp(SJoint-l).sum(axis=0))))
    PredictedLabel=numpy.argmax(SPost,axis=0)
    
    A=PredictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])
    print("error rate: %f"%(1-acc))
            
    

if __name__=='__main__':
    labeledData=load(FILENAME)
    (trainData,trainLabel),(testData,testLabel)=split_db_2tol(labeledData.dsAttributes,labeledData.dsLabel)
    (muc,Cc)=MVG(trainData,trainLabel)
    inferClass(testData,testLabel)
    inferClassLog(testData,testLabel)