import numpy
import matplotlib.pyplot as plt
import numpy.linalg
import scipy.linalg
import scipy.special

labelToN={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
nToLabel=['Iris-setosa','Iris-versicolor','Iris-virginica']
attributeToN={'Sepal-length':0,'Sepal-width':1,'Petal-length':2,'Petal-width':3}
nToAttribute=['Sepal-length','Sepal-width','Petal-length','Petal-width']
PC=[1/3,1/3,1/3]
FILENAME="iris.csv"
TEST=1

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

def within_class_covariance_M(data,label):
    N=label.shape[0]
    Sw=numpy.zeros((data.shape[0],data.shape[0]))
    for c in range(len(nToLabel)):
        elementOfC=data[:,label==c]
        nc=elementOfC.shape[1]
        Sw+=(numpy.cov(elementOfC,bias=True)*nc)/N
    return Sw

def MVG(trainData,trainLabel):
    #mvg parameters
    muc=numpy.zeros((trainData.shape[0],1,len(nToLabel)))
    for i in range(len(nToLabel)):
        muc[:,0,i]=trainData[:,trainLabel==i].mean(axis=1)
    Cc=numpy.zeros((trainData.shape[0],trainData.shape[0],len(nToLabel)))
    for i in range(len(nToLabel)):#cost N_ATTRxN_ATTRxN_DATA
        Cc[:,:,i]=numpy.cov(trainData[:,trainLabel==i],bias=True)
    return (muc,Cc)

def MVG_NaiveBayes(trainData,trainLabel):#compute only variances
    #mvg parameters
    muc=numpy.zeros((trainData.shape[0],1,len(nToLabel)))
    for i in range(len(nToLabel)):
        muc[:,0,i]=trainData[:,trainLabel==i].mean(axis=1)
    Cc=numpy.zeros((trainData.shape[0],len(nToLabel)))
    for i in range(len(nToLabel)):#cost N_ATTRxN_DATA
        Cc[:,i]=numpy.var(trainData[:,trainLabel==i],axis=1)
    return (muc,Cc)

def MVG_tied(trainData,trainLabel):#compute only one (unique for each class) covariance matrix
    muc=numpy.zeros((trainData.shape[0],1,len(nToLabel)))
    for i in range(len(nToLabel)):
        muc[:,0,i]=trainData[:,trainLabel==i].mean(axis=1)
    Cc=within_class_covariance_M(trainData,trainLabel)#cost N_ATTRxN_ATTRxN_DATA maybe already calculated
    return (muc,Cc)
    
def MVG_naiveBayes_tied(trainData,trainLabel):#compute only one (unique for each class) variances
    muc=numpy.zeros((trainData.shape[0],1,len(nToLabel)))
    for i in range(len(nToLabel)):
        muc[:,0,i]=trainData[:,trainLabel==i].mean(axis=1)
    Cc=numpy.zeros((trainData.shape[0]))
    for i in range(len(nToLabel)):#cost N_ATTRxN_DATA
        elementOfC=trainData[:,trainLabel==i]
        nc=elementOfC.shape[1]
        Cc+=numpy.var(elementOfC,axis=1)*nc/trainData.shape[1]
    return (muc,Cc)

def GAU_ND_pdf(X,mu,C):#compute only one 
    XC=X-mu
    M=X.shape[0]
    const=(numpy.pi*2)**(-0.5*M)
    det=numpy.linalg.det(C)
    L=numpy.linalg.inv(C)
    #for i in range(data.shape[1]):
    #    Pdata[i]=-1/2*XC[:,i].T@L@XC[:,i]
    #efficient way
    v=(XC*(L@XC)).sum(axis=0)
    return const*(det**-0.5)*numpy.exp(-0.5*v)

def GAU_ND_logpdf(X,mu,C):
    XC=X-mu
    M=X.shape[0]
    const=-0.5*M*numpy.log(2*numpy.pi)
    logdet=numpy.linalg.slogdet(C)[1]
    L=numpy.linalg.inv(C)
    #for i in range(data.shape[1]):
    #    Pdata[i]=-1/2*XC[:,i].T@L@XC[:,i]
    #efficient way
    v=(XC*(L@XC)).sum(axis=0)
    return const-0.5*logdet-0.5*v

def GAU_ND_pdf_naiveBayes(X,mu,C):#C is diagonal (diagonal vector)-> less computational expensive
    M=X.shape[0]
    const=(numpy.pi*2)**(-0.5*M)
    det=C.prod(axis=0)*((-1)**M)
    XC=X-mu
    L=1/C
    v=numpy.zeros(X.shape[1])
    #for i in range(data.shape[1]):#better way than explicit iteration?
    #    for j in range(data.shape[0]):#cost N_ATTRxN_DATA
    #        v[i]+=XC[j,i]*L[j]*XC[j,i]
    #efficient way
    v=(XC**2*vcol(L)).sum(axis=0)
    return const*(det**-0.5)*numpy.exp(-0.5*v)

def GAU_ND_logpdf_naiveBayes(X,mu,C):#C is diagonal (diagonal vector)-> less computational expensive
    M=X.shape[0]
    const=-0.5*M*numpy.log(2*numpy.pi)
    logdet=numpy.log(C).sum(axis=0)
    XC=X-mu
    L=1/C
    v=numpy.zeros(X.shape[1])
    #for i in range(data.shape[1]):
    #    for j in range(data.shape[0]):
    #        v[i]+=(XC[j,i]*L[j]*XC[j,i])
    #efficient way
    v=(XC**2*vcol(L)).sum(axis=0)
    return const-0.5*logdet-0.5*v

#if output test is active in inferClass the number of operation increase by 2 op(if) for each bin
#so worst case is if i use leave one out, but this approach is used with small dataset and the 
#bigger part of computation is due to training

def inferClass(testData,testLabel,muc,Cc):
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]=GAU_ND_pdf(testData,muc[:,:,i],Cc[:,:,i])
    
    SJoint=S*vcol(numpy.array(PC))
    SMarg=vrow(SJoint.sum(axis=0))    
    SPost=SJoint/SMarg
    predictedLabel=numpy.argmax(SPost,axis=0)

    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])

    if(TEST==1):
        print("error MVG posterior")
        posterior_MVG=numpy.load('solutions/Posterior_MVG.npy')
        print((posterior_MVG-SPost).max())
        print("error MVG joint")
        joint_MVG=numpy.load('solutions/SJoint_MVG.npy')
        print((joint_MVG-SJoint).max())

    return (predictedLabel,acc)

def inferClassLog(testData,testLabel,muc,Cc,VJoint=[0,0,0]):
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]=GAU_ND_logpdf(testData,muc[:,:,i],Cc[:,:,i])
    SJoint=S+numpy.log(vcol(numpy.array(PC))) #use broadcasting (4,1)->(4,50)
    #l=SJoint.argmax(axis=0)
    #SMarg=l+numpy.log((numpy.exp(SJoint-l).sum(axis=0)))
    SMarg=scipy.special.logsumexp(SJoint,axis=0)
    SPost=SJoint-SMarg
    predictedLabel=numpy.argmax(SPost,axis=0)
    
    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])

    if(TEST==1):
        print("error MVG log posterior")
        posterior=numpy.load('solutions/logPosterior_MVG.npy')
        print((posterior-SPost).max())
        print("error MVG log joint")
        joint=numpy.load('solutions/logSJoint_MVG.npy')
        print((joint-SJoint).max())
        print("error MVG log marginal")
        marginal=numpy.load('solutions/logMarginal_MVG.npy')
        print((marginal-SMarg).max())

    if(TEST==2):
        print("error MVG Leave One Out log joint")
        print((VJoint-SJoint).max())

    return (predictedLabel,acc)

def inferClass_naiveBayes(testData,testLabel,muc,Cc):#more accurate features of my data are uncorrelated (converge with less training data)
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]=GAU_ND_pdf_naiveBayes(testData,muc[:,:,i],Cc[:,i])
    SJoint=S*vcol(numpy.array(PC))
    SMarg=vrow(SJoint.sum(axis=0))    
    SPost=SJoint/SMarg
    predictedLabel=numpy.argmax(SPost,axis=0)

    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])

    if(TEST==1):
        print("error NaiveBayes posterior")
        posterior=numpy.load('solutions/Posterior_NaiveBayes.npy')
        print((posterior-SPost).max())
        print("error NaiveBayes joint")
        joint=numpy.load('solutions/SJoint_NaiveBayes.npy')
        print((joint-SJoint).max())

    return (predictedLabel,acc)

def inferClassLog_naiveBayes(testData,testLabel,muc,Cc,VJoint=[0,0,0]):
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]=GAU_ND_logpdf_naiveBayes(testData,muc[:,:,i],Cc[:,i])
    SJoint=S+numpy.log(vcol(numpy.array(PC))) #use broadcasting (4,1)->(4,50)
    #l=SJoint.argmax(axis=0)
    #SMarg=l+numpy.log((numpy.exp(SJoint-l).sum(axis=0)))
    SMarg=scipy.special.logsumexp(SJoint,axis=0)
    SPost=SJoint-SMarg
    predictedLabel=numpy.argmax(SPost,axis=0)
    
    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])

    if(TEST==1):
        print("error NaiveBayes log posterior")
        posterior=numpy.load('solutions/logPosterior_NaiveBayes.npy')
        print((posterior-SPost).max())
        print("error NaiveBayes log joint")
        joint=numpy.load('solutions/logSJoint_NaiveBayes.npy')
        print((joint-SJoint).max())
        print("error NaiveBayes log marginal")
        marginal=numpy.load('solutions/logMarginal_NaiveBayes.npy')
        print((marginal-SMarg).max())

    if(TEST==2):
        print("error NaiveBayes Leave One Out log joint")
        print((VJoint-SJoint).max())

    return (predictedLabel,acc)

def inferClass_tied(testData,testLabel,muc,Cc):
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]=GAU_ND_pdf(testData,muc[:,:,i],Cc)
    SJoint=S*vcol(numpy.array(PC))
    SMarg=vrow(SJoint.sum(axis=0))    
    SPost=SJoint/SMarg
    predictedLabel=numpy.argmax(SPost,axis=0)

    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])

    if(TEST==1):
        print("error Tied MVG posterior")
        posterior=numpy.load('solutions/Posterior_TiedMVG.npy')
        print((posterior-SPost).max())
        print("error Tied MVG joint")
        joint=numpy.load('solutions/SJoint_TiedMVG.npy')
        print((joint-SJoint).max())

    return (predictedLabel,acc)

def inferClassLog_tied(testData,testLabel,muc,Cc,VJoint=[0,0,0]):
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]=GAU_ND_logpdf(testData,muc[:,:,i],Cc)
    SJoint=S+numpy.log(vcol(numpy.array(PC))) #use broadcasting (4,1)->(4,50)
    #l=SJoint.argmax(axis=0)
    #SMarg=l+numpy.log((numpy.exp(SJoint-l).sum(axis=0)))
    SMarg=scipy.special.logsumexp(SJoint,axis=0)
    SPost=SJoint-SMarg
    predictedLabel=numpy.argmax(SPost,axis=0)
    
    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])

    if(TEST==1):
        print("error Tied MVG log posterior")
        posterior=numpy.load('solutions/logPosterior_TiedMVG.npy')
        print((posterior-SPost).max())
        print("error Tied MVG log joint")
        joint=numpy.load('solutions/logSJoint_TiedMVG.npy')
        print((joint-SJoint).max())
        print("error Tied MVG log marginal")
        marginal=numpy.load('solutions/logMarginal_TiedMVG.npy')
        print((marginal-SMarg).max())

    if(TEST==2):
        print("error Tied MVG Leave One Out log joint")
        print((VJoint-SJoint).max())
    
    return (predictedLabel,acc)

def inferClass_naiveBayes_tied(testData,testLabel,muc,Cc):#more accurate features of my data are uncorrelated (converge with less training data)
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]=GAU_ND_pdf_naiveBayes(testData,muc[:,:,i],Cc)
    SJoint=S*vcol(numpy.array(PC))
    SMarg=vrow(SJoint.sum(axis=0))    
    SPost=SJoint/SMarg
    predictedLabel=numpy.argmax(SPost,axis=0)

    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])

    if(TEST==1):
        print("error TiedNaiveBayes posterior")
        posterior=numpy.load('solutions/Posterior_TiedNaiveBayes.npy')
        print((posterior-SPost).max())
        print("error TiedNaiveBayes joint")
        joint=numpy.load('solutions/SJoint_TiedNaiveBayes.npy')
        print((joint-SJoint).max())

    return (predictedLabel,acc)

def inferClassLog_naiveBayes_tied(testData,testLabel,muc,Cc,VJoint=[0,0,0]):
    S=numpy.zeros((len(nToLabel),testData.shape[1]))
    for i in range(len(nToLabel)):
        S[i,:]=GAU_ND_logpdf_naiveBayes(testData,muc[:,:,i],Cc)
    SJoint=S+numpy.log(vcol(numpy.array(PC))) #use broadcasting (4,1)->(4,50)
    #l=SJoint.argmax(axis=0)
    #SMarg=l+numpy.log((numpy.exp(SJoint-l).sum(axis=0)))
    SMarg=scipy.special.logsumexp(SJoint,axis=0)
    SPost=SJoint-SMarg
    predictedLabel=numpy.argmax(SPost,axis=0)
    
    A=predictedLabel==testLabel
    acc=A.sum()/float(testData.shape[1])

    if(TEST==1):
        print("error TiedNaiveBayes log posterior")
        posterior=numpy.load('solutions/logPosterior_TiedNaiveBayes.npy')
        print((posterior-SPost).max())
        print("error TiedNaiveBayes log joint")
        joint=numpy.load('solutions/logSJoint_TiedNaiveBayes.npy')
        print((joint-SJoint).max())
        print("error TiedNaiveBayes log marginal")
        marginal=numpy.load('solutions/logMarginal_TiedNaiveBayes.npy')
        print((marginal-SMarg).max())

    if(TEST==2):
        print("error TiedNaiveBayes Leave One Out log joint")
        print((VJoint-SJoint).max())
    
    return (predictedLabel,acc)

def KFold(D,L,k,seed=0,type=0):##type: 0 = MVG, 1 = Naive-Bayes, 2 = tied Cov, 3 = tied Cov+naive-Bayes
    nFold=int(D.shape[1]/k)
    numpy.random.seed(seed)
    idx=numpy.random.permutation(D.shape[1])
    acc=0.0

    match type:
        case 4:
            LOOJoints=numpy.load('solutions/LOO_logSJoint_MVG.npy')
        case 5:
            LOOJoints=numpy.load('solutions/LOO_logSJoint_NaiveBayes.npy')
        case 6:
            LOOJoints=numpy.load('solutions/LOO_logSJoint_TiedMVG.npy')
        case 7:
            LOOJoints=numpy.load('solutions/LOO_logSJoint_TiedNaiveBayes.npy')

    for i in range(k):

        idxTrain=numpy.zeros(D.shape[1]-nFold,dtype=numpy.int32)
        idxTrain[:i*nFold]=idx[:i*nFold]
        idxTrain[i*nFold:]=idx[(i+1)*nFold:]
        idxTest=idx[i*nFold:(i+1)*nFold]
        trainData=D[:,idxTrain]
        testData=D[:,idxTest]
        trainLabel=L[idxTrain]
        testLabel=L[idxTest]

        match type:#Cc is in different shape
            case 0:
                (muc,Cc)=MVG(trainData,trainLabel)#Cc.shape=(n_attr,n_attr,n_class)
                (_,partialAcc)=inferClass(testData,testLabel,muc,Cc)
            case 1:
                (muc,Cc)=MVG_NaiveBayes(trainData,trainLabel)#Cc.shape=(n_attr,n_class)
                (_,partialAcc)=inferClass_naiveBayes(testData,testLabel,muc,Cc)
            case 2:
                (muc,Cc)=MVG_tied(trainData,trainLabel)#Cc.shape=(n_attr,n_attr)
                (_,partialAcc)=inferClass_tied(testData,testLabel,muc,Cc)
            case 3:
                (muc,Cc)=MVG_naiveBayes_tied(trainData,trainLabel)#Cc.shape=(n_attr)
                (_,partialAcc)=inferClass_naiveBayes_tied(testData,testLabel,muc,Cc)
            case 4:
                (muc,Cc)=MVG(trainData,trainLabel)#Cc.shape=(n_attr,n_attr,n_class)
                (_,partialAcc)=inferClassLog(testData,testLabel,muc,Cc,VJoint=vcol(LOOJoints[:,idx[i]]))
            case 5:
                (muc,Cc)=MVG_NaiveBayes(trainData,trainLabel)#Cc.shape=(n_attr,n_class)
                (_,partialAcc)=inferClassLog_naiveBayes(testData,testLabel,muc,Cc,VJoint=vcol(LOOJoints[:,idx[i]]))
            case 6:
                (muc,Cc)=MVG_tied(trainData,trainLabel)#Cc.shape=(n_attr,n_attr)
                (_,partialAcc)=inferClassLog_tied(testData,testLabel,muc,Cc,VJoint=vcol(LOOJoints[:,idx[i]]))
            case 7:
                (muc,Cc)=MVG_naiveBayes_tied(trainData,trainLabel)#Cc.shape=(n_attr)
                (_,partialAcc)=inferClassLog_naiveBayes_tied(testData,testLabel,muc,Cc,VJoint=vcol(LOOJoints[:,idx[i]]))

        acc+=partialAcc

    acc/=float(k)
    print("total accuracy:%f\n"%acc)
    

if __name__=='__main__':
    labeledData=load(FILENAME)
    (trainData,trainLabel),(testData,testLabel)=split_db_2tol(labeledData.dsAttributes,labeledData.dsLabel)
    if(TEST==1):
        (muc,Cc)=MVG(trainData,trainLabel)
        inferClass(testData,testLabel,muc,Cc)
        inferClassLog(testData,testLabel,muc,Cc)
        (muc,Cc)=MVG_NaiveBayes(trainData,trainLabel)
        inferClass_naiveBayes(testData,testLabel,muc,Cc)
        inferClassLog_naiveBayes(testData,testLabel,muc,Cc)
        (muc,Cc)=MVG_tied(trainData,trainLabel)
        inferClass_tied(testData,testLabel,muc,Cc)
        inferClassLog_tied(testData,testLabel,muc,Cc)
        (muc,Cc)=MVG_naiveBayes_tied(trainData,trainLabel)
        inferClass_naiveBayes_tied(testData,testLabel,muc,Cc)
        inferClassLog_naiveBayes_tied(testData,testLabel,muc,Cc)
    else:
        print("leave one out MVG:")
        KFold(labeledData.dsAttributes,labeledData.dsLabel,150,seed=0,type=0)
        print("leave one out Naive-Bayes:")
        KFold(labeledData.dsAttributes,labeledData.dsLabel,150,seed=0,type=1)
        print("leave one out tied covariance:")
        KFold(labeledData.dsAttributes,labeledData.dsLabel,150,seed=0,type=2)
        print("leave one out Naive-Bayes tied covariance:")
        KFold(labeledData.dsAttributes,labeledData.dsLabel,150,seed=0,type=3)
        print("leave one out MVG log:")
        KFold(labeledData.dsAttributes,labeledData.dsLabel,150,seed=0,type=4)
        print("leave one out Naive-Bayes log:")
        KFold(labeledData.dsAttributes,labeledData.dsLabel,150,seed=0,type=5)
        print("leave one out tied covariance log:")
        KFold(labeledData.dsAttributes,labeledData.dsLabel,150,seed=0,type=6)
        print("leave one out Naive_Bayes tied covariance log:")
        KFold(labeledData.dsAttributes,labeledData.dsLabel,150,seed=0,type=7)
    
    print("epsilon of float (numpy.float64) ")
    print(numpy.finfo(float).eps)