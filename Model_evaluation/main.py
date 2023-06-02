import numpy
import matplotlib.pyplot as plt
import numpy.linalg
import scipy.linalg
import scipy.special
#--------------------------------OK----------------------------------------------------
#labelToN={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
#nToLabel=['Iris-setosa','Iris-versicolor','Iris-virginica']
#attributeToN={'Sepal-length':0,'Sepal-width':1,'Petal-length':2,'Petal-width':3}
#nToAttribute=['Sepal-length','Sepal-width','Petal-length','Petal-width']
#PC=[1/3,1/3,1/3]

#FILENAME="iris.csv"
#TEST=0
#--------------------------------------------------------------------------------------

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

def confusionMatrix(testLabels, predictedLabels):
    confM=numpy.zeros((len(nToLabel),len(nToLabel)))
    for i in range(testLabels.shape[0]):
        confM[int(predictedLabels[i])][int(testLabels[i])]+=1
    return confM

def irisModelEvaluation():
    labeledData=load(FILENAME)
    (trainData,trainLabel),(testData,testLabel)=split_db_2tol(labeledData.dsAttributes,labeledData.dsLabel)
    #import of iris_Gaussian_classifier use only split_db_2tol no KFold
    (muc,Cc)=MVG(trainData,trainLabel)#Cc.shape=(n_attr,n_attr,n_class)
    (predictedLabels,_)=inferClass(testData,testLabel,muc,Cc)
    confM=confusionMatrix(testLabel,predictedLabels)
    print(confM)
    
    (muc,Cc)=MVG_tied(trainData,trainLabel)#Cc.shape=(n_attr,n_attr)
    (predictedLabels,_)=inferClass_tied(testData,testLabel,muc,Cc)       
    confM=confusionMatrix(testLabel,predictedLabels)
    print(confM)

labelToN={'Inf':0,'Pur':1,'Par':2}
nToLabel=['Inf','Pur','Par']
PC=[1/3,1/3,1/3]


def inferLabelsCC(condLikehood):
    return numpy.argmax(condLikehood,axis=0)

def divinaCommediaModelEvaluation():

    # Load the tercets and split the lists in training and test lists
    
    testLabels = numpy.load('./data/commedia_labels.npy')
    classConditionalLikehoods=numpy.load('./data/commedia_ll.npy')
    predLabels = inferLabelsCC(classConditionalLikehoods)
    
    confM = confusionMatrix(testLabels,predLabels)
    print("confusion matrix")
    print(confM)

def binaryTaskEvaluation(binScores,pi1,CostFN,CostFP):
    t=-numpy.log(pi1*CostFN/((1-pi1)*CostFP))
    return numpy.where(binScores>t,1,0)


def DCF(ConfM,pi1,CostFN,CostFP):
    FNR=ConfM[0][1]/(ConfM[0][1]+ConfM[1][1])
    FPR=ConfM[1][0]/(ConfM[1][0]+ConfM[0][0])
    return pi1*CostFN*FNR+(1-pi1)*CostFP*FPR

def normalizedDCF(ConfM,pi1,CostFN,CostFP):
    return DCF(ConfM,pi1,CostFN,CostFP)/min(pi1*CostFN,(1.0-pi1)*CostFP)

def DCFrates (ConfM,pi1,CostFN,CostFP):
    FNR=ConfM[0][1]/(ConfM[0][1]+ConfM[1][1])
    FPR=ConfM[1][0]/(ConfM[1][0]+ConfM[0][0])
    return (pi1*CostFN*FNR+(1-pi1)*CostFP*FPR,FNR,FPR)

def normalizedDCFrates(ConfM,pi1,CostFN,CostFP):
    (DCF,FNR,FPR)=DCFrates(ConfM,pi1,CostFN,CostFP)
    return (DCF/min(pi1*CostFN,(1.0-pi1)*CostFP),FNR,FPR)

def minDCF(binScores,testLabels,pi1,CostFN,CostFP): #left just for comprehension of incrMinDCF
    minThreshold = -1
    minDCF = numpy.finfo(numpy.float64).max #modify

    for threshold in numpy.sort(numpy.copy(binScores)):
        predLabels = numpy.where(binScores>threshold,1,0)
        confM = confusionMatrix(testLabels,predLabels)
        normDCF = normalizedDCF(confM,pi1,CostFN,CostFP)
        if(normDCF<minDCF):
            minDCF=normDCF
            minThreshold=threshold
    
    return (minDCF,minThreshold) #doesn't return FNRv FPRv 


def incrMinDCF(binScores,testLabels,pi1,CostFN,CostFP):#incremental more efficient version minDCF

    minThreshold = -1
    minDCF = numpy.finfo(numpy.float64).max
    binScoreLabels = numpy.zeros((2,len(binScores)))
    for i in range(len(binScores)):
        binScoreLabels[:,i] = numpy.asarray([binScores[i], testLabels[i]])
    FNRv=numpy.zeros((len(binScores),))
    FPRv=numpy.zeros((len(binScores),))

    ind = numpy.argsort( binScoreLabels[0,:] ); 
    binScoreLabels = binScoreLabels[:,ind]
    
    predLabels = numpy.ones(len(binScores))
    predLabels[0] = 0.0
    confM = confusionMatrix(binScoreLabels[1,:],predLabels)

    for i in range(1,binScoreLabels.shape[1]):
        (normDCF,FNRv[i],FPRv[i]) = normalizedDCFrates(confM,pi1,CostFN,CostFP)
        
        if(normDCF<minDCF):
            minDCF=normDCF
            minThreshold=binScoreLabels[1][i]
        #update confusion matrix (change only 2 values)
        confM[0][int(binScoreLabels[1][i])] += 1 
        confM[1][int(binScoreLabels[1][i])] -= 1

    return (minDCF,minThreshold,FNRv,FPRv) #return FNRv  & FPRv

def plotROC(FNRv,FPRv):
    plt.title('Receiver Operating Characteristic')
    plt.plot(FPRv, 1-FNRv , 'b')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

def calcROC(binScores, testLabels, pi1, CostFN, CostFP):
    (_,_,FNRv,FPRv)=incrMinDCF(binScores,testLabels,pi1,CostFN,CostFP)
    plotROC(FNRv,FPRv)

def calcBayesError(binScores,testLabels):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    effPriors = numpy.exp(-numpy.logaddexp(0,-effPriorLogOdds))
    normDCFv = numpy.zeros((len(effPriors),))
    minDCFv = numpy.zeros((len(effPriors),))
    for i in range(len(effPriors)):
        ep=effPriors[i]
        predLabels = binaryTaskEvaluation(binScores,ep,1,1)
        confM = confusionMatrix(testLabels,predLabels)
        normDCFv[i] = normalizedDCF(confM,ep,1,1)
        (minDCFv[i],_,_,_)=incrMinDCF(binScores,testLabels,ep,1,1)

    plt.title('Bayes Error')
    plt.plot(effPriorLogOdds, normDCFv, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCFv, label='min DCF', color='b') 
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    
def tryBinaryTaskEvaluation():
    print("Divina commedia eps=0.001")
    #Divina commedia with eps = 0.001
    binScores = numpy.load('./data/commedia_llr_infpar.npy')
    testLabels = numpy.load('./data/commedia_labels_infpar.npy')
    #test 1
    predLabels = binaryTaskEvaluation(binScores,0.5,1,1)
    confM = confusionMatrix(testLabels,predLabels)
    print("confusion matrix")
    print(confM)
    cost = DCF(confM,0.5,1,1)
    print("cost: %f:" % cost)
    normCost = normalizedDCF(confM,0.5,1,1)
    print("normalized cost: %f" % normCost)
    #test 2
    predLabels = binaryTaskEvaluation(binScores,0.8,1,1)
    confM = confusionMatrix(testLabels,predLabels)
    print("confusion matrix")
    print(confM)
    cost = DCF(confM,0.8,1,1)
    print("cost: %f" % cost)
    normCost = normalizedDCF(confM,0.8,1,1)
    print("normalized cost: %f" % normCost)
    #test 3
    predLabels = binaryTaskEvaluation(binScores,0.5,10,1)
    confM = confusionMatrix(testLabels,predLabels)
    print("confusion matrix")
    print(confM)
    cost = DCF(confM,0.5,10,1)
    print("cost: %f" % cost)
    normCost = normalizedDCF(confM,0.5,10,1)
    print("normalized cost: %f" % normCost)
    #test4
    predLabels = binaryTaskEvaluation(binScores,0.8,1,10)
    confM = confusionMatrix(testLabels,predLabels)
    print("confusion matrix")
    print(confM)
    cost = DCF(confM,0.8,1,10)
    print("cost: %f" % cost)
    normCost = normalizedDCF(confM,0.8,1,10)
    print("normalized cost: %f" % normCost)

    #more efficient version of minDCF + plot of ROC
    #test1
    (minCost,_,_,_)=incrMinDCF(binScores,testLabels,0.5,1,1)
    print("min cost: %f" % minCost)
    
    #test2
    (minCost,_,_,_)=incrMinDCF(binScores,testLabels,0.8,1,1)
    print("min cost: %f" % minCost)
    
    #test3
    (minCost,_,_,_)=incrMinDCF(binScores,testLabels,0.5,10,1)
    print("min cost: %f" % minCost)
    
    #test4
    (minCost,_,_,_)=incrMinDCF(binScores,testLabels,0.8,1,10)
    print("min cost: %f" % minCost)
    
    #ROC curves (test1)
    calcROC(binScores,testLabels,0.5,1,1)
    
    plt.show()

    #Bayes Error plot of divina commedia with eps = 0.001
    calcBayesError(binScores,testLabels)

    print("Divina commedia eps=1.0")
    #Divina commedia with eps = 1
    binScores = numpy.load('./data/commedia_llr_infpar_eps1.npy')

    #test1
    predLabels = binaryTaskEvaluation(binScores,0.5,1,1)
    confM = confusionMatrix(testLabels,predLabels)
    normCost = normalizedDCF(confM,0.5,1,1)
    print("normalized cost: %f" % normCost)
    (minCost,_,_,_)=incrMinDCF(binScores,testLabels,0.5,1,1)
    print("min cost: %f" % minCost)
    
    #test2
    predLabels = binaryTaskEvaluation(binScores,0.8,1,1)
    confM = confusionMatrix(testLabels,predLabels)
    normCost = normalizedDCF(confM,0.8,1,1)
    print("normalized cost: %f" % normCost)
    (minCost,_,_,_)=incrMinDCF(binScores,testLabels,0.8,1,1)
    print("min cost: %f" % minCost)
    
    #test3
    predLabels = binaryTaskEvaluation(binScores,0.5,10,1)
    confM = confusionMatrix(testLabels,predLabels)
    normCost = normalizedDCF(confM,0.5,10,1)
    print("normalized cost: %f" % normCost)
    (minCost,_,_,_)=incrMinDCF(binScores,testLabels,0.5,10,1)
    print("min cost: %f" % minCost)
    
    #test4
    predLabels = binaryTaskEvaluation(binScores,0.8,1,10)
    confM = confusionMatrix(testLabels,predLabels)
    normCost = normalizedDCF(confM,0.8,1,10)
    print("normalized cost: %f" % normCost)
    (minCost,_,_,_)=incrMinDCF(binScores,testLabels,0.8,1,10)
    print("min cost: %f" % minCost)

    calcBayesError(binScores,testLabels)
    
    plt.show()

    


if __name__=='__main__':

    #irisModelEvaluation() #OK
    
    #divinaCommediaModelEvaluation() #OK
    
    tryBinaryTaskEvaluation()