#
#Progetto Biometric Identity Verification per il corso Machine Learning and Pattern Recognition di Federico Bussolino e Francine Ombala

#initial consideration: from corcoefficient matrix seen high correlation bw attr:
#lot of attributes are higly correlated
#  

import numpy
import matplotlib.pyplot as plt
import numpy.linalg
import scipy.linalg

attributeToN={'attr1':0,'attr2':1,'attr3':2, 'attr4':3, 'attr5':4, 'attr6':5, 'attr7':6, 'attr8':7, 'attr9':8, 'attr10':9}
nToAttribute=['attr1','attr2','attr3','attr4','attr5','attr6','attr7','attr8','attr9','attr10']
labelToN={'0':0,'1':1}
nToLabel=['0','1']
FILENAME="Train.txt"
NO_HIST=False
NO_SCATTER=False

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
        print("error opening Dataset")
        exit(-1)
    
    labeledData=DataList()
    for line in f:
        try:
            record=line.split(',')
            record[-1]=record[-1].replace('\n', '')
            attributes=numpy.array([float(i) for i in record[0:-1]])
            label=labelToN[record[-1]]
            labeledData.dsAttributes.append(attributes)
            labeledData.dsLabel.append(label)
        except:
            print("error parsing line")

    labeledData=DataArray(labeledData.dsAttributes,labeledData.dsLabel)
    return labeledData

def plot_hist(data,label,useUnnamed=False):
    if NO_HIST:
        return
    print("Displaying histogram by attributes (distinct color for distinct label...)")
    for i in range(data.shape[0]):
        plt.figure()
        if (useUnnamed):
            plt.xlabel("attribute%d"%i)
        else:
            plt.xlabel(nToAttribute[i])
        for j in range(len(nToLabel)):
            w=numpy.ones(data[:,label==j][0,:].size)*(1/data[:,label==j][0,:].size) #pesare per fare sì che somma h = 1 ossia percentuali
            plt.hist(data[:,label==j][i,:],label=nToLabel[j],alpha = 0.3,bins=50,density=True) #alpha = trasparenza, bins=numero divisioni, density=true->normalizza t.c sum(numero_valori_bin*ampiezza_bin)=1  ->scala altezza e mostra circa una gaussiana
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        #plt.savefig('hist_%d_%s.pdf' % (i,nToAttribute[i]))

    plt.show()

def plot_scatter(data,label,useUnnamed=False):
    if NO_SCATTER:
        return
    print("Displaying 2d scatter plot by attributes (distinct color for distinct label...)")
    for i in range(data.shape[0]):
        for j in range(i):#doesn't print the same figure with inverted axis
            plt.figure()
            if (useUnnamed):
                plt.xlabel("attribute%d"%i)
                plt.ylabel("attribute%d"%j)
            else:
                plt.xlabel(nToAttribute[i])
                plt.ylabel(nToAttribute[j])
            for k in range(len(nToLabel)):
                plt.scatter(data[:,label==k][i,:],data[:,label==k][j,:],label=nToLabel[k],alpha=0.3)
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            #plt.savefig('scatter_%s_%s.pdf' % (nToAttribute[i],nToAttribute[j]))

    plt.show()

def meanAttr(data):
    meanV=vcol(data.mean(1))
    
    print("Mean:")
    for i in range(meanV.shape[0]):
        print("%s:\t%.2f" % (nToAttribute[i],meanV[i,0]))
    print("\n")

    return meanV

def stdDevAttr(data):
    stdDevV=vcol(data.std(1))

    print("Standard Deviation:")
    for i in range(stdDevV.shape[0]):
        print("%s:\t%.2f" % (nToAttribute[i],stdDevV[i,0]))
    print("\n")

    return stdDevV

def corrM(data):
    pearsonM=numpy.corrcoef(data)

    print("Correlation coefficient (Pearson):")
    for i in range(data.shape[0]+1):
        for j in range(data.shape[0]+1):
            if  (i==0):
                if(j==0):
                    print("\t\t",end="")
                else:
                    print(nToAttribute[j-1]+"\t",end="")
            else:
                if(j==0):
                    print(nToAttribute[i-1]+"\t",end="")
                else:
                    print("%.2f\t\t"%(pearsonM[i-1][j-1]),end="")

        print("")
    print("\n")

    return pearsonM
        
def covM(data):
    covM=numpy.cov(data,bias=True)

    print("Covariance:")
    for i in range(data.shape[0]+1):
        for j in range(data.shape[0]+1):
            if  (i==0):
                if(j==0):
                    print("\t\t",end="")
                else:
                    print(nToAttribute[j-1]+"\t",end="")
            else:
                if(j==0):
                    print(nToAttribute[i-1]+"\t",end="")
                else:
                    print("%.2f\t\t"%(covM[i-1][j-1]),end="")

        print("")
    print("\n")

    return covM

def PCA_solution(D,m):
    mu=vcol(D.mean(1))
    C=numpy.dot(D-mu,(D-mu).T)/D.shape[1]
    s,U=numpy.linalg.eigh(C)
    U=U[:,::-1]
    P=U[:,0,m]
    return P

def PCA_solution_svd(D,m):
    mu=vcol(D.mean(1))
    C=numpy.dot(D-mu,(D-mu).T)/D.shape[1]
    U,_,_=numpy.linalg.svd(C)
    P=U[:,0:m]
    return P

def PCA(data,m_req_dim):#unsupervised
    covarianceM=numpy.cov(data,bias=True)#already center the data, but normalize by n-1
    _,eigenvectors=numpy.linalg.eigh(covarianceM)#eigenvalues aren't necessary
    P = eigenvectors[:, ::-1][:,0:m_req_dim] 
    projectedData=P.T@data
    print("Transform matrix:")
    print(P)
    return (projectedData,P)

def PCA_treshold(data,treshold):#unsupervised
    covarianceM=numpy.cov(data,bias=True)#already center the data, but normalize by n-1
    eigenvalues,eigenvectors=numpy.linalg.eigh(covarianceM)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    required_dim=1
    explained_var=eigenvalues[0]
    tot_explained_var=eigenvalues.sum()
    while (explained_var/tot_explained_var)<treshold:
        explained_var+=eigenvalues[required_dim]
        required_dim+=1

    print("Selected dimensions : %d"%required_dim)
    P=eigenvectors[:,0:required_dim]
    projectedData=P.T@data
    print("Transform matrix:")
    print(P)
    return (projectedData,P)

def PCA_svd(data,m_req_dim):#unsupervised
    covarianceM=numpy.cov(data,bias=True)#already center the data, but normalize by n-1
    U,_,_=numpy.linalg.svd(covarianceM)#singularvalues aren't necessary
    P = U[:,0:m_req_dim]
    projectedData=P.T@data
    print("Transform matrix:")
    print(P)
    return (projectedData,P)

def PCA_treshold_svd(data,treshold):#unsupervised
    covarianceM=numpy.cov(data,bias=True)#already center the data, but normalize by n-1
    U,singularvalues,_=numpy.linalg.svd(covarianceM)

    required_dim=1
    explained_var=singularvalues[0]
    tot_explained_var=singularvalues.sum()
    while (explained_var/tot_explained_var)<treshold:
        explained_var+=singularvalues[required_dim]
        required_dim+=1

    print("Selected dimensions : %d"%required_dim)
    P=U[:,0:required_dim]
    projectedData=P.T@data
    print("Transform matrix:")
    print(P)
    return (projectedData,P)

def within_class_covariance_M(data,label):
    N=label.shape[0]
    Sw=numpy.zeros((data.shape[0],data.shape[0]))
    for c in range(len(nToLabel)):
        elementOfC=data[:,label==c]
        nc=elementOfC.shape[1]
        Sw+=(numpy.cov(elementOfC,bias=True)*nc)/N
    return Sw

def between_class_covariance_M(data,label):
    N=label.shape[0]
    avg=vcol(data.mean(axis=1))
    Sb=numpy.zeros((data.shape[0],data.shape[0]))
    for c in range(len(nToLabel)):
        elementOfC=data[:,label==c]
        avgOfC=vcol(elementOfC.mean(axis=1))
        nc=elementOfC.shape[1]
        Sb+=(((avgOfC-avg)@(avgOfC-avg).T)*nc)/N
    return Sb

def LDA(data,label):#supervised
    m=len(nToLabel)-1#other directions are random
    Sw=within_class_covariance_M(data,label)
    Sb=between_class_covariance_M(data,label)
    _,U=scipy.linalg.eigh(Sb,Sw)#solve the generalized eigenvalue problem
    #eigenvalue Sw^-1@Sb , only C-1 eigenvectors are !=0 -> useless random eigenvectors
    W=U[:,::-1][:,0:m]
    projectedData=W.T@data

    print("Transform matrix:")
    print(W)
    return (projectedData,W)

def LDA_2proj(data,label):#supervised
    m=len(nToLabel)-1#other directions are random
    Sw=within_class_covariance_M(data,label)
    Sb=between_class_covariance_M(data,label)
    
    U,eigv1,_=numpy.linalg.svd(Sw)
    P1=U@numpy.diag(1.0/(eigv1**0.5))@U.T
    SBT=P1@Sb@P1.T#transformed between class covariance
    P2,eigv,_=numpy.linalg.svd(SBT)
    #eigenvalue Sw^-1@Sb , only C-1 eigenvectors are !=0 -> useless random eigenvectors
    W=P1.T@P2[:,0:m]
    projectedData=W.T@data

    print("Transform matrix:")
    print(W)
    return (projectedData,W)

def SbSw(D,L):
    Sb=0
    Sw=0
    mu=vcol(D.mean(1))
    for i in range(len(nToLabel)):
        DCIs=D[:,L==i]
        muCIs=vcol(DCIs.mean())
        Sw+=numpy.dot(DCIs-muCIs,(DCIs-muCIs).T)
        Sb+=DCIs.shape[1]*numpy.dot(muCIs-mu,(muCIs-mu).T)
    Sw/=D.shape[1]
    Sb/=D.shape[1]
    return (Sb,Sw)

def LDA_solution1(D,L,m):
    Sb,Sw=SbSw(D,L)
    s,U=scipy.linalg.eigh(Sb,Sw)
    return U[:,::-1][:,0:m]

def LDA_solution2(D,L,m):
    Sb,Sw=SbSw(D,L)
    U,s,_=numpy.linalg.svd(Sw)#s = 1d vector ->1/s 1d vector -> vcol column vector
    P1=numpy.dot(U,vcol(1.0/s**0.5)*U.T)#may not multiply by U
    SBTilde=numpy.dot(Sb,P1.T)
    U,_,_=numpy.linalg.svd(SBTilde)
    P2=U[:,0:m]
    return numpy.dot(P1.T,P2)
    #exploiting broadcasting
    #  s1  s1 ... s1  *  u11 u12 ... u1n  =  s1*u11 s1*u12 ... s1*u1n 
    #  s2  s2     s2     u21 u22 ... u2n     s2*u21 s2*u22 ... s2*u2n
    #  :   :      :      :   :       :       :      :          :
    #  sn  sn ... sn     un1 un2 ... unn     sn*un1 sn*un2 ... sn*unn
    #not exploit broadcasting
    #  s1 0  ... 0    @  u11 u12 ... u1n  =  s1*u11 s1*u12 ... s1*u1n 
    #  0  s2 ... 0       u21 u22 ... u2n     s2*u21 s2*u22 ... s2*u2n
    #  :  :      :       :   :       :       :      :          :
    #  0  0 ... sn       un1 un2 ... unn     sn*un1 sn*un2 ... sn*unn
    #->s*U=(s*I)@U

def visualizeData(labeledData):
    print("Attributes matrix:")
    print(labeledData.dsAttributes)
    print("Label vector:")
    print(labeledData.dsLabel)
    plot_hist(labeledData.dsAttributes,labeledData.dsLabel)
    plot_scatter(labeledData.dsAttributes,labeledData.dsLabel)
    meanAttr(labeledData.dsAttributes)#mean along rows
    stdDevAttr(labeledData.dsAttributes)
    covM(labeledData.dsAttributes)
    corrM(labeledData.dsAttributes)

def testPCA(labeledData):
    #CompareM=numpy.load("solutions/IRIS_PCA_matrix_m4.npy")
    #print("Compare matrix:")
    #print(CompareM)
    testDir=10
    
    #projectedData,P_not_T=PCA(labeledData.dsAttributes,testDir)
    #plot_hist(projectedData,labeledData.dsLabel,useUnnamed=True)
    #plot_scatter(projectedData,labeledData.dsLabel,useUnnamed=True)
    #since the basis is orthonormal if P_not_T=CompareM <->PT@CompareM=I 
    #print("check if PCA is valid")
    #print((P_not_T.T-CompareM))

    projectedData,P_not_T=PCA_treshold(labeledData.dsAttributes,0.95)
    plot_hist(projectedData,labeledData.dsLabel,useUnnamed=True)
    plot_scatter(projectedData,labeledData.dsLabel,useUnnamed=True)
    #since the basis is orthonormal if P_not_T=CompareM <->PT@CompareM=I 
    #print("check if PCA is valid")
    #print(P_not_T.T@CompareM)

    #projectedData,P_not_T=PCA_svd(labeledData.dsAttributes,testDir)
    #plot_hist(projectedData,labeledData.dsLabel,useUnnamed=True)
    #plot_scatter(projectedData,labeledData.dsLabel,useUnnamed=True)
    #since the basis is orthonormal if P_not_T=CompareM <->PT@CompareM=I 
    #print("check if PCA is valid")
    #print(P_not_T.T@CompareM)
   
    #projectedData,P_not_T=PCA_treshold_svd(labeledData.dsAttributes,1)
    #plot_hist(projectedData,labeledData.dsLabel,useUnnamed=True)
    #plot_scatter(projectedData,labeledData.dsLabel,useUnnamed=True)
    #since the basis is orthonormal if P_not_T=CompareM <->PT@CompareM=I 
    #print("check if PCA is valid")
    #print(P_not_T.T@CompareM)

def testLDA(labeledData):
    #CompareM=numpy.load("solutions/IRIS_LDA_matrix_m2.npy")
    #print("Compare matrix:")
    #print(CompareM)

    projectedData,P_not_T=LDA(labeledData.dsAttributes,labeledData.dsLabel)
    plot_hist(projectedData,labeledData.dsLabel,useUnnamed=True)
    plot_scatter(projectedData,labeledData.dsLabel,useUnnamed=True)
    #print("check if LDA has %s non zero singular values"%(len(nToLabel)-1))
    #print(numpy.linalg.svd(numpy.hstack([CompareM, P_not_T]))[1])
    
    #projectedData,P_not_T=LDA_2proj(labeledData.dsAttributes,labeledData.dsLabel)
    #plot_hist(projectedData,labeledData.dsLabel,useUnnamed=True)
    #plot_scatter(projectedData,labeledData.dsLabel,useUnnamed=True)
    #print("check if LDA has %s non zero singular values"%(len(nToLabel)-1))
    #print(numpy.linalg.svd(numpy.hstack([CompareM, P_not_T]))[1])

    #P_not_T[:,i]=wi i-th vector of the basis
    #P_not_T require to have columns that are lc of columns of CompareM
    #hence we require that singular values !=0 are C-1
    return projectedData,P_not_T
    
    

if __name__ == '__main__':
    # Visualization of data

    labeledData=load(FILENAME)
    #visualizeData(labeledData)
    projectedData=testPCA(labeledData)#not invarint to linear transformation
    
    #testLDA(projectedData)#invarint to linear transformation

    pass