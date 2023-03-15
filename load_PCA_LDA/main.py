import numpy
import matplotlib.pyplot as plt
import numpy.linalg
import scipy.linalg

labelToN={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
nToLabel=['Iris-setosa','Iris-versicolor','Iris-virginica']
attributeToN={'Sepal-length':0,'Sepal-width':1,'Petal-length':2,'Petal-width':3}
nToAttribute=['Sepal-length','Sepal-width','Petal-length','Petal-width']
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

def plot_hist(data,label,useUnnamed=False):
    print("Displaying histogram by attributes (distinct color for distinct label...)")
    for i in range(data.shape[0]):
        plt.figure()
        if (useUnnamed):
            plt.xlabel("attribute%d"%i)
        else:
            plt.xlabel(nToAttribute[i])
        for j in range(len(nToLabel)):
            w=numpy.ones(data[:,label==j][0,:].size)*(1/data[:,label==j][0,:].size) #pesare per fare sì che somma h = 1 ossia percentuali
            plt.hist(data[:,label==j][i,:],label=nToLabel[j],alpha = 0.3,bins=10,density=True) #alpha = trasparenza, bins=numero divisioni, density=true->normalizza t.c sum(numero_valori_bin*ampiezza_bin)=1  ->scala altezza e mostra circa una gaussiana
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        #plt.savefig('hist_%d_%s.pdf' % (i,nToAttribute[i]))

    plt.show()

def plot_scatter(data,label,useUnnamed=False):
    print("Displaying 2d scatter plot by attributes (distinct color for distinct label...)")
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if (i != j):
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
    meanV=data.mean(1).reshape((data.shape[0],1))
    
    print("Mean:")
    for i in range(meanV.shape[0]):
        print("%s:\t%.2f" % (nToAttribute[i],meanV[i,0]))
    print("\n")

    return meanV

def stdDevAttr(data):
    stdDevV=data.std(1).reshape((data.shape[0],1))

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

def PCA(data,m_req_dim):#unsupervised
    covarianceM=numpy.cov(data,bias=True)#already center the data, but normalize by n-1
    eigenvalues,eigenvectors=numpy.linalg.eigh(covarianceM)
    eigenvalues = eigenvalues[ ::-1]
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
    avg=data.mean(axis=1).reshape(data.shape[0],1)
    Sb=numpy.zeros((data.shape[0],data.shape[0]))
    for c in range(len(nToLabel)):
        elementOfC=data[:,label==c]
        avgOfC=elementOfC.mean(axis=1).reshape(data.shape[0],1)
        nc=elementOfC.shape[1]
        Sb+=(((avgOfC-avg)@(avgOfC-avg).T)*nc)/N
    return Sb

def LDA(data,label):#supervised
    m=2
    Sw=within_class_covariance_M(data,label)
    Sb=between_class_covariance_M(data,label)
    s,U=scipy.linalg.eigh(Sb,Sw)#solve the generalized eigenvalue problem
    W=U[:,::-1][:,0:m]
    projectedData=W.T@data
    print("Transform matrix:")
    print(W)
    return (projectedData,W)

def LDA_2proj(data,label):#supervised
    m=2
    avgClass=[]
    Sw=within_class_covariance_M(data,label)
    Sb=between_class_covariance_M(data,label)
    
    U,eigv1,_=numpy.linalg.svd(Sw)
    P1=(U@numpy.diag(1.0/(eigv1**0.5)))@U.T
    SBT=(P1@Sb)@P1.T#transformed between class covariance
    eigv2,P2=numpy.linalg.eig(SBT)
    W=P1.T@P2[:,0:m]
    projectedData=W.T@data

    print("Transform matrix:")
    print(W)
    return (projectedData,W)

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
    CompareM=numpy.load("IRIS_PCA_matrix_m4.npy")
    print("Compare matrix:")
    print(CompareM)
    
    projectedData,P_T=PCA(labeledData.dsAttributes,2)
    plot_hist(projectedData,labeledData.dsLabel,useUnnamed=True)
    plot_scatter(projectedData,labeledData.dsLabel,useUnnamed=True)

    projectedData,P_T=PCA_treshold(labeledData.dsAttributes,0.97)
    plot_hist(projectedData,labeledData.dsLabel,useUnnamed=True)
    plot_scatter(projectedData,labeledData.dsLabel,useUnnamed=True)

def testLDA(labeledData):
    CompareM=numpy.load("IRIS_LDA_matrix_m2.npy")
    print("Compare matrix:")
    print(CompareM)

    projectedData,P_T=LDA(labeledData.dsAttributes,labeledData.dsLabel)
    plot_hist(projectedData,labeledData.dsLabel,useUnnamed=True)
    plot_scatter(projectedData,labeledData.dsLabel,useUnnamed=True)

    projectedData,P_T=LDA_2proj(labeledData.dsAttributes,labeledData.dsLabel)
    plot_hist(projectedData,labeledData.dsLabel,useUnnamed=True)
    plot_scatter(projectedData,labeledData.dsLabel,useUnnamed=True)

if __name__=="__main__":
    
    labeledData=load(FILENAME)
    visualizeData(labeledData)
    testPCA(labeledData)
    testLDA(labeledData)