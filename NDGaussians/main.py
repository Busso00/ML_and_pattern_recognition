import numpy
import matplotlib.pyplot as plt
import numpy.linalg

labelToN={'0':0}
nToLabel=['0']
attributeToN={'0':0}
nToAttribute=['0']
FILENAME="iris.csv"

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

def load_unnamed(filename,n_attr,n_label):#es load image pixel
    for i in range(n_label):
        labelToN['%d'%i]=i
        nToLabel[i]='%d'%i
    for i in range(n_attr):
        attributeToN['%d'%i]=i
        nToAttribute[i]='%d'%i
    load(filename)

def plot_hist(data,label,useUnnamed=False):
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
        
def GAU_pdf(x,mu,var):#1d 2021/2022
    GAU_y=1/numpy.sqrt(2*numpy.pi*var)*numpy.exp(-((x-mu)**2)/(2*var))
    return GAU_y

def GAU_logpdf(x,mu,var):#1d 2021/2022
    GAU_log_y=-1/2*numpy.log(2*numpy.pi)-1/2*numpy.log(var)-((x-mu)**2/(2*var))
    return GAU_log_y

def loglikehood(data,mu,var):#2021/2022
    ll= GAU_logpdf(data,mu,var).sum()
    print("log-likehood of the best fit of your data into a gaussian(mean=data.mean var=data.var):")
    print(ll)
    return ll

def plot_likehood_data1d(data):#2021/2022
    print("fitting your data into a gaussian...")
    plt.figure()
    plt.hist(data,bins=100,alpha=0.3,density=True)
    min=data.min()
    max=data.max()
    XPlot = numpy.linspace(min, max, 1000)
    plt.plot(XPlot,numpy.exp(GAU_logpdf(XPlot, data.mean(), data.var())))
    loglikehood(data,data.mean(),data.var())
    plt.show()

def GAU_ND_logpdf_chol(data,mu,C):#nd (2021/2022/2023) 
    #To avoid numerical issues due to exponentiation of large numbers, in many practical cases it’s more
    #convenient to work with the logarithm of the density
    #exponentiaion of xTC-1x given x high dimensional possibly gives high value
    Pdata=numpy.zeros((data.shape[1],))
    M=data.shape[0]
    (_,logdetC)=numpy.linalg.slogdet(C)#first return value is sign of logdet
    centered_data=data-mu
    
    Pdata+=-M/2*numpy.log(2*numpy.pi)-1/2*logdetC

    L=numpy.linalg.cholesky(C)#O(n3)-> can be inefficient with high number of features
    Y=numpy.linalg.inv(L)@centered_data
    mahlanobisDistance=(Y**2).sum(axis=0)#distance for every sample->axis=0

    Pdata-=1/2*mahlanobisDistance
    #for i in range(data.shape[1]):#ask for optimization insight
    #    Pdata[i]-=1/2*(centered_data[:,i].T@numpy.linalg.inv(C))@centered_data[:,i]
    return Pdata

def sol_GAU_ND_1sample(x,mu,C):
    xc=x-mu
    M=x.shape[0]
    const=-0.5*M*numpy.log(2*numpy.pi)
    logdet=numpy.linalg.slogdet(C)[1]
    L=numpy.linalg.inv(C)
    v=numpy.dot(xc.T,numpy.dot(L,xc)).ravel()

def GAU_ND_logpdf(X,mu,C):
    XC=X-mu
    M=X.shape[0]
    const=-0.5*M*numpy.log(2*numpy.pi)
    logdet=numpy.linalg.slogdet(C)[1]
    L=numpy.linalg.inv(C)
    v=(XC*numpy.dot(L,XC)).sum(axis=0)
    return const-0.5*logdet-0.5*v


def ML_parameter_ND_GAU(data):#2022/2023
    MLmu=vcol(data.mean(axis=1))
    MLC=numpy.zeros((data.shape[0],data.shape[0]))
    MLC+=numpy.cov(data,bias=True)
    return (MLmu,MLC)

def loglikelihood_ND(data,mu,C):#2022/2023
    ll=GAU_ND_logpdf(data,mu,C).sum()
    return ll

def test_gaussian():#test 2021/2022
    XGAU = numpy.load('solutions/XGau.npy')
    print('XGAU samples')
    print(XGAU)
    plot_hist(vrow(numpy.array(XGAU)),numpy.zeros(XGAU.size))#modified nbins
    #plot a gaussian
    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot, GAU_pdf(XPlot, 1.0, 2.0))
    plt.show()
    #compare gaussians
    pdfSol = numpy.load('solutions/CheckGAUPdf.npy')
    pdfGau = GAU_pdf(XPlot, 1.0, 2.0)
    print("mean distance between your function and gaussian:")
    print(numpy.abs(pdfSol - pdfGau).mean())
    #non logarithmic likehood gives too small result
    ll_samples = GAU_pdf(XGAU, 1.0, 2.0)
    likelihood = ll_samples.prod()
    print("example non logarithmic likehood between samples and a gaussian w mean 0 var 1:")
    print(likelihood)
    #logarithmic likehood gives appreciable result
    ll_samples = GAU_logpdf(XGAU, 1.0, 2.0)#il vettore di probabilità[i] di XGAU[i] assumendo che i campioni XGAU[i] si distribuiscano come gaussiana con media 1 e varianza 2
    likelihood = ll_samples.sum()
    print("example logarithmic likehood between samples and a gaussian w mean 0 var 1:")
    print(likelihood)
    #loglikehood between data and best fit gaussian
    plot_likehood_data1d(XGAU)

def test_gaussian_ND():#test 2021/2022

    XND = numpy.load('solutions/XND.npy')
    mu = numpy.load('solutions/muND.npy')
    C = numpy.load('solutions/CND.npy')
    pdfSol = numpy.load('solutions/llND.npy')
    pdfGau = GAU_ND_logpdf(XND, mu, C)
    print("error on 2021/2022 solution")
    print(numpy.abs(pdfSol - pdfGau).max())

def test_gaussian_ND2():#test 2022/2023
    
    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    m = numpy.ones((1,1)) * 1.0
    C = numpy.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), numpy.exp(GAU_ND_logpdf(vrow(XPlot), m, C)))
    plt.show()

    pdfSol = numpy.load('solutions/llGAU.npy')
    pdfGau = GAU_ND_logpdf(vrow(XPlot), m, C)
    print("1-d error")
    print(numpy.abs(pdfSol - pdfGau).max())

    XND = numpy.load('solutions/XND2.npy')
    mu = numpy.load('solutions/muND2.npy')
    C = numpy.load('solutions/CND2.npy')
    pdfSol = numpy.load('solutions/llND2.npy')
    pdfGau = GAU_ND_logpdf(XND, mu, C)
    print("error on 2022/2023 solution")
    print(numpy.abs(pdfSol - pdfGau).max())

    (MLmu,MLC)=ML_parameter_ND_GAU(XND)
    print("MLmu-ND")
    print(MLmu)
    print("MLC-ND")
    print(MLC)

    print("loglikelihood for ML estimated parameters")
    print(loglikelihood_ND(XND,MLmu,MLC))

    X1D=numpy.load('solutions/X1D.npy')
    (MLmu,MLC)=ML_parameter_ND_GAU(X1D)
    print("MLmu-1D")
    print(MLmu)
    print("MLC-1D")
    print(MLC)

    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), numpy.exp(GAU_ND_logpdf(vrow(XPlot), MLmu, MLC)))
    plt.show()

    ll = loglikelihood_ND(X1D, MLmu, MLC)
    print(ll)

if __name__=="__main__":
    test_gaussian()
    test_gaussian_ND()
    test_gaussian_ND2()
