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
        
def GAU_pdf(x,mu,var):#1d
    GAU_y=1/numpy.sqrt(2*numpy.pi*var)*numpy.exp(-((x-mu)**2)/(2*var))
    return GAU_y

def GAU_logpdf(x,mu,var):#1d
    GAU_log_y=-1/2*numpy.log(2*numpy.pi)-1/2*numpy.log(var)-((x-mu)**2/(2*var))
    return GAU_log_y

def loglikehood(data,mu,var):
    ll= GAU_logpdf(data,mu,var).sum()
    print("log-likehood of the best fit of your data into a gaussian(mean=data.mean var=data.var):")
    print(ll)
    return ll

def plot_likehood_data1d(data):
    print("fitting your data into a gaussian...")
    plt.figure()
    plt.hist(data,bins=100,alpha=0.3,density=True)
    min=data.min()
    max=data.max()
    XPlot = numpy.linspace(min, max, 1000)
    plt.plot(XPlot,numpy.exp(GAU_logpdf(XPlot, data.mean(), data.var())))
    loglikehood(data,data.mean(),data.var())
    plt.show()

def GAU_ND_logpdf(data,mu,C):#nd
    GAU_ND_log_y=numpy.zeros((data.shape[1],))
    M=data.shape[0]
    SIGMA=numpy.linalg.det(C)
    centered_data=data-mu
    
    GAU_ND_log_y+=-M/2*numpy.log(2*numpy.pi)-1/2*numpy.log(SIGMA)
    for i in range(data.shape[1]):    
        GAU_ND_log_y[i]-=1/2*(centered_data[:,i].T@numpy.linalg.inv(C))@centered_data[:,i]

    return GAU_ND_log_y

def test_gaussian():
    XGAU = numpy.load('XGau.npy')
    print('XGAU samples')
    print(XGAU)
    plot_hist(vrow(numpy.array(XGAU)),numpy.zeros(XGAU.size))#modified nbins
    #plot a gaussian
    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot, GAU_pdf(XPlot, 1.0, 2.0))
    plt.show()
    #compare gaussians
    pdfSol = numpy.load('CheckGAUPdf.npy')
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

def test_gaussian_ND():
    XND = numpy.load('XND.npy')
    mu = numpy.load('muND.npy')
    C = numpy.load('CND.npy')
    pdfSol = numpy.load('llND.npy')
    
    pdfGau = GAU_ND_logpdf(XND,mu,C)#il vettore di probabilita[i] di XND[:,i] assumendo che i campioni XGAU[:,i] si distribuiscano come ND-gaussiana con medie mu e covarianze C
   
    print(numpy.abs(pdfSol - pdfGau).mean())

if __name__=="__main__":
    test_gaussian()
    test_gaussian_ND()
