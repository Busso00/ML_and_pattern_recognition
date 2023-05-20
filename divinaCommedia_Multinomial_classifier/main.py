import data.load as dl
import numpy

def vcol(v):
    return v.reshape((v.size,1))

def vrow(v):
    return v.reshape((1,v.size))

wordDictionary={}
def buildDictCount(D):
    for doc in D:
        for word in doc.split(' '):
            if word not in wordDictionary:
                wordDictionary[word]=len(wordDictionary)

def wordProbability(D,count):
    wordOccurrencies=numpy.zeros(count)
    NWordC=0
    for doc in D:
        for word in doc.split(' '):
            wordOccurrencies[wordDictionary[word]]+=1
            NWordC+=1
    #eps hyperparameter
    return vcol((wordOccurrencies+0.001)/NWordC)

labelToN={'Inf':0,'Pur':1,'Par':2}
nToLabel=['Inf','Pur','Par']
PC=[1/3,1/3,1/3]

def inferClass(D,P,nLabel):
    pred=numpy.zeros(len(D),dtype=numpy.int32)
    j=0
    for doc in D:
        Vword=doc.split(' ')
        ll=numpy.zeros(nLabel)
        for c in range(nLabel):
            ll[c]+=numpy.log(PC[c])
            for word in Vword:
                if word in wordDictionary:
                    ll[c]+=numpy.log(P[wordDictionary[word],c])
        pred[j]=ll.argmax()
        j+=1
    return pred



def evalAccuracy(pred,actual):
    v=pred==actual
    acc=v.sum()/actual.shape[0]
    print(acc)
    return acc

if __name__ == '__main__':

    # Load the tercets and split the lists in training and test lists
    
    lInf, lPur, lPar = dl.load_data()

    lInf_train, lInf_evaluation = dl.split_data(lInf, 4)
    lPur_train, lPur_evaluation = dl.split_data(lPur, 4)
    lPar_train, lPar_evaluation = dl.split_data(lPar, 4)
    
    #method 2
    print("discriminate between inferno, purgatorio and paradiso")
    buildDictCount(lInf_train)
    buildDictCount(lPur_train)
    buildDictCount(lPar_train)

    PInf=wordProbability(lInf_train,len(wordDictionary))
    PPur=wordProbability(lPur_train,len(wordDictionary))
    PPar=wordProbability(lPar_train,len(wordDictionary))
    P=numpy.hstack([PInf,PPur,PPar])

    pred=inferClass(lInf_evaluation,P,3)
    print("accuracy inferno")
    accInf=evalAccuracy(pred,0*numpy.ones(pred.shape[0],dtype=numpy.int32))
    pred=inferClass(lPur_evaluation,P,3)
    print("accuracy purgatorio")
    accPur=evalAccuracy(pred,1*numpy.ones(pred.shape[0],dtype=numpy.int32))
    pred=inferClass(lPar_evaluation,P,3)
    print("accuracy paradiso")
    accPar=evalAccuracy(pred,2*numpy.ones(pred.shape[0],dtype=numpy.int32))
    print("overall accuracy")
    print((accInf*len(lInf_evaluation)+accPur*len(lPur_evaluation)+accPar*len(lPar_evaluation))/(len(lInf_evaluation)+len(lPur_evaluation)+len(lPar_evaluation)))

    
    wordDictionary={}

    print("discriminate between inferno and purgatorio")
    buildDictCount(lInf_train)
    buildDictCount(lPur_train)

    PInf=wordProbability(lInf_train,len(wordDictionary))
    PPur=wordProbability(lPur_train,len(wordDictionary))
    P=numpy.hstack([PInf,PPur])

    pred=inferClass(lInf_evaluation,P,2)
    print("accuracy inferno")
    accInf=evalAccuracy(pred,0*numpy.ones(pred.shape[0],dtype=numpy.int32))
    pred=inferClass(lPur_evaluation,P,2)
    print("accuracy purgatorio")
    accPur=evalAccuracy(pred,1*numpy.ones(pred.shape[0],dtype=numpy.int32))
    print("overall accuracy")
    print((accInf*len(lInf_evaluation)+accPur*len(lPur_evaluation))/(len(lInf_evaluation)+len(lPur_evaluation)))


    wordDictionary={}

    print("discriminate between purgatorio and paradiso")
    buildDictCount(lPur_train)
    buildDictCount(lPar_train)

    PPur=wordProbability(lPur_train,len(wordDictionary))
    PPar=wordProbability(lPar_train,len(wordDictionary))
    P=numpy.hstack([PPur,PPar])

    pred=inferClass(lPur_evaluation,P,2)
    print("accuracy purgatorio")
    accPur=evalAccuracy(pred,0*numpy.ones(pred.shape[0],dtype=numpy.int32))
    pred=inferClass(lPar_evaluation,P,2)
    print("accuracy paradiso")
    accPar=evalAccuracy(pred,1*numpy.ones(pred.shape[0],dtype=numpy.int32))
    print("overall accuracy")
    print((accPur*len(lPur_evaluation)+(accPar*len(lPar_evaluation)))/(len(lPur_evaluation)+len(lPar_evaluation)))

    wordDictionary={}

    print("discriminate between inferno and paradiso")
    buildDictCount(lInf_train)
    buildDictCount(lPar_train)

    PInf=wordProbability(lInf_train,len(wordDictionary))
    PPar=wordProbability(lPar_train,len(wordDictionary))
    P=numpy.hstack([PInf,PPar])

    pred=inferClass(lInf_evaluation,P,2)
    print("accuracy inferno")
    accInf=evalAccuracy(pred,0*numpy.ones(pred.shape[0],dtype=numpy.int32))
    pred=inferClass(lPar_evaluation,P,2)
    print("accuracy paradiso")
    accPar=evalAccuracy(pred,1*numpy.ones(pred.shape[0],dtype=numpy.int32))
    print("overall accuracy")
    print((accInf*len(lInf_evaluation)+(accPar*len(lPar_evaluation)))/(len(lInf_evaluation)+len(lPar_evaluation)))
