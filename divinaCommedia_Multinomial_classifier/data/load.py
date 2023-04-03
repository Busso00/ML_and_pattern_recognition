import sys

def load_data():

    lInf = []

    if sys.version_info.major == 3: # Check if Python version is Python 3 or Python 2
        f=open('data/inferno.txt', encoding="ISO-8859-1")
    else:
        f=open('data/inferno.txt')

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    if sys.version_info.major == 3: # Check if Python version is Python 3 or Python 2
        f=open('data/purgatorio.txt', encoding="ISO-8859-1")
    else:
        f=open('data/purgatorio.txt')

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    if sys.version_info.major == 3: # Check if Python version is Python 3 or Python 2
        f=open('data/paradiso.txt', encoding="ISO-8859-1")
    else:
        f=open('data/paradiso.txt')
    for line in f:
        lPar.append(line.strip())
    f.close()
    
    return lInf, lPur, lPar

def split_data(l, n):

    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])
            
    return lTrain, lTest


