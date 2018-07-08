# Example of kNN implemented from Scratch in Python
# Retirado de https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# Acesso em 06/07/2018 10:24
# Alterações realizadas para funcionamento no python 3
import csv
import random
import math
import operator

# Os dados estão no formato CSV sem uma linha de cabeçalho. E não possuem qualquer tipo de identificação
# Podemos abrir o arquivo com a função aberta e ler as linhas de dados usando a função de leitura no módulo csv .

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            # Nesse trecho de código é dividido os conjuntos de dados de treinamento que o kNN pode usar para fazer previsões
            # E um conjunto de dados de teste que podemos usar para avaliar a precisão do modelo.
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

#Para fazer previsões, é preciso calcular a similaridade entre quaisquer duas instâncias de dados.
# Isso é necessário para que possamos localizar as k instâncias de dados mais semelhantes no conjunto de dados de treinamento
#  para um determinado membro do conjunto de dados de teste e, por sua vez, fazer uma previsão

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

# Função que retorna k vizinhos mais similares do conjunto de treinamento para uma determinada instância de teste.
# utilizada para coletar as instâncias mais semelhantes para uma determinada instância invisível utilizando a similaridade.
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Depois de localizar os vizinhos mais semelhantes para uma instância de teste,
# A próxima tarefa é criar uma resposta prevista com base nesses vizinhos.
# Podemos fazer isso permitindo que cada vizinho vote em seu atributo de classe e tome a maioria como a previsão.
# Abaixo, há uma função para obter a maioria dos votos de resposta de um número de vizinhos.
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# O calculo da precisão das previsões é feito com a razão entre o total de predições corretas e todas as previsões feitas, chamada de precisão de classificação.
# Abaixo está a função que soma o total de previsões corretas e retorna a precisão como uma porcentagem das classificações corretas.
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def main():
    # Preparando os dados
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('iris.data', split, trainingSet, testSet)
    print("Train set: " + repr(len(trainingSet)))
    print("Test set: " + repr(len(testSet)))
    # Gerando Predições
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()