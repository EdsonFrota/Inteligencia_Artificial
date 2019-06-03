
# coding: utf-8

# In[53]:


#Aluno: Edson Frota  - Matricula: 201515412
#Curso: Engenharia de Computação - UFG
# Perceptron

import numpy as np
import matplotlib.pyplot as plt
import random as rand

def main():
    X = [[1,1],[1,0],[0,1],[0,0]]
    d = [1,-1,-1,-1]
    w, bias = Perceptron(Epocas = 100, E = 1, Alpha = .1, X=X, d=d)
    plotar(w[0],w[1],bias,"Porta lógica AND com Perceptron")
    
def Funcao_Ativação(U):
    if(U > 0):
        return (1)
    else:
        return(-1)

def Potencial_Ativação(Entradas, Peso, Bias):
    aux = 0
    for a in range(len(Entradas)):
        aux = aux + Entradas[a] * Peso[a]
    return (aux + Bias)

def Soma_Erro(Erros):
    S = 0
    for erro in Erros:
            if erro !=0:
                S += 1
    return (S)
    
def Perceptron(Epocas, E, Alpha, X, d):
    w = [rand.random()for i in range(2)]
    b = rand.random()
    t = 1

    while t < Epocas and E > 0:
        edson = []
        for a in range(len(X)):
            Saida = Funcao_Ativação(Potencial_Ativação(X[a], w, b))
            edson.append(d[a]- Saida)
            
            for flag in range(len(w)):
                w[flag] = w[flag] + edson[a] * Alpha * X[a][flag]
            b += Alpha * edson[a]
        E = Soma_Erro(edson)
        t += 1
    return(w,b)

def plotar(w1,w2,bias,title):
    xvals = np.arange(-1, 3, 0.01)     
    newyvals = (((xvals * w2) * - 1) - bias) / w1
    plt.plot(xvals, newyvals, 'r-')    
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.axis([-1,2,-1,2])
    plt.plot([0,1,0],[0,0,1], 'b^')
    plt.plot([1],[1], 'go')
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.show()

if __name__ == '__main__':
    main()

