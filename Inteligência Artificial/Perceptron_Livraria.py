
# coding: utf-8

# In[9]:


"""
@author: 
    
    Universidade Federal de Goiás
    Engenharia de Computação
    
    Aluno: Edson Frota
    
    	Implementação da rede neural Perceptron
	w = w + N * (d(k) - y) * x(k)
"""
import copy

class Perceptron:
    def __init__(self, amostras, saidas, taxa_aprendizado = 1, epocas = 1000):
        self.amostras = amostras # todas as amostras
        self.saidas = saidas # saídas respectivas de cada amostra
        self.taxa_aprendizado = taxa_aprendizado # taxa de aprendizado (entre 0 e 1)
        self.epocas = epocas # número de épocas
        self.num_amostras = len(amostras) # quantidade de amostras
        self.num_amostra = len(amostras[0]) # quantidade de elementos por amostra
        self.pesos = copy.deepcopy(saidas) # vetor de pesos, copia a saida por causa do tamanho do vetor
        
        self.num_saidas = len(saidas[0])
        
        # função para treinar a rede
    def treinar(self):
        
        # adiciona 1 para cada uma das amostras
        for amostra in self.amostras:
            amostra.insert(0, 1)
            
        # inicia o vetor de pesos com valores iguais a 1
        for i in range(self.num_amostras):
            for j in range(self.num_saidas):
                self.pesos[i][j] = 1
                
        # inicia o contador de epocas
        num_epocas = 0
        while True:           
            erro = False # o erro inicialmente inexiste
            # para todas as amostras de treinamento
            for i in range(self.num_amostras):
                for s in range(self.num_saidas):
                    u = 0
                    for j in range(self.num_amostra):
                        u += self.pesos[j][s] * self.amostras[i][j]
                        # obtém a saída da rede utilizando a função de ativação
                    y = self.sinal(u)
                    # verifica se a saída da rede é diferente da saída desejado
                    if y != self.saidas[i][s]:
                        # calcula o erro: subtração entre a saída desejada e a saída da rede
                        erro_aux = self.saidas[i][s] - y
                        # faz o ajuste dos pesos para cada elemento da amostra
                        for j in range(self.num_amostra):
                            self.pesos[j][s] = self.pesos[j][s] + self.taxa_aprendizado * erro_aux * self.amostras[i][j]
                        erro = True # ainda existe erro
                        # incrementa o número de épocas
                        num_epocas += 1
                        
# critério de parada é pelo número de épocas ou se não existir erro
            if num_epocas > self.epocas or not erro:
               break
        print("\nPESOS DEFINIDOS APÓS A FASE DE TREINAMENTO:")
        print(self.pesos[:self.num_amostra+1])
        
    def testar(self):
        teste = copy.deepcopy(self.amostras)
        
# utiliza o vetor de pesos que foi ajustado na fase de treinamento
        u = copy.deepcopy(self.saidas)
        for i in range(self.num_amostras):
            for j in range(self.num_saidas):
                u[i][j] = 0
        y = copy.deepcopy(self.saidas)
        pesos = copy.deepcopy(self.pesos[:self.num_amostra+1])
        for s in range(self.num_saidas):
            for i in range(self.num_amostras):
                for j in range(self.num_amostra):
                    u[j][s] +=  teste[i][j] * pesos[j][s] 
        y[j][s] = self.sinal(u[j][s])
        
        print(y)
        
	# função de ativação: degrau bipolar (sinal)
    def sinal(self, u):
        return 1 if u >= 0 else 0
    
# amostras: um total de 8 amostras
amostras = [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]]

# saídas desejadas de cada amostra
saidas = [[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 1]]

# conjunto de amostras de testes
testes = copy.deepcopy(amostras)

# cria uma rede Perceptron
rede = Perceptron(amostras = amostras, saidas = saidas, taxa_aprendizado = 1, epocas = 1000)

print("\nAMOSTRAS:")
print(amostras)

print("\nSAÍDAS ESPERADAS:")
print(saidas)

# treina a rede
rede.treinar()

print("\nSAÍDAS GERADAS :")
rede.testar()

