import numpy as np
import random
import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import copy

class JogoDaVelha:
    def __init__(self):
        self.tabuleiro = np.zeros(9, dtype=int) 
        self.terminado = False

    def reiniciar(self):
        self.tabuleiro = np.zeros(9, dtype=int)
        self.terminado = False
        return tuple(self.tabuleiro)

    def jogada(self, acao, jogador):
        if self.tabuleiro[acao] != 0:
            return tuple(self.tabuleiro), -10, self.terminado  
        
        self.tabuleiro[acao] = jogador

        if self.verificar_vencedor(jogador):
            self.terminado = True
            return tuple(self.tabuleiro), 1 if jogador == 1 else -1, self.terminado  
        elif 0 not in self.tabuleiro:
            self.terminado = True
            return tuple(self.tabuleiro), 0, self.terminado 

        return tuple(self.tabuleiro), 0, self.terminado  

    def verificar_vencedor(self, jogador):
        posicoes_vencedoras = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        return any(all(self.tabuleiro[i] == jogador for i in pos) for pos in posicoes_vencedoras)

    def obter_estado_tensor(self):
        return torch.FloatTensor(self.tabuleiro).unsqueeze(0)

class AgenteQLearning:
    def __init__(self, taxa_aprendizado=0.1, fator_desconto=0.9, taxa_exploracao=0.1):
        self.tabela_q = {}
        self.taxa_aprendizado = taxa_aprendizado
        self.fator_desconto = fator_desconto
        self.taxa_exploracao = taxa_exploracao

    def escolher_acao(self, estado):
        estado = tuple(estado)
        acoes_disponiveis = [i for i, val in enumerate(estado) if val == 0]
        if not acoes_disponiveis:
            return None
            
        if random.random() < self.taxa_exploracao or estado not in self.tabela_q:
            return random.choice(acoes_disponiveis)
        return max(self.tabela_q[estado], key=self.tabela_q[estado].get)

    def aprender(self, estado, acao, recompensa, proximo_estado):
        estado, proximo_estado = tuple(estado), tuple(proximo_estado)
        if estado not in self.tabela_q:
            self.tabela_q[estado] = {i: 0 for i, val in enumerate(estado) if val == 0}
        if proximo_estado not in self.tabela_q:
            self.tabela_q[proximo_estado] = {i: 0 for i, val in enumerate(proximo_estado) if val == 0}

        max_proximo_q = max(self.tabela_q[proximo_estado].values()) if self.tabela_q[proximo_estado] else 0
        alvo_td = recompensa + self.fator_desconto * max_proximo_q
        self.tabela_q[estado][acao] += self.taxa_aprendizado * (alvo_td - self.tabela_q[estado][acao])

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class AgenteDQN:
    def __init__(self, taxa_exploracao=0.1, fator_desconto=0.95):
        self.rede = DQN()
        self.rede_alvo = copy.deepcopy(self.rede)
        self.otimizador = optim.Adam(self.rede.parameters(), lr=0.001)
        self.memoria = deque(maxlen=10000)
        self.taxa_exploracao = taxa_exploracao
        self.fator_desconto = fator_desconto
        self.contador_atualizacao = 0

    def escolher_acao(self, estado):
        if random.random() < self.taxa_exploracao:
            return random.choice([i for i, val in enumerate(estado) if val == 0])
        
        estado_tensor = torch.FloatTensor(estado).unsqueeze(0)
        with torch.no_grad():
            valores_q = self.rede(estado_tensor)
        
        mascara = torch.FloatTensor([1 if x == 0 else -float('inf') for x in estado])
        acao = (valores_q + mascara).argmax().item()
        return acao

    def armazenar_transicao(self, estado, acao, recompensa, proximo_estado, terminado):
        self.memoria.append((estado, acao, recompensa, proximo_estado, terminado))

    def aprender(self, tamanho_lote=32):
        if len(self.memoria) < tamanho_lote:
            return
        
        lote = random.sample(self.memoria, tamanho_lote)
        estados, acoes, recompensas, proximos_estados, terminados = zip(*lote)
        
        estados = torch.FloatTensor(estados)
        acoes = torch.LongTensor(acoes).unsqueeze(1)
        recompensas = torch.FloatTensor(recompensas).unsqueeze(1)
        proximos_estados = torch.FloatTensor(proximos_estados)
        terminados = torch.FloatTensor(terminados).unsqueeze(1)
        
        q_atual = self.rede(estados).gather(1, acoes)
        
        with torch.no_grad():
            q_proximo = self.rede_alvo(proximos_estados).max(1)[0].unsqueeze(1)
            alvo = recompensas + (1 - terminados) * self.fator_desconto * q_proximo
        
        perda = nn.MSELoss()(q_atual, alvo)
        
        self.otimizador.zero_grad()
        perda.backward()
        self.otimizador.step()
        
        self.contador_atualizacao += 1
        if self.contador_atualizacao % 100 == 0:
            self.rede_alvo.load_state_dict(self.rede.state_dict())

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class AgentePolicyGradient:
    def __init__(self):
        self.rede = PolicyNetwork()
        self.otimizador = optim.Adam(self.rede.parameters(), lr=0.01)
        self.memoria = []
        self.recompensa_episodio = 0

    def escolher_acao(self, estado):
        estado_tensor = torch.FloatTensor(estado).unsqueeze(0)
        probs = self.rede(estado_tensor)
        mascara = torch.FloatTensor([1 if x == 0 else 0 for x in estado])
        probs_validas = probs * mascara
        
        if probs_validas.sum().item() == 0:
            return random.choice([i for i, val in enumerate(estado) if val == 0])
        
        probs_validas /= probs_validas.sum()
        acao = torch.multinomial(probs_validas, 1).item()
        return acao

    def armazenar_transicao(self, estado, acao, recompensa):
        self.memoria.append((estado, acao, recompensa))
        self.recompensa_episodio += recompensa

    def aprender(self):
        if not self.memoria:
            return
        
        estados, acoes, recompensas = zip(*self.memoria)
        estados = torch.FloatTensor(estados)
        acoes = torch.LongTensor(acoes)
        recompensas = torch.FloatTensor(recompensas)
        
        # Normalizar recompensas
        recompensas = (recompensas - recompensas.mean()) / (recompensas.std() + 1e-7)
        
        probs = self.rede(estados)
        probs_acoes = probs.gather(1, acoes.unsqueeze(1)).squeeze()
        
        perda = -torch.mean(torch.log(probs_acoes) * self.recompensa_episodio)
        
        self.otimizador.zero_grad()
        perda.backward()
        self.otimizador.step()
        
        self.memoria = []
        self.recompensa_episodio = 0

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
            nn.Softmax(dim=-1)
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

class AgenteActorCritic:
    def __init__(self):
        self.rede = ActorCritic()
        self.otimizador = optim.Adam(self.rede.parameters(), lr=0.001)
        self.memoria = []

    def escolher_acao(self, estado):
        estado_tensor = torch.FloatTensor(estado).unsqueeze(0)
        probs = self.rede.actor(estado_tensor)
        mascara = torch.FloatTensor([1 if x == 0 else 0 for x in estado])
        probs_validas = probs * mascara
        
        if probs_validas.sum().item() == 0:
            return random.choice([i for i, val in enumerate(estado) if val == 0])
        
        probs_validas /= probs_validas.sum()
        acao = torch.multinomial(probs_validas, 1).item()
        return acao

    def armazenar_transicao(self, estado, acao, recompensa, proximo_estado, terminado):
        self.memoria.append((estado, acao, recompensa, proximo_estado, terminado))

    def aprender(self):
        if not self.memoria:
            return
        
        estados, acoes, recompensas, proximos_estados, terminados = zip(*self.memoria)
        
        estados = torch.FloatTensor(estados)
        acoes = torch.LongTensor(acoes)
        recompensas = torch.FloatTensor(recompensas)
        proximos_estados = torch.FloatTensor(proximos_estados)
        terminados = torch.FloatTensor(terminados)
        
        
        valores = self.rede.critic(estados).squeeze()
        proximos_valores = self.rede.critic(proximos_estados).squeeze()
        
        
        vantagens = recompensas + (1 - terminados) * 0.95 * proximos_valores - valores
        
      
        probs = self.rede.actor(estados)
        probs_acoes = probs.gather(1, acoes.unsqueeze(1)).squeeze()
        
        perda_ator = -(torch.log(probs_acoes) * vantagens.detach()).mean()
        perda_critico = nn.MSELoss()(valores, recompensas + (1 - terminados) * 0.95 * proximos_valores)
        perda_total = perda_ator + perda_critico
        
        self.otimizador.zero_grad()
        perda_total.backward()
        self.otimizador.step()
        
        self.memoria = []

class InterfaceJogoDaVelha:
    def __init__(self, raiz):
        self.ambiente = JogoDaVelha()
        self.estado = self.ambiente.reiniciar()
        
        #
        self.tipo_agente = "DQN" 
        
        if self.tipo_agente == "Q-Learning":
            self.agente = AgenteQLearning()
        elif self.tipo_agente == "DQN":
            self.agente = AgenteDQN()
        elif self.tipo_agente == "PolicyGradient":
            self.agente = AgentePolicyGradient()
        elif self.tipo_agente == "ActorCritic":
            self.agente = AgenteActorCritic()
        else:
            raise ValueError("Tipo de agente inválido")

        self.raiz = raiz
        self.raiz.title(f"Jogo da Velha - {self.tipo_agente}")
        self.raiz.configure(bg="#FFC0CB")

        self.botoes = []
        for i in range(9):
            btn = tk.Button(raiz, text="", font=("Comic Sans MS", 20), width=6, height=2,
                          bg="#FF69B4", fg="white", relief=tk.GROOVE,
                          activebackground="#FFB6C1", command=lambda i=i: self.jogada_jogador(i))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.botoes.append(btn)

        self.botao_reiniciar = tk.Button(raiz, text="Reiniciar", font=("Comic Sans MS", 14),
                                       bg="#FF1493", fg="white", relief=tk.RAISED,
                                       command=self.reiniciar_jogo)
        self.botao_reiniciar.grid(row=3, column=0, columnspan=3, pady=10)

    def jogada_jogador(self, indice):
        if self.ambiente.tabuleiro[indice] == 0 and not self.ambiente.terminado:
            self.atualizar_tabuleiro(indice, "X", 1)
            if not self.ambiente.terminado:
                self.raiz.after(500, self.jogada_agente)

    def jogada_agente(self):
        acao = self.agente.escolher_acao(self.estado)
        if acao is not None:
            self.atualizar_tabuleiro(acao, "O", -1)

    def atualizar_tabuleiro(self, indice, simbolo, jogador):
        proximo_estado, recompensa, terminado = self.ambiente.jogada(indice, jogador)
        
        if hasattr(self.agente, 'armazenar_transicao'):
            if self.tipo_agente == "PolicyGradient":
                self.agente.armazenar_transicao(self.estado, indice, recompensa)
            else:
                self.agente.armazenar_transicao(self.estado, indice, recompensa, proximo_estado, terminado)
        
        self.estado = proximo_estado
        self.botoes[indice].config(text=simbolo, state="disabled", bg="#DB7093")
        
    
        if jogador == -1 and hasattr(self.agente, 'aprender'):
            self.agente.aprender()

        if terminado:
            if recompensa == 1 and jogador == 1:
                messagebox.showinfo("Fim de Jogo", "Você venceu!")
            elif recompensa == -1 and jogador == -1:
                messagebox.showinfo("Fim de Jogo", "O agente venceu!")
            else:
                messagebox.showinfo("Fim de Jogo", "Empate!")

    def reiniciar_jogo(self):
        self.estado = self.ambiente.reiniciar()
        for btn in self.botoes:
            btn.config(text="", state="normal", bg="#FF69B4")

if __name__ == "__main__":
    raiz = tk.Tk()
    interface = InterfaceJogoDaVelha(raiz)
    raiz.mainloop()