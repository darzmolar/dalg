#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Created on Wed Oct 19 15:48:00 2018

@author: Daniel Cuesta, Alejandro Garo
"""

import string, random
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

import queue

from sklearn.linear_model import LinearRegression

import networkx as nx

# Ejercicio 1B: Agrandando nuestra ED
def m_mg_2_d_mg(m_mg):
    """
    De un multigrafo dirigido sin pesos, genera un diccionario de adyacencia a partir de su matriz de adyacencia.
    g[u]: diccionario de diccionarios con los nodos destino y el num de caminos con su peso (1 por ser unweighted)
    g[u][v]: num de caminos con su peso (1)
    Devuelve la matriz de adyacencia.

    Parámetros
    ----------
    m_mg: matriz de adyacencia, contiene el numero de caminos del nodo u al uv
    """
    for i in range(m_mg.shape[0]):
        d_mg[i] = {}
        for j in range(m_mg.shape[1]):
            d_mg[i][j] = {}
            for n_edge in range(m_mg[i][j]):
                d_mg[i][j].update({n_edge : 1})
    return d_mg

def rand_unweight_multigraph(n_nodes, num_max_multiple_edges = 3, prob = 0.5):
    """
    Genera un multigrafo dirigido sin pesos haciendo uso del metodo binomial del paquete random de numpy.
    Parámetros
    ----------
    n_nodes: numero de nodos
    num_max_multiple_edges: numero maximo de aristas de un nodo
    prob: probabilidad de generacion de las aristas
    """
    return np.random.binomial(num_max_multiple_edges, prob, (n_nodes,n_nodes))


def graph_2_multigraph(d_g):
    """
    Convierte un grafo que usa nuestra ED diccionario anterior a la nueva

    Parámetros
    ----------
    d_g: grafo usando la ED diccionario anterior
    """
    d_mg = {}
    for u in d_g.keys():
        d_mg.update({u : {}})
        # print(d_mg[u])
        #
        #
        
        for v in d_g[u].keys():
            d_mg[u].update({v:{0:1}})

    return d_mg
                
            
def print_multi_graph(d_mg):
    """
    Imprime el multigrafo dirigido sin pesos
    
    Parámetros
    ----------
    d_mg: multigrafo usando la ED diccionario
    """
    for u in d_mg.keys():
        # print(d_mg[u])
        for v in d_mg[u].keys():
            l_w = []
            for w in d_mg[u][v].keys():
                l_w.append([d_mg[u][v][w]])
            print("(" + str(u) + "," + str(v) + ")" + ": " + str(l_w))


# Ejercicio 2A: Detectando caminos eulerianos.

def adj_inc_directed_multigraph(d_mg):
    """
    Devuelve las adyacencias e incidencias de cada vertice del multigrafo d_mg
    
    Parámetros
    ----------
    d_mg: multigrafo usando la ED diccionario
    """
    
    inc = [0]*len(d_mg)
    adj = [0]*len(d_mg)
    for u in d_mg.keys():
        for v in d_mg[u].keys():
            #evitamos contar los ciclos en un mismo nodo
            if u != v:
                adj[u] = adj[u] + len(d_mg[u][v])
                inc[v] = inc[v] + len(d_mg[u][v])

    return (inc, adj)


def isthere_euler_path_directed_multigraph(d_mg):
    """
    Devuelve True o False segun haya o no un camino euleriano en el multigrafo d_mg
    
    Parámetros
    ----------
    d_mg: multigrafo usando la ED diccionario
    """
    
    inc, adj = adj_inc_directed_multigraph(d_mg)
    
    if sum(inc) != sum(adj):
        return False
    
    l = [adj_i - inc_i for adj_i, inc_i in zip(adj, inc)]
    k = 0
    for u in l:
        if u == 1:
            k+=1
        if u > 1 or u < -1 or (inc[u] == 0 and adj[u] == 0) or k == 2:
            return False
    
    return True



def first_last_euler_path_directed_multigraph(d_mg):
    """
    Devuelve el punto inicial y el punto final del multigrafo d_mg
    
    Parámetros
    ----------
    d_mg: multigrafo usando la ED diccionario
    """
    
    first = []
    last = []
    
    if isthere_euler_path_directed_multigraph(d_mg) == False:
        return []
    
    inc, adj = adj_inc_directed_multigraph(d_mg)
    
    for u in range(len(inc)):
        if inc[u] - adj[u] > 0:
            last = u
        
        if inc[u] - adj[u] < 0:
            first = u
    
    if first == [] or last == []:
        first = 0
        last = 0
    
    return first, last


def euler_walk_directed_multigraph(u, d_mg):
    """
    Devuelve un camino euleriano del multigrafo d_mg
    
    Parámetros
    ----------
    u: vertice inicial
    d_mg: multigrafo usando la ED diccionario
    """
    
    cam_eu = [u]
    n_act = u
    
    inc, adj = adj_inc_directed_multigraph(d_mg)
    
    if adj[u] == 0:
        return []

    for x in d_mg.keys():
        if n_act == x:
            for y in d_mg[x].keys():
                if n_act != y:
                    cam_eu.append(y)
                    if list(d_mg[x][y].keys()) != []:
                        k = list(d_mg[x][y].keys())[0]
                        del d_mg[x][y][k]
                        adj[x] = adj[x] - 1
                        inc[y] = inc[y] - 1
                        n_act = y
    return cam_eu
    

def next_first_node(l_path, d_mg):
    """
    Devuelve el siguiente nodo del camino euleriano del multigrafo d_mg
    
    Parámetros
    ----------
    l_path: lista de nodos visitados
    d_mg: multigrafo usando la ED diccionario
    """
    
    u = l_path[-1]
    
    l_2 = euler_walk_directed_multigraph(u, d_mg)
    
    return l_2[1]
    

def path_stitch(path_1, path_2):
    """
    Devuelve la union de las dos listas
    
    Parámetros
    ----------
    path1: lista de nodos 1
    path2: lista de nodos 2
    """    
    
    if path_1[-1] == path_2[0]:
        l_path = path_1
        l_path.pop(-1)
        for x in range(len(path_2)):
            l_path.append(path_2[x])
    else:
        l_path = path_2
        l_path.pop(-1)
        for x in range(len(path_1)):
            l_path.append(path_1[x])
    
    return l_path
    

def euler_path_directed_multigraph(d_mg):
    """
    Devuelve el camino del multigrafo d_mg
    
    Parámetros
    ----------
    l_path: lista de nodos visitados
    d_mg: multigrafo usando la ED diccionario
    """
    
    l_path = []
    
    for x in d_mg.keys():
        if l_path == []:
            l_path = euler_walk_directed_multigraph(x, d_mg)
            
        else:
            path_aux = euler_walk_directed_multigraph(x, d_mg)
            if path_aux != []:
                l_path = path_stitch(l_path, path_aux)
    
    return l_path


def isthere_euler_circuit_directed_multigraph(d_mg):
    """
    Devuelve True o False segun haya o no un circuito euleriano en el multigrafo d_mg
    
    Parámetros
    ----------
    d_mg: multigrafo usando la ED diccionario
    """
    
    if isthere_euler_path_directed_multigraph == False:
        return False
    
    first, last = first_last_euler_path_directed_multigraph(d_mg)
    
    if first != last:
        return False
    
    return True



def euler_circuit_directed_multigraph(d_mg, u=0):
    
    if isthere_euler_circuit_directed_multigraph == False:
        return
    
    cam_eu = [u]
    n_act = u
    
    inc, adj = adj_inc_directed_multigraph(d_mg)
    
    if adj[u] == 0:
        return []
    while sum(inc) != 0:
        for x in d_mg.keys():
            if n_act == x:
                for y in d_mg[x].keys():
                    if n_act != y:
                        cam_eu.append(y)
                        if list(d_mg[x][y].keys()) != []:
                            k = list(d_mg[x][y].keys())[0]
                            del d_mg[x][y][k]
                            adj[x] = adj[x] - 1
                            inc[y] = inc[y] - 1
                            n_act = y
    return cam_eu
    

# Ejercicio 3-A Secuenciación de lecturas

def random_sequence(len_seq):
    """
    Genera una secuencia aleatoria de longitud len_seq
    
    Parámetros
    ----------
    len_seq:longitud de la secuencia
    """
    seq = []
    elems_seq = ['A', 'C', 'G', 'T']
    for i in range(len_seq):
        seq.append(random.choice(elems_seq))
    return seq


def spectrum(sequence, len_read):
    """
    Genera un l-espectro a partir de la secuencia con tamaño de lectura len_read 
    Si el tamaño es incorrecto devuelve []

    Parámetros
    ----------
    sequence: secuencia
    len_read: tamaño de lectura
    """
    l_spectr = []
    lrandom_spectr = []
    i = 0
    # for i in range(len(sequence)):
    if len_read < 2:
        print("Tamaño de lectura incorrecta.")
        return []
    
    while(len(sequence[i : i + len_read : ]) == len_read):
        if sequence[i: i + len_read:] not in l_spectr:
            l_spectr.append(sequence[i: i + len_read:])
        # l_spectr.append(sequence[i: i + len_read:])
        i += 1
    # print("ESPECTRO ORDENADO: ", l_spectr)
    return random.sample(l_spectr, len(l_spectr))

def spectrum_2(spectr):
    """
    Genera un (l-1)-espectro a partir de un l-espectro
    Si el tamaño es incorrecto devuelve []
    Parámetros
    ----------
    spectr: l-espectro
    """
    if len(spectr[0]) < 2:
        print("Tamaño de espectro incorrecto")
        return []
    
    l1_spectr = []
    for sub_spectr in spectr:
        if sub_spectr[:len(sub_spectr) - 1] not in l1_spectr:
            l1_spectr.append(sub_spectr[:len(sub_spectr) - 1])
        if sub_spectr[1 : len(sub_spectr)] not in l1_spectr:
            l1_spectr.append(sub_spectr[1 : len(sub_spectr)])
    return l1_spectr

def spectrum_2_graph(spectr):
    """
    Genera el diccionario asociado a un espectro
    
    Parámetros
    ----------
    spectr: espectro
    """
    d_mg = {}
    n_paths = 0
    
    for sub_spectr in spectr:
        d_mg[spectr.index(sub_spectr)] = {}
        # print("Sub_spectr: ", sub_spectr)
        # print("D_MG1: ", d_mg)
        # print("Index: ", spectr.index(sub_spectr))
        for i, end_sub_spectr in enumerate(spectr):
            # Compruebo si hace ciclo:
            if i != d_mg[spectr.index(sub_spectr)]:
                if sub_spectr[1:] == end_sub_spectr[:len(end_sub_spectr) - 1]:
                    d_mg[spectr.index(sub_spectr)].update({spectr.index(end_sub_spectr) : {n_paths : 1}})
                    #print(spectr.index(sub_spectr))
                    #print(spectr.index(end_sub_spectr))
                    n_paths += 1
            n_paths = 0
        # if d_mg[spectr.index(sub_spectr)] == {}:
        #     print("Vacio")
    # print(d_mg)
    return d_mg



# Ejercicio 3-B Todo junto

def check_sequencing(len_seq, len_read):
    """
    Construye un espectro con lecturas de tamaño len_read y su camino euleriano.
    Si no tuviera camino euleriano, lo devuelve como []
    Devuelve el camino y el espectro.
    Parámetros
    ----------
    len_seq: longitud de la secuencia
    len_read: longitud de las lecturas
    """
    
    # Genero la secuencia
    seq = random_sequence(len_seq)
    # print("Secuencia")
    # print(seq)
    # Genero los espectros
    l_spectr = spectrum(seq, len_read)
    # print("Espectro desordenado")
    # print(l_spectr)

    # Las convierto a dict y busco el camino euleriano
    dg_mg_spectr = spectrum_2_graph(l_spectr)
    # print("Dicts")
    # print(dg_mg_spectr)
    
    # Check de las adyacencias e incidencias
    inc, adj = adj_inc_directed_multigraph(dg_mg_spectr)
    # print(inc, adj)
    dg_mg_u= first_last_euler_path_directed_multigraph(dg_mg_spectr)    
    # print("Nodo inicial y final:")
    # print(dg_mg_u)
    if dg_mg_u != []:
        for i in range(len(adj)):
            if i == dg_mg_u[0]:           # Nodo inicial, inc[0] = adj[0] - 1
                if inc[i] != adj[i] - 1:
                    print("La incidencia del nodo inicial tiene que ser la adyacencia del nodo inicial - 1")
                    return [], l_spectr 
            elif i == dg_mg_u[1]:
                if inc[i] != adj[i] + 1:          # Nodo final, inc[last] = adj[last] + 1
                    print("La incidencia del nodo final tiene que ser la adyacencia del nodo final + 1")
                    return [], l_spectr
            else:
                if inc[i] != adj[i]:
                    print("Los nodos intermedios tienen que tener adyacencias e incidencias iguales")
                    return [], l_spectr
    else:
        print("No hay camino euleriano")
        return [], l_spectr

    # Obtencion del EP
    ep_seq = euler_path_directed_multigraph(dg_mg_spectr)
    print("Camino euleriano de seq: " + str(ep_seq))
    
    return ep_seq, l_spectr
        
def path_2_sequence(l_path, spectrum_2):
    """
    Reconstruye una secuencia dado un camino euleriano y su espectro
    Devuelve el camino y el espectro, en caso de no existir camino euleriano 
    o fallo en la longitud de las lecturas, devuelve []
    
    Parámetros
    ----------
    l_path: camino
    spectrum_2: espectro
    """
    
    if l_path is []:
        print("No existe camino euleriano para este espectro.")
        return []
    
    # print(l_path, spectrum_2)
    if len(spectrum_2[0]) < 2:
        print("Tamaño de secuencia incorrecta")
        return []
    
    # Las convierto a dict y busco el camino euleriano
    dg_mg_spectr = spectrum_2_graph(spectrum_2)
    # print("Dicts")
    # print(dg_mg_spectr)
    
    # Check de las adyacencias e incidencias
    inc, adj = adj_inc_directed_multigraph(dg_mg_spectr)
    # print(inc, adj)
    dg_mg_u = first_last_euler_path_directed_multigraph(dg_mg_spectr)    
    # print("Nodo inicial y final:")
    # print(dg_mg_u)
    if dg_mg_u != []:
        for i in range(len(adj)):
            if i == dg_mg_u[0]:           # Nodo inicial, inc[0] = adj[0] - 1
                if inc[i] != adj[i] - 1:
                    print("La incidencia del nodo inicial tiene que ser la adyacencia del nodo inicial - 1")
                    return [], l_spectr 
            elif i == dg_mg_u[1]:
                if inc[i] != adj[i] + 1:          # Nodo final, inc[last] = adj[last] + 1
                    print("La incidencia del nodo final tiene que ser la adyacencia del nodo final + 1")
                    return [], l_spectr
            else:
                if inc[i] != adj[i]:
                    print("Los nodos intermedios tienen que tener adyacencias e incidencias iguales")
                    return [], l_spectr
    else:
        print("No hay camino euleriano")
        return [], l_spectr

    # Obtencion del EP
    ep_seq = euler_path_directed_multigraph(dg_mg_spectr)
    print("Camino euleriano de seq: " + str(ep_seq))
    # Reconstruccion de la secuencia
    seq_rec_aux = [[]]*len(ep_seq)
    j = 0
    for i in ep_seq:
        # print(spectrum_2[i])
        if j == 0:
            seq_rec_aux[j] = spectrum_2[i]
        else:
            seq_rec_aux[j] = spectrum_2[i][len(spectrum_2[i]) -1:] 
        j += 1
    
    seq_rec = []
    for sublist in seq_rec_aux:
        for elem in sublist:
            seq_rec.append(elem)

    return seq_rec

