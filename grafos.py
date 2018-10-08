# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:40:00 2017

@author: Enrique Aracil, Daniel Cuesta
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import gzip
from six.moves import cPickle
import os
import queue
from queue import PriorityQueue

def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50.):
	
	"""
	Genera una matriz de adyacencia a partir de un grafo dirigido ponderado.
	Devuelve la matriz de adyacencia.

	Parámetros
	----------
	n_nodes : numero de nodos
	sparse_factor : proporcion de ramas
	max_weight : peso maximo
	"""

	matriz = np.zeros((n_nodes,n_nodes))

	for i in range (0,n_nodes):
		for j in range (0,n_nodes):
			if i != j:
				aleat = np.random.rand()
				if aleat < sparse_factor:
					aleat = np.random.randint(1, max_weight)
					matriz[i][j] = aleat
				else:
					matriz[i][j] = np.inf
	return matriz

def cuenta_ramas(m_g):
	
	"""
	Cuenta el numero de ramas que hay en la matriz de adyacencia.
	Devuelve el numero de ramas.

	Parámetros
	----------
	m_g : matriz de adyacencia
	"""

	num = 0
	for i in range (0, m_g.shape[0]):
		for j in range (0, m_g.shape[1]):
			if i != j:
				if np.isinf(m_g[i][j]) != True:
					num += 1
	return num

def check_sparse_factor(n_grafos, n_nodes, sparse_factor):
	
	"""
	Genera las matrices de varios grafos aleatorios y sus proporciones.
	Devuelve la media de las proporciones.

	Parámetros
	----------
	n_grafos : numero de grafos
	n_nodes : numero de nodos
	sparse_factor : proporcion de ramas
	"""

	acc = 0
	for i in range(0,n_grafos):
		m = rand_matr_pos_graph(n_nodes, sparse_factor)
		acc += cuenta_ramas(m)

	return acc//(i+1)

def m_g_2_d_g(m_g):
	
	"""
	Genera un diccionario de listas de adyacencia del grafo 
	definido por la matriz.
	Devuelve el diccionario.

	Parámetros
	----------
	m_g : matriz de adyacencia
	"""

	dic = {}

	for i in range (0, m_g.shape[0]):
		tuplas = []
		for j in range (0, m_g.shape[1]):
			if i != j:
				if np.isinf(m_g[i][j]) != True:
					tuplas.append((j, m_g[i][j]))
		dic.update({i:tuplas})

	return dic

def d_g_2_m_g(d_g):
	
	"""
	Genera una matriz de adyacencia del grafo 
	definido por el diccionario.
	Devuelve la matriz.

	Parámetros
	----------
	d_g : diccionario de listas de adyacencia
	"""

	#recorrerDic
	keys = list(d_g.keys())
	max = keys[-1]
	n_reci = []
	for values in d_g.values():
		for tupla in values:
			n_reci.append(tupla[0])
	#n_reci.sort(None, True)
	if n_reci[0] > max:
		max = n_reci[0]

	matriz = np.zeros((max+1,max+1))

	#rellenamos infs
	for i in range (0,max+1):
		for j in range (0,max+1):
			if i != j:
				matriz[i][j] = np.inf

	#rellenamos los costes
	for i in d_g:
		for tupla in d_g[i]:
			matriz[i][tupla[0]] = tupla[1]

	return matriz

def save_object(obj, f_name='obj.pklz', save_path='.'):
	
	"""
 	Guarda un objeto Python de manera comprimida en un fichero.

	Parámetros
	----------
	obj : objeto python
	f_name : fichero donde guardar
	save_path : 
	"""

	f = gzip.open(f_name,'wb')
	cPickle.dump(obj, f)
	f.close()

	if save_path!='.':
		os.rename(os.getcwd()+"/"+f_name, 
			save_path+"/"+f_name)

def read_object(f_name, save_path='.'):
	
	"""
 	Devuelve un objeto Python guardado en un fichero.

	Parámetros
	----------
	f_name : fichero donde leer
	save_path : 
	"""

	#cambia work directory
	if save_path!='.':
		path = os.getcwd()
		os.chdir(save_path)

	f = gzip.open(f_name,'rb')
	obj = cPickle.load(f)
	f.close()

	#vuelve a work directory original
	if save_path!='.':
		os.chdir(path)

	return obj

def d_g_2_TGF(d_g, f_name):
	
	"""
	Guarda el grafo definido por el diccionario en un fichero.

	Parámetros
	----------
	d_g : diccionario de listas de adyacencia
	f_name : fichero donde guardar
	"""

	f = open(f_name, 'w')

	for key in d_g.keys():
		f.write(str(key)+"\n")
	f.write("#\n")

	for i in d_g:
		for tupla in d_g[i]:
			f.write(str(i)+" ")
			f.write(str(tupla[0])+" ")
			f.write(str(tupla[1])+"\n")

	f.close()

def TGF_2_d_g(f_name):
	
	"""
	Genera un diccionario de listas de adyacencia a partir de un fichero.
	Devuelve el diccionario.

	Parámetros
	----------
	f_name : fichero donde leer
	"""
	d_g = {}
	relation = 0
	nodes = []

	f = open(f_name, 'r')
	for line in f:

		if relation == 0:
			nodes.append(line)

		if relation == 1:
			s = line.split()
			if d_g.has_key(int(s[0])) == True:
				#recuperar lista de tuplas de key y actualizarlo
				l_tuplas = d_g.get(int(s[0]))
				l_tuplas.append((int(s[1]), float(s[2])))
				d_g.update({int(s[0]):l_tuplas})
			else:
				#crear key con lista tupla
				l_tuplas = [(int(s[1]), float(s[2]))]
				d_g.update({int(s[0]):l_tuplas})

		if line.find("#") != -1:
			relation = 1

	f.close()

	return d_g


def dijkstra_d(d_g, u):

	"""
	Genera las tablas de previos y de distancias minimas entre 
	el vertice y los demas de un grafo dado por un diccionario.
	Devuelve las tablas de previos y de distancias minimas.

	Parámetros
	----------
	d_g : diccionario de listas de adyacencia
	u : vertice
	"""

	#u, elemento(nodo) del que se calculan distancias
	#d_g grafo en dicc

	pq = queue.PriorityQueue()
	distancia = {}
	previo = {}
	visitado = {}

	#crear diccionarios con indice el nodo y (distancia, previo, visitado)
	n_keys = len(list(d_g.keys()))

	for i in range(0,n_keys):
		distancia.update({i:np.inf})
		previo.update({i:None})
		visitado.update({i:False})

	#la distancia en u es cero
	distancia.update({u:0})

	#insertamos en la pq la tupla (d[u], u)
	pq.put((distancia.get(u), u))

	#Bucle mientras pq no vacia:
	while pq.empty() == False:
		#Escogemos la primera tupla de la pq y pasamos a este nodo n
		tupla = pq.get()
		n = tupla[1]
		#Si este nodo no ha sido visitado lo visitamos v[n] = True
		if visitado.get(n) == False:
			visitado.update({n:True})

			#Para cada camino z en todos los caminos de n
			caminos = d_g.get(n)
			for tupla_camino in caminos:
				z = tupla_camino[0]
				coste_n_z = tupla_camino[1]

				#Si distancia[z] > (distancia[n] + coste(n, z):
				if distancia.get(z) > (distancia.get(n) + coste_n_z):
					#distancia[z] = (distancia[n] + coste(n, z)
					distancia.update({z:(distancia.get(n) + coste_n_z)})
					#previo[z] = n
					previo.update({z:n})
					#insertamos tupla (d[z], z) en pq
					pq.put((distancia.get(z),z))

	return previo, distancia

def dijkstra_m(m_g, u):

	"""
	Genera las tablas de previos y de distancias minimas entre 
	el vertice y los demas de un grafo dado por un matriz.
	Devuelve las tablas de previos y de distancias minimas.

	Parámetros
	----------
	m_g : matriz de listas de adyacencia
	u : vertice
	"""

	#u, elemento(nodo) del que se calculan distancias
	#d_g grafo en dicc

	pq = queue.PriorityQueue()
	distancia = {}
	previo = {}
	visitado = {}

	#crear diccionarios con indice el nodo y (distancia, previo, visitado)
	n_keys = m_g.shape[0]

	for i in range(0,n_keys):
		distancia.update({i:np.inf})
		previo.update({i:None})
		visitado.update({i:False})

	#la distancia en u es cero
	distancia.update({u:0})

	#insertamos en la pq la tupla (d[u], u)
	pq.put((distancia.get(u), u))

	#Bucle mientras pq no vacia:
	while pq.empty() == False:
		#Escogemos la primera tupla de la pq y pasamos a este nodo n
		tupla = pq.get()
		n = tupla[1]
		#Si este nodo no ha sido visitado lo visitamos v[n] = True
		if visitado.get(n) == False:
			visitado.update({n:True})

			#Para cada camino z en todos los caminos de n
			caminos = []
			for i in range(0,n_keys):
				if n != i and np.isinf(m_g[n][i]) != True:
					caminos.append((i, m_g[n][i]))

			for tupla_camino in caminos:
				z = tupla_camino[0]
				coste_n_z = tupla_camino[1]

				#Si distancia[z] > (distancia[n] + coste(n, z):
				if distancia.get(z) > (distancia.get(n) + coste_n_z):
					#distancia[z] = (distancia[n] + coste(n, z)
					distancia.update({z:(distancia.get(n) + coste_n_z)})
					#previo[z] = n
					previo.update({z:n})
					#insertamos tupla (d[z], z) en pq
					pq.put((distancia.get(z),z))

	return previo, distancia

def min_paths(u, p):

	"""
	Genera a los caminos minimos desde un vertice a los demas 
	vertices del grafo.
	Devuelve los vertices con sus costes.

	Parámetros
	----------
	u : vertice
	p : tabla de previos
	"""

	d_paths = {}
	#para cada nodo/key menos u en p
	for key in p.keys():
		if key != u:
			coste = 1
			if p.get(key) == u:
				#coste
				d_paths.update({key:coste})
			else:
				v = p.get(key)
				while p.get(v) != None:					
					coste+=1
					if p.get(v) == u:
						#coste
						d_paths.update({key:coste})
						break
					v = p.get(v)
				if p.get(v) == None:
					d_paths.update({key:None})
		else:
			d_paths.update({key:None})
	return d_paths

def time_dijkstra_m(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):

	"""
	Mide los tiempos que tarda en ejecutar el algoritmo de Dijkstra
	sobre diferentes grafos con diferentes cantidades de nodos dados en 
	matrices.
	Devuelve los tiempos.

	Parámetros
	----------
	n_graphs : numero de grafos
	n_nodes_ini : tamaño de inicio
	n_nodes_fin : tamaño de fin
	step : incremento
	sparse_factor : proporcion de ramas
	"""

	#tiempo inicial
	time1 = time.clock()

	n_nodes = n_nodes_ini

	while n_nodes < n_nodes_fin:

		for i in range (0, n_graphs):		
			#genera grafo en matriz
			matriz = rand_matr_pos_graph(n_nodes, sparse_factor)
			#dijkstra en matriz para cada u
			for u in range (0,matriz.shape[0]-1):
				dijkstra_m(matriz, u)

		n_nodes += step

	#tiempo final
	time2 = time.clock()

	return time2 - time1


def time_dijkstra_d(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):

	"""
	Mide los tiempos que tarda en ejecutar el algoritmo de Dijkstra
	sobre diferentes grafos con diferentes cantidades de nodos dados en diccionarios.
	Devuelve los tiempos.

	Parámetros
	----------
	n_graphs : numero de grafos
	n_nodes_ini : tamaño de inicio
	n_nodes_fin : tamaño de fin
	step : incremento
	sparse_factor : proporcion de ramas
	"""

	#tiempo inicial
	time1 = time.clock()

	n_nodes = n_nodes_ini

	while n_nodes < n_nodes_fin:

		for i in range (0, n_graphs):		
			#genera grafo en matriz
			matriz = rand_matr_pos_graph(n_nodes, sparse_factor)
			#genera dic a partir de matriz
			dic = m_g_2_d_g(matriz)
			#dijkstra en dic para cada u
			for u in dic.keys():
				dijkstra_d(dic, u)

		n_nodes += step

	#tiempo final
	time2 = time.clock()

	return time2 - time1


def dijkstra_m_all_pairs(m_g):

	"""
	Genera una matriz cuya fila i contenga las distancias minimas dij 
	entre el nodo i y los demas nodos j del grafo.
	Devuelve la matriz.

	Parámetros
	----------
	m_g : matriz de adyacencia
	"""

	matriz = np.zeros((m_g.shape[0],m_g.shape[1]))

	for i in range(0,m_g.shape[0]):
		distancias = dijkstra_m(m_g, i)[1]
		for j in distancias.keys():
			matriz[i][j] = distancias.get(j)
	return matriz


def floyd_warshall(m_g):

	"""
	Genera una matriz cuya fila i contenga las distancias minimas dij 
	entre el nodo i y los demas nodos j del grafo.
	Devuelve la matriz.

	Parámetros
	----------
	m_g : matriz de adyacencia
	"""

	n_nodes = m_g.shape[0]
	matriz = np.zeros( (n_nodes, n_nodes, n_nodes+1) )
	matriz[:, :, 0] = m_g

	for k in range(n_nodes):
		for i in range(n_nodes):
			for j in range(n_nodes):
				#t = dik+dkj
				t = matriz[i, k, k] + matriz[k, j, k]
				#dij(k) = min(dij(k-1), dik(k-1)+dkj(k-1))
				matriz[i, j, k+1] = min(matriz[i, j, k], t)

	return matriz[:, :, n_nodes]


def time_dijkstra_m_all_pairs(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor):

	"""
	Mide los tiempos que tarda en ejecutar el algoritmo de Dijkstra
	sobre diferentes grafos con diferentes cantidades de nodos.
	Devuelve los tiempos.

	Parámetros
	----------
	n_graphs : numero de grafos
	n_nodes_ini : tamaño de inicio
	n_nodes_fin : tamaño de fin
	step : incremento
	sparse_factor : proporcion de ramas
	"""

	#tiempo inicial
	time1 = time.clock()

	n_nodes = n_nodes_ini

	while n_nodes < n_nodes_fin:

		for i in range (0, n_graphs):		
			#genera grafo en matriz
			matriz = rand_matr_pos_graph(n_nodes, sparse_factor)
			#aplica dijkstra all pairs a cada matriz
			dijkstra_m_all_pairs(matriz)

		n_nodes += step

	#tiempo final
	time2 = time.clock()

	return time2 - time1

	
def time_floyd_warshall(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor):

	"""
	Mide los tiempos que tarda en ejecutar el algoritmo de Floyd-Warshall
	sobre diferentes grafos con diferentes cantidades de nodos dados en diccionarios.
	Devuelve los tiempos.

	Parámetros
	----------
	n_graphs : numero de grafos
	n_nodes_ini : tamaño de inicio
	n_nodes_fin : tamaño de fin
	step : incremento
	sparse_factor : proporcion de ramas
	"""

	#tiempo inicial
	time1 = time.clock()

	n_nodes = n_nodes_ini

	while n_nodes < n_nodes_fin:

		for i in range (0, n_graphs):		
			#genera grafo en matriz
			matriz = rand_matr_pos_graph(n_nodes, sparse_factor)
			#aplica dijkstra all pairs a cada matriz
			floyd_warshall(matriz)

		n_nodes += step

	#tiempo final
	time2 = time.clock()

	return time2 - time1
