#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Created on Wed Oct 19 15:48:00 2018

@author: Daniel Cuesta, Alejandro Garo
"""

import string, random
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import math
import cmath

from cmath import pi
from cmath import exp


# In[3]:


def swap(t, i, j):
    aux = t[i]
    t[i] = t[j]
    t[j] = aux


# In[30]:


def split(t, ini, fin):
    """
    Reparte los elementos utilizando como pivote el primer elemento
    
    Parametros
    ----------
    t : tabla
    ini : primer elemento
    fin : ultimo elemento
    """
    pivote = t[ini]
    m = ini
                   
    for i in range(ini+1, fin+1):
        if t[i] < pivote :
            m +=1
            swap(t, i, m)

    swap(t, ini, m)

    return m
            


# In[108]:


def split_pivot(t, ini, fin, pivot=None):
    """
    Reparte los elementos utilizando como pivote el elemento en la posicion pivot
    
    Parametros
    ----------
    t : tabla
    ini : primer elemento
    fin : ultimo elemento
    """
    if (pivot == None):
        pivote = t[ini]
    else:
        pivote = t[pivot]

    swap(t, ini, pivot)
    m = ini

    for i in range(ini+1, fin+1):
        if t[i] < pivote :
            m +=1
            swap(t, i, m)

    swap(t, ini, m)

    return m
    


# In[116]:


def qselect(t, ini, fin, ind, pivot=None):
    """
    Selecciona el elemento de la posicion definida por ind.

    Parámetros
    ----------
    t : tabla
    ini : primer elemento
    fin : ultimo elemento
    ind : indice de la tabla 
    """
    if ini > ind or ind > fin:
        return

    m = split_pivot(t, ini, fin, ind)

    if ind == m:
        return t[m], m
    elif ind < m:
        return qselect(t, ini, m-1, ind)
    else:
        return qselect(t, m+1, fin, ind)


# In[138]:


def qselect_sr(t, ini, fin, ind, pivot=None):
    """
    Selecciona el elemento de la posicion definida por ind sin recursion.

    Parámetros
    ----------
    t : tabla
    ini : primer elemento
    fin : ultimo elemento
    ind : indice de la tabla 
    """
    if ini > ind or ind > fin:
        return
    
    m = split_pivot(t, ini, fin, ind)
    
    while (ind!=m):
    
        m = split_pivot(t, ini, fin, ind)
        
        if ind < m:
            ini = ini
            fin = m-1
        else:
            ini = m+1
            fin = fin

    return t[m], m


# In[27]:


def pivot_5(t, ini, fin):
    """
    Realiza la mediana de medianas de la tabla.

    Parámetros
    ----------
    t : tabla
    ini : primer elemento
    fin : ultimo elemento
    """
    ts = sorted(t)
    return (ts[ini]+ts[fin])//2


# In[28]:


def qselect_5(t, ini, fin, pos):
    """
    Selecciona el elemento en posicion pos.

    Parámetros
    ----------
    t : tabla
    ini : primer elemento
    fin : ultimo elemento
    pos : elemento a buscar 
    """
    if ini > pos or pos > fin:
        return
    
    piv = pivot_5(t, ini, fin)
    sp = split_pivot(t, ini, fin, piv)
    
    while (pos!=sp):
        piv = pivot_5(t, ini, fin)
        sp = split_pivot(t, ini, fin, piv)

        if pos < sp:
            #ini = ini
            fin = sp-1
        else:
            ini = sp+1
            #fin = fin

    return t[sp], sp


# In[29]:


def qsort_5(t, ini, fin):
    """
    Ordena la tabla.

    Parámetros
    ----------
    t : tabla
    ini : primer elemento
    fin : ultimo elemento
    """
    if ini > fin: 
        return None
    if ini == fin:
        return t
    
    else:
        piv = pivot_5(t, ini, fin)
        sp = split_pivot(t, ini, fin, piv)
        
        if ini < sp-1:
            if qsort_5(t, ini, sp-1).any() == None:
                return None
        
        if sp+1 < fin:
            if qsort_5(t, sp+1, fin).any() == None:
                return None
    return t

