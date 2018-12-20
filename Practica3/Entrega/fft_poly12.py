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
import math
import cmath

from cmath import pi, exp


# In[2]:


def fft(t):
    """
    Algoritmo para el cálculo de la transformada rápida de Fourier
    sobre un array. Calcula también la menor potencia de t que es 
    mayor que el número de elementos de la tabla, añadiéndole un
    número adecuado de 0s.
    Devuelve un array numpy de números complejos

    Parámetros
    ----------
    t: array sobre el cual se calcula la FFT. 
    """
    
    k = len(t)
    
    if k == 1 :
        return t
    
    pow2 = math.ceil(math.log(k,2))
    
    t = np.array(list(t) + (2**pow2 - k) * [int(0)], dtype = complex)
    tt = np.array(list(t) + (2**pow2 - k) * [int(0)], dtype = complex)
        
    c_e = t[::2]
    c_o = t[1::2]

    f_e = fft(c_e)
    f_o = fft(c_o)

    for i in range(2**pow2):
        tt[i] = f_e[i % 2**(pow2-1)] + exp(2j*pi*(i/(2**pow2))) * f_o[i % 2**(pow2-1)]

    # Redondeo
    for i in range(len(tt)):
        tt[i] = round(np.conj(tt[i].real), 2) + round(np.conj(tt[i].imag), 2) * 1j
    # Conjugado
    # tt = [np.conj(elem) for elem in tt]
    
    return tt


# In[4]:


def invert_fft(t, fft_func=fft):
    """
    Algoritmo para el cálculo de la transformada rápida inversa de Fourier
    sobre un array. Calcula también la menor potencia de t que es 
    mayor que el número de elementos de la tabla, añadiéndole un
    número adecuado de 0s.
    Devuelve un array numpy de números complejos

    Parámetros
    ----------
    t: array sobre el cual se calcula la IFFT. 
    """
    k = len(t)
    
    pow2 = math.ceil(math.log(k,2))
    
    # t = np.array(list(t) + (2**pow2 - k) * [int(0)], dtype = complex)
    ifft_t = np.array(list(t) + (2**pow2 - k) * [int(0)], dtype = complex)

    fft_t = fft_func(ifft_t)
    
    # Conjugado
    ifft_t = [elem/2**pow2 for elem in fft_t]

    return ifft_t
    


# In[7]:


# 1-B: Multiplicando polinomios y numeros

def rand_polinomio(long = 2**10, base = 10):
    """
    Genera un polinomio aleatorio de longitud long cuyo grado es 
    long - 1. Los coeficientes estan entre 2 <= base <= 10.
    Devuelve una lista de ints, cuyos grados van de menor a mayor.

    Parámetros
    ----------
    long: longitud del polinomio.
    base: base del polinomio, nos sirve para calcular los coeficientes
    """
    # Comprobacion de la base
    if base < 2 or base > 10:
        return -1
    
    # Computo del polinomio
    l_pol = np.random.randint(0, high = base, size = (1, long)) # En este metodo, base = base - 1
    return l_pol[0]

def poli_2_num(l_pol, base = 10):
    """
    Calcula el valor del polinomio para la base introducida
    usando la regla de Horner.
    Devuelve un int.
    Ej, la lista [4 7 6] nos da polinomio 
    f(x) = 6x^2 + 7^x + 4 = x*(6x + 7) + 4; si x = 10; f(x) = 674
    
    Parámetros
    ----------
    l_pol: polinomio a evaluar.
    base: base deseada 
    """
    if len(l_pol) == 1:    
        return l_pol[0]
    else:
        return l_pol[0] + base * poli_2_num(l_pol[1:], base)
    
# Genera un numero
def rand_numero(num_digits, base = 10):
    """
    Genera un int aleatorio de numero de digitos num_digits y base
    la introducida como parámetro. Hace uso de las funciones pol_2_num y 
    rand_polinomio.
    Devuelve el número generado.
    
    Parámetros
    ----------
    num_digits: numero de digitos
    base: base deseada 
    """
    num = 0
    coef_min = -1
    l_pol_max = [base - 1]
    
    # Encontramos el polinomio en el cual se encuentra el numero con el numero de digitos deseado
    # Para ello establecemos que tiene que tiene que satisfacer que tenga longitud + 1 que num_digits
    # porque tanto el numero minimo como maximo se encontrara en dicho polinomio de dicha longitud
    while len(str(num)) < num_digits + 1:
        l_pol_max.insert(0, base - 1)
        num = poli_2_num(l_pol_max, base)
    
    num = 0
    # Encontramos el polinomio aleatorio al cual sustituyendole la base nos da el numero que tiene
    # de longitud num_digits
    while True:
        l_pol = rand_polinomio(len(l_pol_max), base)
        num = poli_2_num(l_pol, base)
        if len(str(num)) == num_digits:
            break

    return num

def num_2_pol(num, base = 10):
    """
    Convierte un número en su polinomio teniendo en cuenta la base
    introducida.
    Retorna una lista de enteros con los coeficientes ordenados de 
    menor a mayor grado.
    
    Parámetros
    ----------
    num: número a transformar
    base: base deseada 
    """
    # Encontramos el maximo coeficiente del polinomio
    max_deg = math.floor(math.log(num, base))
    l_pol = [0] * (max_deg + 1) # Inicializacion de la lista resultado

    while num > 0:
        for coef in range(9, -1, -1): # Buscamos el coeficiente maximo
            num_aux = num
            n = coef*(base**max_deg) # Operacion coef*x^grado polinomico

            if n <= num_aux:
                num_aux -= n
                num = num_aux
                l_pol[max_deg] = coef
                max_deg -= 1
                break

    return l_pol


# In[32]:


# Multiplicación de polinomios usando el método tradicional
def mult_polinomios(l_pol_1, l_pol_2):
    """
    Multiplica dos polinomios segun el metodo tradicional. Los
    coeficientes vienen dados por las listas pasadas por parámetro.
    Devuelve una lista de enteros con la multiplicación de polinomios,
    cuyos coeficientes están ordenados de menor a mayor grado.
    
    Parámetros
    ----------
    l_pol_1: lista polinómica 1
    l_pol_2: lista polinómica 2
    """
    l_mult = [0] * (len(l_pol_1) + len(l_pol_2) - 1)

    for i in range(len(l_pol_1)):
        for j in range(len(l_pol_2)):
            l_mult[i + j] = (l_pol_1[i] * l_pol_2[j]) + l_mult[i + j]
    
    return l_mult

# Multiplicacion de polinomios usando fft
def mult_polinomios_fft(l_pol_1, l_pol_2, fft_func = fft):
    """
    Multiplica dos polinomios usando la FFT. Los
    coeficientes vienen dados por las listas pasadas por parámetro.
    Devuelve una lista de enteros con la multiplicación de polinomios,
    cuyos coeficientes están ordenados de menor a mayor grado.
    
    Parámetros
    ----------
    l_pol_1: lista polinómica 1
    l_pol_2: lista polinómica 2
    """
    # Correccion de las longitudes de las listas polinomicas para la fft
    if len(l_pol_1) > len(l_pol_2):
        k = len(l_pol_1)
    else:
        k = len(l_pol_2)
        
    if k == 1 :
        return t
    
    pow2 = math.ceil(math.log(k,2))

    l_pol_1 = np.array(list(l_pol_1) + (2**pow2 - len(l_pol_1)) * [int(0)], dtype = complex)
    l_pol_2 = np.array(list(l_pol_2) + (2**pow2 - len(l_pol_2)) * [int(0)], dtype = complex)

    # FFT de cada polinomio
    l_pol_1 = fft_func(l_pol_1)
    l_pol_2 = fft_func(l_pol_2)

    # l_mult = np.array([0] * k, dtype = complex)
    l_mult = np.array([0] * (2**pow2), dtype = complex)

    # Multiplicacion polinomica
    for i in range(len(l_pol_1)):
        l_mult[i] = l_pol_1[i] * l_pol_2[i]

    # Calculo del conjugado
    l_mult = [np.conj(elem) for elem in l_mult]
    
    # Calculo de la 2da FFT
    l_mult = fft_func(l_mult)
    
    # Retorno, en el cual se redondea la parte real y se divide por el tamanho de la lista
    return [int(round(elem.real/2**pow2)) for elem in l_mult]


# Multiplicacion de numeros llevando dichos numeros a sus 
# expresiones polinomicas y siguiendo el metodo tradicional
def mult_numeros(num1, num2):
    """
    Multiplica dos números llevándolos a sus polinomios correspondientes,
    multiplicando dichos polinomios usando la función mult_polinomios
    y calculando el número que le corresponde
    Base supuesta = 10
    Retorna el número resultado del producto de num1*num2
    
    Parámetros
    ----------
    num1: número 1
    num2: número 2
    """
    # Convertimos los numeros a polinomios con base = 10
    l_pol_1 = num_2_pol(num1)
    l_pol_2 = num_2_pol(num2)

    # Multiplicacion de polinomios
    l_mult = mult_polinomios(l_pol_1, l_pol_2)
    
    # Computo del numero resultante
    num_res = 0
    base = 10
    
    for i in range(len(l_mult)):
        num_res += l_mult[i] * (base ** i)

    return num_res

# Multiplicacion de numeros usando fft
def mult_numeros_fft(num1, num2, fft_func = fft):
    """
    Multiplica dos números llevándolos a sus polinomios correspondientes,
    multiplicando dichos polinomios usando la función mult_polinomios_fft
    y calculando el número que le corresponde
    Base supuesta = 10
    Retorna el número resultado del producto de num1*num2
    
    Parámetros
    ----------
    num1: número 1
    num2: número 2
    """
    base = 10
    num = 0
    
    # Conversion de los numeros a polinomios, para ejecutar la multiplicacion usando fft
    l_pol_1 = num_2_pol(num1)
    l_pol_2 = num_2_pol(num2)
    
    pow2_1 = math.ceil(math.log(len(l_pol_1), 2))
    pow2_2 = math.ceil(math.log(len(l_pol_2), 2))
    
    l_pol_1 = np.array(list(l_pol_1) + (2**pow2_1 - len(l_pol_1)) * [int(0)])
    l_pol_2 = np.array(list(l_pol_2) + (2**pow2_2 - len(l_pol_2)) * [int(0)])
    # Multiplicacion de los polinomios usando fft
    l_mult = mult_polinomios_fft(l_pol_1, l_pol_2, fft_func)

    # Computo del numero
    for i in range(len(l_mult)):
        num += l_mult[i].real * (base**i)
        
    return num


# In[ ]:


# 1-C. Midiendo Tiempos

def time_mult_numeros(n_pairs, num_digits_ini, num_digits_fin, step):
    """
    Genera n_pairs parejas de numeros con num_digits_ini + step*k digitos
    con un maximo de num_digits_fin de número de digitos y devuelve una
    lista con el tiempo medio de sus multiplicaciones usando el algoritmo
    mult_numeros
    Retorna una lista con los tiempos medios de sus multiplicaciones
    
    Parámetros
    ----------
    n_pairs: numero de parejas de números a generar
    num_digits_ini: número de digitos iniciales
    num_digits_fin: número de digitos máximos
    step: numero fijo de digitos para el crecimiento del numero de digitos
    de los numeros a multiplicar
    """  
    l_time = []
    num_digits = num_digits_ini
    k = 0
    
    while k != n_pairs:
        time_ini = time.clock() 
        num1 = rand_numero(num_digits)
        num2 = rand_numero(num_digits)
        
        mult = mult_numeros(num1, num2)

        if num_digits >= num_digits_fin:
            num_digits = num_digits_fin
        else:
            num_digits += k*step
        k += 1
            
        time_fin = time.clock()
        l_time.append(time_fin - time_ini)

    return l_time
    
    
def time_mult_numeros_fft(n_pairs, num_digits_ini, num_digits_fin, step, fft_func=fft):
    """
    Genera n_pairs parejas de numeros con num_digits_ini + step*k digitos
    con un maximo de num_digits_fin de número de digitos y devuelve una
    lista con el tiempo medio de sus multiplicaciones usando el algoritmo
    mult_numeros_fft
    Retorna una lista con los tiempos medios de sus multiplicaciones
    
    Parámetros
    ----------
    n_pairs: numero de parejas de números a generar
    num_digits_ini: número de digitos iniciales
    num_digits_fin: número de digitos máximos
    step: numero fijo de digitos para el crecimiento del numero de digitos
    de los numeros a multiplicar
    """
    l_time = []
    num_digits = num_digits_ini
    k = 0
    
    while k != n_pairs:
        time_ini = time.clock()
        num1 = rand_numero(num_digits)
        num2 = rand_numero(num_digits)
        
        mult = mult_numeros_fft(num1, num2)

        if num_digits >= num_digits_fin:
            num_digits = num_digits_fin
        else:
            num_digits += k*step    
        k += 1
        
        time_fin = time.clock()
        l_time.append(time_fin - time_ini)
        
    return l_time
    

