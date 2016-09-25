#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import json
import codecs
import collections
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# Función que crea una bolsa de palabras con el texto de los tweets de todos los usuarios leídos de un fichero dado
# Parámetros:   fileName > Fichero del cual se leen los identificadores de los usuarios de Twitter
#               activeStopWords > Indica si se eliminan o no las stop words. True indica que se eliminan las stopwords
# Valor de retorno: Un diccionario con la cuenta de aparición de cada una de las palabras tokenizadas
def createBagOfWords(fileName, activeStopWords):

    # Inicializamos el tokenizador
    Tokenizador = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=True)

    # Abrimos el fichero para su lectura
    inFile = codecs.open(fileName, 'r', encoding='utf-8')

    # Diccionario de palabras donde se almacenará la frecuencia de aparición
    frequencyDictionary = {}

    # Almacenamos las stopwords, si las vamos a utiliza en el proceso de creación de la BOW
    stopWordsSpanish = []
    if activeStopWords:
        stopWordsSpanish = stopwords.words('spanish')

    # Recorremos cada una de las líneas del fichero
    for line in inFile:

        # Leemos el usuario, país y sexo del fichero
        userId, country, sex = line.split(":::")

        # Montamos el nombre del fichero de los tweets de un usuario
        fileNameUser = "./Data/hispatweets/" + country + "/" + userId + ".json"
        # print("File Name User:", fileNameUser)

        # Recorremos todos los tweets de un determinado usuario
        with codecs.open(fileNameUser, 'r', encoding='utf-8') as f:
            for line in f:
                tweet = json.loads(line)

                tweetTokenized = Tokenizador.tokenize(' '.join(tweet['text'].split()))
                #print("Tweet Tokenizado: ", tweetTokenized)

                # Eliminamos las stopwords si así se ha indicado por parámetro
                if activeStopWords:
                    wordsTextTweet = [w for w in tweetTokenized if w.lower() not in stopWordsSpanish]
                else:
                    wordsTextTweet = tweetTokenized

                # Contamos la frecuencia de aparición de las distintas palabras de todos los tweets
                for word in wordsTextTweet:
                    if word in frequencyDictionary:
                        frequencyDictionary[word] = frequencyDictionary[word] + 1
                    else:
                        frequencyDictionary[word] = 1


    inFile.close()

    # Devolvemos el diccionario con las frecuencias de aparición de cada una de las palabras
    return frequencyDictionary


# Función para guardar en un fichero una lista de palabras pasadas como parámetro
# Parámetros:   fileName > Fichero donde se guardarán las palabras
#               bagOfWords > Lista de palabras a guardar
def saveBagOfWords(fileName, bagOfWords):
    with codecs.open(fileName, 'w', 'utf8') as f:
        for k, v in bagOfWords:
            f.write(k + '\n')


# Función que lee un diccionario de un fichero y lo devuelve en una estructura de diccionario Python con un contador
# asociado a cada palabra
# Parámetros: fileName > Nombre del fichero donde se encuentra el diccionario
# Valor de retorno:  El diccionario en una estructura de diccionario Python donde la clave es la palabra y el valor
# un contador inicializado a 0.
def readDictionary(fileName):
    # Leemos cada una de la líneas, y los guardamos en un diccionario
    inFile = codecs.open(fileName, 'r', encoding='utf-8')
    dictionary = {line.strip('\n'): 0 for line in inFile}
    inFile.close()

    return dictionary


# Función que lee un conjunto de identificadores de un fichero con una estructura "identificador (tabulador) polaridad"
# Parámetros:   fileName > Fichero del cual se leen los identificadores
#               fileNameBOW > Fichero del cual se lee la bolsa de palabras
#               frequency > Parámetro que indica si en la matriz se almacenan frecuencias absolutas (0), relativas (1)
#               o binarias (2)
# Valor de retorno: matrix   > Matriz de características con las frecuencias de aparición de las palabras más frecuentes por cada usuario
#                   vSex     > Vector que contiene el sexo de todos los usuarios de Twitter analizados
#                   vCountry > Vector que contiene el país de origen de todos los usuarios de Twitter analizados
def createMatrixArrays(fileName, fileNameBOW, frequency):

    # Inicializamos el tokenizador
    Tokenizador = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=True)

    # Abrimos el fichero para su lectura
    inFile = codecs.open(fileName, 'r', encoding='utf-8')

    # Leemos el diccionario que contiene la bolsa de palabras (las 1000 más comunes)
    mostCommonWordsDictionary = readDictionary(fileNameBOW)

    # Inicializamos la matriz y los arrays de sexo y país que devolveremos
    matrix = []
    vSex = []
    vCountry = []

    # Recorremos cada una de las líneas del fichero
    for line in inFile:

        # Diccionario de palabras donde se almacenará la frecuencia de aparición
        frequencyDictionary = {}

        # Copiamos el diccionario a utilizar con los contadores inicializados a 0
        frequencyDictionary = mostCommonWordsDictionary.copy()

        # Leemos el usuario, país y sexo del fichero
        userId, country, sex = line.split(":::")

        # Montamos el nombre del fichero de los tweets de un usuario
        fileNameUser = "./Data/hispatweets/" + country + "/" + userId + ".json"
        #print("File Name User:", fileNameUser)

        # Inicializamos el contador de palabras de un usuario
        totalWordsUser = 0

        # Recorremos todos los tweets de un determinado usuario
        with codecs.open(fileNameUser, 'r', encoding='utf-8') as f:
            for line in f:
                tweet = json.loads(line)

                # Tokenizamos el texto del tweet habiendo eliminado separadores "raros"
                tweetTokenized = Tokenizador.tokenize(' '.join(tweet['text'].split()))

                # Actualizamos el número de palabras de un usuario
                totalWordsUser = totalWordsUser + len(tweetTokenized)
                #print("Usuario: ", userId, " - Total de palabras (USUARIO):", totalWordsUser)

                # Contamos la aparición de las palabras de los tweets en la bolsa de palabras
                for word in tweetTokenized:
                    if word in frequencyDictionary:
                        frequencyDictionary[word] = frequencyDictionary[word] + 1


        # En función del parámetro frecuencia, devolveremos la frecuencia relativa de apariciones, la absoluta o la binaria
        if frequency == 0:
            # Devolvemos la frecuencia absoluta
            # Nos quedamos con la frecuencia de aparición de las distintas palabras siempre en el mismo orden (ordenados por la key)
            lineMatrix = [v for (k, v) in sorted(frequencyDictionary.items())]
        elif frequency == 1:
            # Devolvemos la frecuencia relativa
            lineMatrix = []
            for (k, v) in sorted(frequencyDictionary.items()):
                if (v>0):
                    lineMatrix.append(v/totalWordsUser)
                else:
                    lineMatrix.append(v)
            #print("v: ", v, " - totalWordsUser: ", totalWordsUser)
        elif frequency == 2:
            # Devolvemos la frecuencia binaria
            # Nos quedamos con la frecuencia de aparición de las distintas palabras siempre en el mismo orden (ordenados por la key)
            lineMatrix = []
            for (k, v) in sorted(frequencyDictionary.items()):
                if (v>0):
                    lineMatrix.append(1)
                else:
                    lineMatrix.append(0)
        else:
            # Si se indica cualquier otro valor almacenamos la frecuencia absoluta de las apariciones
            # Nos quedamos con la frecuencia de aparición de las distintas palabras siempre en el mismo orden (ordenados por la key)
            lineMatrix = [v for (k, v) in sorted(frequencyDictionary.items())]


        # Anyadimos las distintas frecuencias de aparición a la matriz
        matrix.append(lineMatrix)

        # Incluimos los valores de sexo y país del usuario en el que nos encontramos
        vSex.append(sex.strip('\n'))
        vCountry.append(country)

    inFile.close()


    # Devolvemos la matriz, el vector con los valores del sexo y el vector con los valores de los países
    return matrix, vSex, vCountry



#####################################################################################################################
#####################################################################################################################

# PROCESO DE CONSTRUCCIÓN DE LA BOLSA DE PALABRA (BAG OF WORDS - BOW)
# Descomentar para generar la bolsa de palabras para el sexo

# # BOLSA DE PALABRAS PARA EL SEXO
#
# # Creamos la bolsa de palabras del fichero de training para el sexo (cogemos TODAS las palabras)
# fileNameTrainingSet = "./Data/hispatweets/training.txt"
# BagOfWordsTrainingSex = createBagOfWords(fileNameTrainingSet, False)
#
# # Nos quedamos con las 1000 palabras más frecuentes de todos los tweets leídos
# mostCommonBOWTrainingSex = collections.Counter(BagOfWordsTrainingSex).most_common(1000)
#
# # Guardamos las 1000 palabras más usadas en un fichero para no tener que repetir este proceso
# fileNameBagOfWordsTrainingSex = "./Data/BOWSex.txt"
# saveBagOfWords(fileNameBagOfWordsTrainingSex, mostCommonBOWTrainingSex)

#####################################################################################################################

# BOLSA DE PALABRAS PARA EL PAÍS
# Descomentar para generar la bolsa de palabras para el país de origen

# # Creamos la bolsa de palabras del fichero de training para el sexo (eliminamos algunas palabras - StopWords)
# fileNameTrainingSet = "./Data/hispatweets/training.txt"
# BagOfWordsTrainingCountry = createBagOfWords(fileNameTrainingSet, True)
#
# # Nos quedamos con las 1000 palabras más frecuentes de todos los tweets leídos
# mostCommonBOWTrainingCountry = collections.Counter(BagOfWordsTrainingCountry).most_common(1000)
#
# # Guardamos las 1000 palabras más usadas en un fichero para no tener que repetir este proceso
# fileNameBagOfWordsTrainingCountry = "./Data/BOWCountry.txt"
# saveBagOfWords(fileNameBagOfWordsTrainingCountry, mostCommonBOWTrainingCountry)



#####################################################################################################################
#####################################################################################################################

# PROCESO DE CREACIÓN DE MATRIZ Y VECTORES DE CARACTERÍSTICAS

# MATRIZ Y VECTORES PARA EL SEXO

# Creamos la matriz de características. Cada fila de la matriz recoge la frecuencia de aparición de las 1000
# palabras más comunes que aparecen en los tweets de un determinado autor, es decir, habrá una línea por cada autor.
# Además de eso se devolverán dos vectores, uno donde se indicará si el autor es hombre, mujer o desconocido (male, female, unknown).
# Y otro donde se indica el país al que pertenece el autor (argentina, chile, colombia, españa, mexico, peru o venezuela).

# Cambiar el parámetro frequencySex y frequencyCountry para generar la matrix de características con distintas frecuencias.

#frequencySex = 0        # Frecuencias absolutas
#frequencySex = 1        # Frecuencias relativas
frequencySex = 2        # Frecuencias binarias
fileNameTrainingSet = "./Data/hispatweets/training.txt"
fileNameBagOfWordsTraining = "./Data/BOWSex.txt"
matrixSexTraining, vSexTraining, vCountryTraining = createMatrixArrays(fileNameTrainingSet, fileNameBagOfWordsTraining, frequencySex)

fileNameTestSet = "./Data/hispatweets/test.txt"
matrixSexTest, vSexTest, vCountryTest = createMatrixArrays(fileNameTestSet, fileNameBagOfWordsTraining, frequencySex)


#####################################################################################################################

# MATRIZ Y VECTORES PARA EL PAÍS

# Cambiar el parámetro frequencySex y frequencyCountry para generar la matrix de características con distintas frecuencias.

#frequencyCountry = 0        # Frecuencias absolutas
#frequencyCountry = 1        # Frecuencias relativas
frequencyCountry = 2        # Frecuencias binarias
fileNameTrainingSet = "./Data/hispatweets/training.txt"
fileNameBagOfWordsTraining = "./Data/BOWCountry.txt"
matrixCountryTraining, vSexTraining, vCountryTraining = createMatrixArrays(fileNameTrainingSet, fileNameBagOfWordsTraining, frequencyCountry)

fileNameTestSet = "./Data/hispatweets/test.txt"
matrixCountryTest, vSexTest, vCountryTest = createMatrixArrays(fileNameTestSet, fileNameBagOfWordsTraining, frequencyCountry)


#####################################################################################################################
#####################################################################################################################

# FASE DE ENTRENAMIENTO Y EVALUACIÓN DE LOS MODELOS CREADOS

####################    KNN     ####################

# Entrenamos el modelo para la característica de sexo
sexClassifier = KNeighborsClassifier()
sexClassifier.fit(matrixSexTraining, vSexTraining)
print("Modelo para el sexo entrenado con K-NEIGHBORS ... ")

# Evaluamos el modelo para la caracterísitica de sexo
vSexPredicted = sexClassifier.predict(matrixSexTest)
print("Modelo para el sexo evaluado con K-NEIGHBORS ...")

# Comprobamos la eficiencia del modelo
print( "%d muestras mal clasificadas de %d" % ( (vSexTest != vSexPredicted).sum(), len(vSexTest) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (vSexTest == vSexPredicted).sum() ) / len(vSexTest) ) )

# ###################################################################################################################

# Entrenamos el modelo para la característica del país
countryClassifier = KNeighborsClassifier()
countryClassifier.fit(matrixCountryTraining, vCountryTraining)
print("Modelo para el país entrenado con K-NEIGHBORS ... ")

# Evaluamos el modelo para la caracterísitica del país
vCountryPredicted = countryClassifier.predict(matrixCountryTest)
print("Modelo para el país evaluado con K-NEIGHBORS ...")

# Comprobamos la eficiencia del modelo
print( "%d muestras mal clasificadas de %d" % ( (vCountryTest != vCountryPredicted).sum(), len(vCountryTest) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (vCountryTest == vCountryPredicted).sum() ) / len(vCountryTest) ) )

#

####################   GAUSSIAN NAIVES BAYES    ####################

# Entrenamos el modelo para la característica de sexo
sexClassifier = GaussianNB()
sexClassifier.fit(matrixSexTraining, vSexTraining)
print("Modelo para el sexo entrenado con Naive Bayes ... ")

# Evaluamos el modelo para la caracterísitica de sexo
vSexPredicted = sexClassifier.predict(matrixSexTest)
print("Modelo para el sexo evaluado con Naive Bayes ...")

# Comprobamos la eficiencia del modelo
print( "%d muestras mal clasificadas de %d" % ( (vSexTest != vSexPredicted).sum(), len(vSexTest) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (vSexTest == vSexPredicted).sum() ) / len(vSexTest) ) )

###################################################################################################################

# Entrenamos el modelo para la característica del país
countryClassifier = GaussianNB()
countryClassifier.fit(matrixCountryTraining, vCountryTraining)
print("Modelo para el país entrenado con Naive Bayes ... ")

# Evaluamos el modelo para la caracterísitica del país
vCountryPredicted = countryClassifier.predict(matrixCountryTest)
print("Modelo para el país evaluado con Naive Bayes ...")

# Comprobamos la eficiencia del modelo
print( "%d muestras mal clasificadas de %d" % ( (vCountryTest != vCountryPredicted).sum(), len(vCountryTest) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (vCountryTest == vCountryPredicted).sum() ) / len(vCountryTest) ) )



####################    SVM     ####################

# Entrenamos el modelo para la característica de sexo
sexClassifier = SVC()
sexClassifier.fit(matrixSexTraining, vSexTraining)
print("Modelo para el sexo entrenado con SVC ... ")

# Evaluamos el modelo para la caracterísitica de sexo
vSexPredicted = sexClassifier.predict(matrixSexTest)
print("Modelo para el sexo evaluado con SVC ...")

# Comprobamos la eficiencia del modelo
print( "%d muestras mal clasificadas de %d" % ( (vSexTest != vSexPredicted).sum(), len(vSexTest) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (vSexTest == vSexPredicted).sum() ) / len(vSexTest) ) )

###################################################################################################################

# Entrenamos el modelo para la característica del país
countryClassifier = SVC()
countryClassifier.fit(matrixCountryTraining, vCountryTraining)
print("Modelo para el país entrenado con SVC ... ")

# Evaluamos el modelo para la caracterísitica del país
vCountryPredicted = countryClassifier.predict(matrixCountryTest)
print("Modelo para el país evaluado con SVC ...")

# Comprobamos la eficiencia del modelo
print( "%d muestras mal clasificadas de %d" % ( (vCountryTest != vCountryPredicted).sum(), len(vCountryTest) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (vCountryTest == vCountryPredicted).sum() ) / len(vCountryTest) ) )





