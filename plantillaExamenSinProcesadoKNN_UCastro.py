# This is a sample Python script.
# Press Mayus+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from datetime import datetime
import getopt
import sys
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

k=''
d=''
p='./'
f="seeds_datasetForTheExam_SubGrupo3.csv"
oFile=""
m="uniform"
r=0
classifier="Class" #Ultima columna a medir

def datetime_to_epoch(d):
    return datetime.datetime(d).strftime('%s')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:k:K:d:D:p:f:h',['output=','k=','K=','d=','D=','path=','iFile','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg
        elif opt == '-k':
            k = arg
        elif opt == '-K':
            K = arg
        elif opt ==  '-d':
            d = arg
        elif opt ==  '-D':
            D = arg
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-h','--help'):
            print(' -o outputFile \n -k numberOfItems \n -d distanceParameter \n -p inputFilePath \n -f inputFileName \n ')
            exit(1)

    if p == './':
        iFile=p+str(f)
    else:
        iFile = p+"/" + str(f)
    # astype('unicode') does not work as expected

    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)

    #comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera linea por defecto la considerara como la que almacena los nombres de los atributos
    # comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado

    #print(ml_dataset.head(5))


   # ml_dataset = ml_dataset[(#Nombres de todas las cabeceras de las columnas de las tablas, ejemplo-->)'Altura','Peso','Sexo','ColorOjo','Class']
   #Se introduce todos los nombres de las cabeceras de las columnas
    colums = list(ml_dataset.columns)
    ml_dataset=ml_dataset[colums]

    # Q tipo de dato tengo


    categorical_features = []
    
    #numerical_features = [Nombres de la cabeceras de las columnas de la tabla separadas por '','']
    #O haciendo esto, recoge todas menos la ultima columna definida al inicio
    numerical_features =list(ml_dataset.columns)
    numerical_features.remove(classifier)
    text_features = []

    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')

    # Los valores posibles de la clase a predecir.
    # Puede ser de 3 clases
    target_map = {'3': 0, '2': 1, '1':2} # q valor le asignas a cada atributo
    ml_dataset['__target__'] = ml_dataset[classifier].map(str).map(target_map)
    del ml_dataset[classifier]

    # Borra las filas en las que la clase a predecir no aparece.
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))

    # Se crean las particiones de Train/Test
    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])#Elegimos la muestra para entrenar
    print(train.head(5))                                                                                        #El 20% para test, indice aleatorio
    print(train['__target__'].value_counts())
    print(test['__target__'].value_counts())

    # Lista con los atributos que cuando faltan en una instancia hagan que se tenga que borrar
    drop_rows_when_missing = []
    
    # Lista con los atributos que cuando faltan en una instancia se tenga que corregir haciendo la media, mediana, etc. del resto
    #impute_when_missing = [{'feature': 'num_var45_ult1', 'impute_with': 'MEAN'},
    #                      {'feature': 'num_op_var39_ult1', 'impute_with': 'MEAN'},
    #                       {'feature': 'num_op_var40_comer_ult3', 'impute_with': 'MEAN'},
    #                       {'feature': 'num_var45_ult3', 'impute_with': 'MEAN'},
    #                       {'feature': 'num_aport_var17_ult1', 'impute_with': 'MEAN'}]


    valoresdecolumna= list(ml_dataset.columns)
    valoresdecolumna.remove('__target__')
    impute_when_missing = []              
    for i in range(0, len(valoresdecolumna)):
        impute_when_missing.append({'feature': valoresdecolumna[i],'impute_with' : 'MEAN'})
        
    
    
    
    # Valores del conjunto Train
    trainX = train.drop('__target__', axis=1)
    #trainX = train['__target__']

    # Valores del conjunto Test
    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    # Etiquetas del conjunto Train
    trainY = np.array(train['__target__'])
    # Etiquetas del conjunto Test
    testY = np.array(test['__target__'])

    # Más de una clase y haya mucha diferencia entre unos y otras
    #sampling_strategy = {0: 10, 1: 10, 2: 10}
    #undersample = RandomUnderSampler(sampling_strategy=0.5) #la mayoria va a estar representada el doble de veces , se puede utilizar 0.5 o sampling_strategy

    #Se reemplazan los conjuntos Train/Test con unos conjuntos a los que se les ha realizado undersampling
    #trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    #testXUnder,testYUnder = undersample.fit_resample(testX, testY)



    for valorD in range(int(d), int(D) +1):
    	for valorK in range(int(k), int(K) +1):
            # Solo hacer si los vecinos son impares
            if valorK % 2 != 0:

# Inicializa el KNN 
                clf = KNeighborsClassifier(n_neighbors=valorK, weights='uniform', algorithm='auto', leaf_size=30, p=valorD)

		# Dice si el peso es uniforme o balanceado

                clf.class_weight = "balanced"

				#Instruir a la regresion lineal que aprenda de los datos trainX y trainY

                clf.fit(trainX, trainY)


				# Build up our result dataset

				# The model is now trained, we can apply it to our test set:

                predictions = clf.predict(testX)
                probas = clf.predict_proba(testX)

                predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
                cols = [
                u'probability_of_value_%s' % label
                for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
                        ]
                probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

                    # Build scored dataset
                results_test = testX.join(predictions, how='left')
                results_test = results_test.join(probabilities, how='left')
                results_test = results_test.join(test['__target__'], how='left')
                results_test = results_test.rename(columns= {'__target__': 'TARGET'})

                i=0
                for real,pred in zip(testY,predictions):
                    print(real,pred)
                    i+=1
                    if i>5:
                        break

                print(f1_score(testY, predictions, average=None))
                print(classification_report(testY,predictions))
                print(confusion_matrix(testY, predictions, labels=[1,0]))
print("bukatu da")

#--PARA VER LOS ENTORNOS--
#anaconda-navigator
#--ELEGIR ENTORNO--
#conda activate sad 
#--LIBRERIAS--
#pip3 install pickle5
#pip3 install imblearn
#pip3 install pandas
#pip3 install scikit-learn
#pip3 install datetime--solo si da error 
#--EJECUTAR RECETAS--
#python plantillaExamenSinProcesadoKNN.py -k 1 -K 3 -d 1 -D 3 -o prueba.csv
#python PlantillaArbolesSinPreprocesar.py -k 1 -d 1 -D 1 -m 3 -M 6 -o fsinPro.csv

#RESULTADOS SIN PREPROCESO
#Para este experimento se ha usado el archivo knn_no_pre.py (python knn_no_pre.py -k 3 -d 1 -m uniform -o fnopre.csv) en la versión de Python 2.7. Se ha utilizado la semilla 42 para la elección de las instancias para el train.

#Los resultados obtenidos en primer lugar, sin preprocesado (con un k = 3, p = 1 y un peso “uniform”), nos dan un f1_score micro, macro y weighted de 0.95, 0.96 y 0.96 respectivamente. Podemos confirmar que el modelo da unos resultados muy altos para no haber preprocesado los datos. A continuación veremos si los resultados han mejorado aplicando el reescalado y el undersampling.
RESULTADOS CON PREPROCESO
#Para este experimento se han usado los archivos knn_ex.sh (./knn_ex.sh 3 5 1) y knn_pre.py en la versión de Python 2.7. Se ha utilizado la semilla 42 para la elección de las instancias para el train.

#Los resultados obtenidos ahora han mejorado en 0.2 en todos los f_score con k = 3, p = 1 y un peso “uniform”. Para k = 5, los resultados son los mismos que los obtenidos sin preproceso.

#Una vez visto esto, podemos confirmar que la elección de los preprocesos ha sido correcta por la mejora obtenida con k = 3. Es normal ver la reducción del f_score con k = 5 ya que existen más vecinos que votan en la decisión de la clase de una dicha instancia. Es por ello, que al obtener los mismos valores con tres vecinos que con cinco, podemos confirmar que la elección de los preprocesos ha sido correcta.
