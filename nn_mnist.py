import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# conjunto de entrenamiento
train_x, train_y = train_set
#etiquetas del conjunto de entrenamiento
train_y = one_hot(train_y.astype(int), 10) #pasamos a one_hot vector -> [0,0,0,0,1,0,0,0,0,0]
# conjunto de validacion
valid_x, valid_y = valid_set
# etiquetas del conjunto de validacion
valid_y = one_hot(valid_y.astype(int),10)
# conjunto de test
test_x, test_y = test_set
# etiquetas del conjunto de test
test_y = one_hot(test_y.astype(int), 10)



# ---------------- Visualizing some element of the MNIST dataset --------------

#descomentar estas lineas si se desea la tabla de aprendizaje
#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]


# TODO: the neural net!!
x = tf.placeholder("float", [None, 784]) # imagenes
y_ = tf.placeholder("float", [None, 10]) #etiquetas => nums de 0 a 9

# dos capas de neuronas pero dividiendo 28/28
W3 = tf.Variable(np.float32(np.random.rand(784,28))*0.1)
b3 = tf.Variable(np.float32(np.random.rand(28))*0.1)

W4 = tf.Variable(np.float32(np.random.rand(28,10))*0.1)
b4 = tf.Variable(np.float32(np.random.rand(10))*0.1)
h3 = tf.nn.sigmoid(tf.matmul(x,W3) + b3)

y = tf.nn.softmax(tf.matmul(h3, W4) + b4)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# iniciar las variables para comenzar la sesion
init = tf.initialize_all_variables()

# declaramos la sesion y la comenzamos
sess = tf.Session()
sess.run(init)


print "----------------------"
print "   Start training...  "
print "----------------------"

# tamano de muestras que vamos a ir tratando cada vez
batch_size = 20

# entrenamiento de la neurona
epoch = 0
errorA = 20000
errorB = 20000
while errorA <= errorB:
    # bucle de 0 a num_filas_x_data/20
    for jj in xrange(len(train_x) / batch_size):
        # cogemos los conjuntos de entrenamiento de 20 en 20
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        # generamos una sesion para actualizar el gradiente con nuevas x e y_
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    epoch = epoch + 1
    # mostramos el error del entrenamiento con el conjunto de validacion
    errorB = errorA
    errorA = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    print "Epoch #:", epoch, "Error: ", errorA
    print "----------------------------------------------------------------------------------"

# Comprobamos finalmente si funciona con los datos de test
result = sess.run(y, feed_dict={x: test_x})
    # creamos una tupla de dos elementos: b y r
valError = 0
for b, r in zip(test_y, result):
        # comprobamos si alguno de los valores no es como deberia
    if np.argmax(b) != np.argmax(r):
       valError += 1
    #print b, "-->", r
print "----------------------------------------------------------------------------------"
print "Numero valores erroneos: ", valError