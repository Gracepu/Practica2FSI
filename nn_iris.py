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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

# conjunto entrenamiento -> 70%
x_data_train = x_data[:105]
y_data_train = y_data[:105]

# conjunto validacion -> 15%
x_data_valid = x_data[105:128]
y_data_valid = y_data[105:128]

# conjunto test -> 15%
x_data_test = x_data[128:]
y_data_test = y_data[128:]

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

# guarda espacio para floats en modo matriz de n filas y 4/3 columnas
x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

# inicializa w y b con precision de float 32 bits
W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

# sigmoid(x*w + b)
h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
        # h = tf.matmul(x, W1) + b1  # Try this!
# hallamos los resultados/salida
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

# hallamos el error
loss = tf.reduce_sum(tf.square(y_ - y))

# hacemos descenso por el gradiente con el error hallado para llegar al minimo
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

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
error = 20
while error > 1.5 and epoch < 500:
    # bucle de 0 a num_filas_x_data/20
    for jj in xrange(len(x_data) / batch_size):
        # cogemos los conjuntos de entrenamiento de 20 en 20
        batch_xs = x_data_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        # generamos una sesion para actualizar el gradiente con nuevas x e y_
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    epoch = epoch + 1
    # mostramos el error del entrenamiento con el conjunto de validacion
    error = sess.run(loss, feed_dict={x: x_data_valid, y_: y_data_valid})
    print "Epoch #:", epoch, "Error: ", error
    print "----------------------------------------------------------------------------------"

# Comprobamos finalmente si funciona con los datos de test
result = sess.run(y, feed_dict={x: x_data_test})
    # creamos una tupla de dos elementos: b y r
valError = 0
for b, r in zip(y_data_test, result):
        # comprobamos si alguno de los valores no es como deberia
    if np.argmax(b) != np.argmax(r):
       valError += 1
    print b, "-->", r
print "----------------------------------------------------------------------------------"
print "Numero valores erroneos: ", valError