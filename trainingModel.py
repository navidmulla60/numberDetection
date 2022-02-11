
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#normalize the data

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()

#now create a model
model = tf.keras.models.Sequential()

#make the data to 1 dimentions 
model.add(tf.keras.layers.Flatten())

#add a hidden layer (below we adding a dense layer that means a fully connected layer)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#adding an output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


#compile the model we defined above

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, epochs=3)

#Evaluate the model by feeding test data

val_loss, val_acc = model.evaluate(x_test, y_test)

print("Loss: ", val_loss)
print("Accuracy: ",val_acc)


#saving the model
model.save('epic_num_reader.model')

