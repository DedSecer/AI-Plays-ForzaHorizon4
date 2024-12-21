import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import ConvLSTM2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, TimeDistributed
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

past_frames = 20
rows = 100
cols = 200
channels = 1

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print('Are we using GPU?:', tf.test.is_gpu_available())

balanced_data = np.load('Files/preprocessed_data_lstm.npy', allow_pickle=True)
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
    np.asarray([item[0] for item in balanced_data]), np.array([item[1] for item in balanced_data]),
    test_size=0.2,
    random_state=233)

# Step2 : Define model

model = Sequential()
model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                     activation='tanh',
                     data_format="channels_last",
                     recurrent_dropout=0.2,
                     return_sequences=True,
                     input_shape=(train_inputs.shape[1],
                                  train_inputs.shape[2],
                                  train_inputs.shape[3],
                                  train_inputs.shape[4])))
model.add(BatchNormalization())
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last')))

model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                     activation='tanh',
                     data_format="channels_last",
                     recurrent_dropout=0.2,
                     return_sequences=True))
model.add(BatchNormalization())
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last')))

model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                     activation='tanh',
                     data_format="channels_last",
                     recurrent_dropout=0.2,
                     return_sequences=False))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(16, activation="softmax"))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
plot_model(model, to_file='Files/convlstm_model_structure_plot.png', show_shapes=True, show_layer_names=True)

# Step3 : Train model

print('input  shape:', train_inputs.shape)
print('output shape:', train_outputs.shape)

history = model.fit(train_inputs, train_outputs, epochs=50, batch_size=50,
          validation_data=(test_inputs, test_outputs))

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])#acc最新版keras已经无法使用
plt.plot(history.history['val_accuracy'])#val_acc最新版keras已经无法使用
plt.title('Model accuracy')#图名
plt.ylabel('Accuracy')#纵坐标名
plt.xlabel('Epoch')#横坐标名
plt.legend(['Train', 'Test'], loc='upper left')#角标及其位置
plt.savefig('Files/lstm-accuracy.png')
# plt.show()

plt.close()
#如果不想画在同一副图上，可关闭再打开
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#图像保存方法
plt.savefig('Files/lstm-loss.png')

# Step4 : Evaluate model

model.save('Files/cv_convlstm_model{' + str(past_frames) + 'steps}.h5')
print('Model saved.')
loaded_model = tf.keras.models.load_model('Files/cv_convlstm_model{' + str(past_frames) + 'steps}.h5')
print('Model loaded.')
test_loss, test_acc = loaded_model.evaluate(np.asarray(test_inputs), np.array(test_outputs))
print('Val. accuracy:', test_acc)
