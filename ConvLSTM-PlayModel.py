import numpy as np
import cv2
import queue
import tensorflow as tf
import Utilities.onehot as oh
import Utilities.translate_result as tr
from Utilities.grabscreen import grab_screen
from Utilities.cv_img_processing import edge_processing, bird_view_processing, crop_screen
import time
import pyautogui
import threading

def hold_key(key, hold_time):
    while True:
        pyautogui.keyDown(key)
        time.sleep(hold_time)
        pyautogui.keyUp(key)
        time.sleep(1 - hold_time)

def start_up(hold_time):
    start_time = time.time()
    while time.time() - start_time < hold_time:
        pyautogui.keyDown('w')
    pyautogui.keyUp('w')

past_frames = 20

time_checkpoint_a, time_checkpoint_b = 0, 0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
thickness = 1
frames_counter = 0

# 检查是否使用 GPU
print('Are we using GPU?:', tf.test.is_gpu_available())

loaded_model = tf.keras.models.load_model('Files/cv_convlstm_model{' + str(past_frames) + 'steps}.h5')
q = queue.Queue(maxsize=past_frames)

# 键码映射
key_map_pyautogui = ['a', 's', 'd']
while q.qsize() < past_frames:
    frames_counter += 1
    # for fps calculation
    # acquire screen signal.
    # then, cropping, gray scaling, post-processing
    screen = grab_screen()
    resized_screen = bird_view_processing(screen)

    cv2.imshow('test', resized_screen)

    # manipulate the gray scale image matrix shape (width,height) -> (width,height,depth)
    test_inputs = np.expand_dims(resized_screen, axis=-1)

    # this is the past-n-frames queue for generating sequential data for LSTM network
    if q.qsize() == past_frames:
        q.get()
    q.put(test_inputs)

sequential_input = np.asarray(list(q.queue))
# data feed to the model has to be inside a list, so I did this.
sequential_input = np.expand_dims(sequential_input, axis=0)
prediction = loaded_model.predict(sequential_input)

start_up(1)
threading.Thread(target=hold_key, args=('w', 0.3)).start()
while True:
    frames_counter += 1

    # for fps calculation
    time_checkpoint_a = time.time()
    # acquire screen signal.
    # then, cropping, gray scaling, post-processing
    screen = grab_screen()
    resized_screen = bird_view_processing(screen)

    cv2.imshow('test', resized_screen)

    # manipulate the gray scale image matrix shape (width,height) -> (width,height,depth)
    test_inputs = np.expand_dims(resized_screen, axis=-1)

    # this is the past-n-frames queue for generating sequential data for LSTM network
    if q.qsize() == past_frames:
        q.get()
    q.put(test_inputs)

    sequential_input = np.asarray(list(q.queue))
    # data feed to the model has to be inside a list, so I did this.
    sequential_input = np.expand_dims(sequential_input, axis=0)
    prediction = loaded_model.predict(sequential_input)


    # acquire the argmax
    prediction_argmax = np.argmax(prediction[0])
    # generate a one_hot encoded result according to the predicted argmax value
    one_hot_result = [np.eye(prediction[0].shape[0])[prediction_argmax].astype(int).tolist()]
    # generate the prediction confidence
    confidence = max(prediction[0])
    # interpret the keyboard action
    predicted_action = tr.translate_wasd(oh.onehot_decode(one_hot_result)[0])

    last_action = ''
    # Apply the predicted action to the computer
    if confidence > 0.8:
        for action in predicted_action:
            if action in key_map_pyautogui:
                pyautogui.keyDown(action)
                # time.sleep(0.04)
                pyautogui.keyUp(action)
    else:
        for action in last_action:
            if action in key_map_pyautogui:
                pyautogui.keyDown(action)
                # time.sleep(0.04)
                pyautogui.keyUp(action)
    last_action = predicted_action

    # for fps calculation
    time_checkpoint_b = time.time()
    # insert the fps information to the screen
    fps = 1 / (time_checkpoint_b - time_checkpoint_a)

    # print some real-time information
    print('\rframe:', frames_counter,
          'pred:', predicted_action,
          'confidence:', confidence,
          'fps:', fps,
          end='')
    # some cv2 ritual
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
