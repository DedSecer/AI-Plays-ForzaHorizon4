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
import signal
import sys

hold_key_continue = True
def signal_handler(signal, frame):
    print('Caught Ctrl+C / SIGINT signal')
    global hold_key_continue
    hold_key_continue = False
    sys.exit(0)

def hold_key(key, hold_time, release_time):
    while hold_key_continue:
        pyautogui.keyDown(key)
        pyautogui.keyUp(key)
        time.sleep(release_time)

def start_up(hold_time):
    start_time = time.time()
    while time.time() - start_time < hold_time:
        pyautogui.keyDown('w')
    pyautogui.keyUp('w')

past_frames = 20

time_checkpoint_a, time_checkpoint_b = 0, 0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_color = (255, 255, 255)
thickness = 1
frames_counter = 0

# 检查是否使用 GPU
print('Are we using GPU?:', tf.test.is_gpu_available())

loaded_model = tf.keras.models.load_model('Files/cv_convlstm_model{' + str(past_frames) + 'steps}.h5')

# 创建一个自定义的推理函数，显式设置 training=False
@tf.function
def predict_with_training_false(input_data):
    return loaded_model(input_data, training=False)

q = queue.Queue(maxsize=past_frames)

# 键码映射
key_map_pyautogui = ['a', 's', 'd']
def full_que():
    while q.qsize() < past_frames:
        # for fps calculation
        # acquire screen signal.
        # then, cropping, gray scaling, post-processing
        screen = grab_screen()
        resized_screen = bird_view_processing(screen)

        cv2.imshow('test', cv2.resize(resized_screen, (480, 270)))
        # manipulate the gray scale image matrix shape (width,height) -> (width,height,depth)
        test_inputs = np.expand_dims(resized_screen, axis=-1)

        # this is the past-n-frames queue for generating sequential data for LSTM network
        if q.qsize() == past_frames:
            q.get()
        q.put(test_inputs)

full_que()

sequential_input = np.asarray(list(q.queue))
# data feed to the model has to be inside a list, so I did this.
sequential_input = np.expand_dims(sequential_input, axis=0)

# 定义推理函数
def inference_task(input_data, result_holder):
    result_holder.append(predict_with_training_false(input_data))

# 创建一个列表来保存推理结果
prediction_result = []

# 创建推理线程
inference_thread = threading.Thread(target=inference_task, args=(sequential_input, prediction_result))
inference_thread.start()

# 设置超时时间（秒）
timeout = 1.0
inference_thread.join(timeout)

# 检查推理是否在超时时间内完成
if inference_thread.is_alive():
    print("Inference timed out")

    # 采取适当的措施，例如终止线程或重试推理
else:
    prediction = prediction_result[0]

# start_up(1)
threading.Thread(target=hold_key, args=('w', 0.04, 0.06)).start()
signal.signal(signal.SIGINT, signal_handler)
while True:
    frames_counter += 1

    # for fps calculation
    time_checkpoint_a = time.time()
    # acquire screen signal.
    # then, cropping, gray scaling, post-processing
    screen = grab_screen()
    resized_screen = bird_view_processing(screen)

    # manipulate the gray scale image matrix shape (width,height) -> (width,height,depth)
    test_inputs = np.expand_dims(resized_screen, axis=-1)

    # this is the past-n-frames queue for generating sequential data for LSTM network
    if q.qsize() == past_frames:
        q.get()
    q.put(test_inputs)

    sequential_input = np.asarray(list(q.queue))
    # data feed to the model has to be inside a list, so I did this.
    sequential_input = np.expand_dims(sequential_input, axis=0)

    # 创建一个列表来保存推理结果
    prediction_result = []

    # 创建推理线程
    inference_thread = threading.Thread(target=inference_task, args=(sequential_input, prediction_result))
    inference_thread.start()

    # 设置超时时间（秒）
    timeout = 2.0
    inference_thread.join(timeout)

    # 检查推理是否在超时时间内完成
    if inference_thread.is_alive():
        print("Inference timed out")
        q = queue.Queue(maxsize=past_frames)
        while q.qsize() < past_frames:
            q.put(test_inputs)
        # 采取适当的措施，例如终止线程或重试推理
        continue
    else:
        prediction = prediction_result[0]

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
                pyautogui.keyUp(action)
    else:
        for action in last_action:
            if action in key_map_pyautogui:
                pyautogui.keyDown(action)
                pyautogui.keyUp(action)
    last_action = predicted_action

    # 在图像上显示预测信息
    display_text = f"Prediction: {predicted_action}, Confidence: {confidence:.2f}"
    texted_screen = cv2.resize(resized_screen, (480, 270))
    cv2.putText(texted_screen, display_text, (10, 30), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # 显示图像
    cv2.imshow('test', texted_screen)

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
