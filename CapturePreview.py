import cv2
from Utilities.grabscreen import grab_screen
from Utilities.cv_crop_processing import crop_screen
from Utilities.cv_edge_processing import edge_processing
import numpy as np

while True:
    # grab the screen image
    screen = grab_screen()

    # cropping the image
    cropped_screen = crop_screen(screen)

    resized_image = edge_processing(screen)

    cv2.imshow('original', screen)
    cv2.imshow('processed', resized_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
