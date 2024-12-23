import cv2
from Utilities.grabscreen import grab_screen
from Utilities.cv_img_processing import edge_processing, bird_view_processing, crop_screen
import numpy as np
from advancedlinedectsys import process_image


# while True:
#     # grab the screen image
#     screen = grab_screen()

#     # cropping the image
#     cropped_screen = crop_screen(screen)

#     resized_image = edge_processing(screen)

#     cv2.imshow('original', screen)
#     cv2.imshow('processed', resized_image)

#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

while True:
    # grab the screen image
    screen = grab_screen()  # 根据需要调整区域

    # process the image using advancedlinedectsys's process_image function
    processed_image = process_image(screen)
    # processed_image = bird_view_processing(screen)


    # display the original and processed images
    # cv2.imshow('Original', screen)
    cv2.imshow('Processed', processed_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break