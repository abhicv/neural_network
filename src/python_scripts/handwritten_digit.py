import nn
# import numpy as np
# import cv2
import sys

# def print_image(pixel_array):
#     for row in range(0, 28):
#         for col in range(0, 28):
#             print("{p:.1f} ".format(p = pixel_array[row * 28 + col]), end="")
#         print("")
#     print("")

# cv2.namedWindow("display", cv2.WINDOW_KEEPRATIO)

network = nn.load_network_from_file("digit_99.wanb");
print(network)

# image = cv2.imread("sample_6.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# th, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
# th, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY_INV)
# kernel = np.ones((5, 5), np.uint8)
# dilated = cv2.dilate(binary, kernel, iterations=1)
# contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     ROI = binary[y:y + h, x:x + w]
#     ROI = cv2.copyMakeBorder(ROI, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#     resized = cv2.resize(ROI, (28, 28), interpolation=cv2.INTER_AREA)

#     converting 0 - 255 pixel values to 0 - 1 range
#     pixel_array = [pixel / 255.0 for pixel in np.array(resized).flatten()]
#     print_image(pixel_array)

#     nn.feedforward(network, pixel_array)
    
#     predicted_digit = -1
#     prob = 0.0;
#     for n in range(0, 10):
#         if prob < network.layers[network.layer_count-1].neurons[n].activation: 
#             prob = network.layers[network.layer_count-1].neurons[n].activation
#             predicted_digit = n
        
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     image = cv2.putText(image, str(predicted_digit), (x+10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1, cv2.LINE_AA)
    
# cv2.imshow('display', image)
# cv2.waitKey(0)

# cv2.destroyWindow('display')
