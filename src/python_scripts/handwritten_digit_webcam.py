import nn
import numpy as np
import cv2

cv2.namedWindow("display", cv2.WINDOW_KEEPRATIO)

vid = cv2.VideoCapture(0)
network = nn.load_network_from_file("digit_99.wanb")

print(network)

kernel = np.ones((5, 5), np.uint8)

while True:
    ret, image = vid.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    th, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ROI = binary[y:y + h, x:x + w]

        area = ROI.shape[0] * ROI.shape[1];         

        if area < 200:
            continue

        ROI = cv2.copyMakeBorder(ROI, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        resized = cv2.resize(ROI, (28, 28), interpolation=cv2.INTER_LINEAR)
        
        # converting [0 - 255] color values to [0 - 1] values
        pixel_array = [pixel / 255.0 for pixel in np.array(resized).flatten()]
        nn.feedforward(network, pixel_array)
        
        predicted_digit = -1
        prob = 0.0;

        for n in range(0, 10):
            if prob < network.layers[network.layer_count-1].neurons[n].activation: 
                prob = network.layers[network.layer_count-1].neurons[n].activation
                predicted_digit = n

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(image, str(predicted_digit), (x+10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('display', image)

    c = cv2.waitKey(1)
    if c == 27:
        break

vid.release()

cv2.destroyWindow('display')
