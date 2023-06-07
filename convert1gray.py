import cv2

color_image = cv2.imread('target.jpg')

gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('target_image.jpg', gray_image)
