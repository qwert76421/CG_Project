import cv2

color_image = cv2.imread('newtarget_color.jpg')

gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('newtarget_gray.jpg', gray_image)
