from PIL import ImageFont

font_path = "fonts/4.ttf"
font_size = 100

font = ImageFont.truetype(font_path, font_size)
# font_height = font.getsize("A")[1] 

import cv2
import numpy as np
from PIL import Image, ImageDraw


image = np.ones((640, 640, 3), dtype=np.uint8)
color = (255, 255, 255)
text = ["0", "1", "2", "3","4","5","6","7","8","9", "A", "B", "C", "D", "E", "F"]


image_pil = Image.fromarray(image)
draw = ImageDraw.Draw(image_pil)
draw.text((300, 300), text[0], font=font, fill=color)

image_with_text = np.array(image_pil)  # Convert PIL image back to OpenCV image

image_with_text = cv2.bitwise_not(image_with_text)

import time
cv2.imwrite("train_image/" + str(time.time()) + ".jpg", image_with_text) 

for angle in range(1, 45, 1):
    h, w = image_with_text.shape[:2]
    center = (w//2, h//2)
    print(image_with_text)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_with_text, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # time.sleep(1)
    cv2.imwrite("train_image/" + str(time.time()) + ".jpg", rotated)

for angle in range(-45, -1, 1):
    h, w = image_with_text.shape[:2]
    center = (w//2, h//2)
    print(image_with_text)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_with_text, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # time.sleep(1)
    cv2.imwrite("train_image/" + str(time.time()) + ".jpg", rotated)




# cv2.imshow("Image with Text", image_with_text)
# cv2.waitKey(0)
# cv2.destroyAllWindows()