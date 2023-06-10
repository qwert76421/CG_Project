import cv2
import os

def denoise_image(image):
    # 轉換為灰度圖像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.convertScaleAbs(gray_image)

    # 非局部平均去噪
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    return denoised_image

# 資料夾路徑
folder_path = 'input/'

# 檢查輸出資料夾是否存在，若不存在則建立資料夾
output_folder_path = 'denoised_images/'
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 讀取資料夾中的圖像
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        # 讀取圖像
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # 去噪
        denoised_image = denoise_image(image)

        # 儲存去噪後的圖像
        output_image_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_image_path, denoised_image)

        print(f'Denoised image saved: {output_image_path}')

