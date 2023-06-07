import cv2
import glob

# 取得影像列表
image_files = glob.glob('input_images/*.jpg')  # 設定符合您的影像路徑和格式

# 迴圈處理每張影像
for image_file in image_files:
    # 讀取彩色影像
    color_image = cv2.imread(image_file)

    # 將彩色影像轉換為灰度影像
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 取得影像檔案名稱
    filename = image_file.split('/')[-1]  # 取得最後一個斜線後的部分作為檔案名稱

    # 儲存灰度影像
    output_file = 'output_images/gray_' + filename  # 設定灰度影像的儲存路徑和檔名
    cv2.imwrite(output_file, gray_image)

