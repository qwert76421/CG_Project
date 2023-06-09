import cv2
import os
import numpy as np

similarity_threshold = 0.7
min_match_count = 10

def compute_affine_transform(image1, image2):
    # 自行實現計算兩張圖像之間的仿射變換矩陣的函式
    # 這裡只是一個示例，需要根據具體情況進行實現
    return np.eye(2, 3)

def generate_binary_mask(image, M):
    # 創建與圖像大小相同的空白遮罩
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 將圖像的所有像素進行仿射變換，應用到遮罩中
    mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return mask

def sift_matching_and_composite(image_folder):
    # 建立儲存結果的資料夾
    result_folder = 'results'
    os.makedirs(result_folder, exist_ok=True)

    # 讀取資料夾內的所有圖像檔案
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]

    # 建立SIFT物件
    sift = cv2.SIFT_create()

    # 讀取目標圖像
    target_image = cv2.imread(os.path.join(image_folder, image_files[0]))

    # 對每一張參考圖像進行配准
    for i in range(1, len(image_files)):
        # 讀取參考圖像
        reference_image = cv2.imread(os.path.join(image_folder, image_files[i]))

        # 偵測特徵點並計算描述子
        kp1, des1 = sift.detectAndCompute(reference_image, None)
        kp2, des2 = sift.detectAndCompute(target_image, None)

        # 進行特徵匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # 應用RANSAC外點剔除
        good_matches = []
        for m, n in matches:
            if m.distance < similarity_threshold * n.distance:
                good_matches.append(m)

        # 判斷特徵點數量是否足夠進行尺寸匹配
        if len(good_matches) >= min_match_count:
            # 計算仿射變換矩陣
            M = compute_affine_transform(reference_image, target_image)

            # 生成二進制遮罩
            mask = generate_binary_mask(reference_image, M)

            # 進行後續的內部圖像分解等操作

            # 將配准結果和遮罩應用到目標圖像中

    # 儲存最終結果
    result_filename = os.path.join(result_folder, "merged_image.jpg")
    cv2.imwrite(result_filename, target_image)

# 執行SIFT特徵匹配並進行圖像合成
sift_matching_and_composite('input')
