import cv2
import os

def sift_matching_with_ransac(image_folder):
    # 建立儲存結果的資料夾
    result_folder = 'matching_results'
    os.makedirs(result_folder, exist_ok=True)

    # 讀取資料夾內的所有圖像檔案
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]

    # 建立SIFT物件
    sift = cv2.SIFT_create()

    # 對每一對圖像進行特徵匹配
    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            # 讀取圖像
            image1 = cv2.imread(os.path.join(image_folder, image_files[i]), cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(os.path.join(image_folder, image_files[j]), cv2.IMREAD_GRAYSCALE)

            # 偵測特徵點並計算描述子
            kp1, des1 = sift.detectAndCompute(image1, None)
            kp2, des2 = sift.detectAndCompute(image2, None)

            # 進行特徵匹配
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # 應用RANSAC外點剔除
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # 儲存匹配結果
            result_filename = os.path.join(result_folder, f"matching_result_{image_files[i]}_{image_files[j]}.jpg")
            matching_result = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(result_filename, matching_result)

# 執行SIFT特徵匹配並應用RANSAC
sift_matching_with_ransac('input')

