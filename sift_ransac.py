import cv2
import os

def sift_matching_with_ransac(query_image, reference_folder):
    # 載入目標圖像
    query_gray = cv2.imread(query_image, cv2.IMREAD_GRAYSCALE)

    # 建立儲存結果的資料夾
    result_folder = 'matching_results'
    os.makedirs(result_folder, exist_ok=True)

    # 進行SIFT特徵匹配並儲存結果
    for filename in os.listdir(reference_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            reference_image = cv2.imread(os.path.join(reference_folder, filename), cv2.IMREAD_GRAYSCALE)

            # 建立SIFT物件
            sift = cv2.SIFT_create()

            # 偵測特徵點並計算描述子
            kp_query, des_query = sift.detectAndCompute(query_gray, None)
            kp_reference, des_reference = sift.detectAndCompute(reference_image, None)

            # 進行特徵匹配
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des_query, des_reference, k=2)

            # 應用RANSAC外點剔除
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # 儲存匹配結果
            result_filename = os.path.join(result_folder, f"matching_result_{filename}")
            matching_result = cv2.drawMatches(query_gray, kp_query, reference_image, kp_reference, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(result_filename, matching_result)

# 執行SIFT特徵匹配並應用RANSAC
sift_matching_with_ransac('target_gray.jpg', 'statue of liberty')

