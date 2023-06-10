import cv2
import os
import numpy as np

def warp_triangle(src_img, dst_img, src_pts, dst_pts):
    # 計算仿射變換矩陣
    warp_matrix = cv2.getAffineTransform(src_pts, dst_pts)

    # 對三角形區域應用仿射變換
    warped_triangle = cv2.warpAffine(src_img, warp_matrix, (dst_img.shape[1], dst_img.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 創建三角形區域的遮罩
    mask = np.zeros_like(dst_img)
    cv2.fillConvexPoly(mask, np.int32(dst_pts), (255, 255, 255))

    # 將變形後的三角形區域應用到目標圖像上
    dst_img = cv2.bitwise_and(dst_img, cv2.bitwise_not(mask))
    dst_img = cv2.bitwise_or(warped_triangle, dst_img)

    return dst_img

# 設定目標圖像路徑和參考圖像資料夾路徑
target_image_path = 'target_gray.jpg'
reference_folder_path = 'denoised_images/'
epsilon = 5.0

# 讀取目標圖像
target_image = cv2.imread(target_image_path, 0)

# 建立SIFT物件
sift = cv2.SIFT_create()

# 儲存註冊後的圖像的資料夾路徑
registered_folder_path = 'registered_images/'

# 檢查註冊後的圖像資料夾是否存在，若不存在則建立資料夾
if not os.path.exists(registered_folder_path):
    os.makedirs(registered_folder_path)

# 搜尋參考圖像資料夾內的所有圖像
for filename in os.listdir(reference_folder_path):
    if filename.endswith('.jpg'):
        # 讀取參考圖像
        reference_image_path = os.path.join(reference_folder_path, filename)
        reference_image = cv2.imread(reference_image_path, 0)

        # 執行SIFT特徵提取和匹配
        kp_target, des_target = sift.detectAndCompute(target_image, None)
        kp_reference, des_reference = sift.detectAndCompute(reference_image, None)

        # 使用特徵匹配算法（例如基於最近鄰居的匹配）進行特徵匹配
        matcher = cv2.BFMatcher()
        matches = matcher.match(des_target, des_reference)

        # 提取匹配的特徵點位置
        match_points_target = [kp_target[match.queryIdx].pt for match in matches]
        match_points_reference = [kp_reference[match.trainIdx].pt for match in matches]

        # 使用RANSAC算法估計投影矩陣
        ransac_threshold = 4.0  # RANSAC閾值，用於判斷是否為內點
        match_points_reference = np.array(match_points_reference)
        match_points_target = np.array(match_points_target)
        homography, inliers = cv2.findHomography(match_points_reference, match_points_target, cv2.RANSAC, ransac_threshold)

        # 將參考圖像註冊到目標圖像
        registered_image = cv2.warpPerspective(reference_image, homography, (target_image.shape[1], target_image.shape[0]))

        # 保存註冊後的圖像
        registered_image_path = os.path.join(registered_folder_path, filename)
        cv2.imwrite(registered_image_path, registered_image)

        # 計算再投影誤差
        reprojection_errors = []
        for i in range(len(match_points_target)):
            # 將目標特徵點透過投影矩陣的逆映射到參考圖像上
            target_point = np.array([match_points_target[i][0], match_points_target[i][1], 1])
            reference_point = np.dot(np.linalg.inv(homography), target_point)
            reference_point /= reference_point[2]  # 正規化

            # 計算再投影誤差
            error = np.linalg.norm(match_points_reference[i] - reference_point[:2])
            reprojection_errors.append(error)
        #將再投影誤差存儲到註冊後的圖像
        #registered_image = cv2.putText(registered_image, f"Reprojection Error: {mean_reprojection_error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 計算平均再投影誤差
        mean_reprojection_error = np.mean(reprojection_errors)

        # 判斷是否將圖像包含在參考影像集中
        reprojection_threshold = epsilon  # 預定的再投影誤差閾值

        # 保存註冊後的圖像
        registered_image_path = os.path.join(registered_folder_path, filename)
        cv2.imwrite(registered_image_path, registered_image)

        if mean_reprojection_error < epsilon:
            # 執行全域對齊，將整個參考圖像根據投影矩陣註冊到目標圖像
            registered_image = cv2.warpPerspective(reference_image, homography, (target_image.shape[1], target_image.shape[0]))
        else:
            # 建立Subdiv2D物件進行Delaunay三角剖分
            target_subdiv = cv2.Subdiv2D((0, 0, target_image.shape[1]-1, target_image.shape[0]-1))
            reference_subdiv = cv2.Subdiv2D((0, 0, target_image.shape[1]-1, target_image.shape[0]-1))
            
            # 檢查點的坐標值是否在圖像範圍內
            valid_points = []
            for point in match_points_target:
                x, y = point[0], point[1]
                if 0 <= x < target_image.shape[1] and 0 <= y < target_image.shape[0]:
                    valid_points.append(point)

            # 將有效的點傳入 Subdiv2D 的 insert 函式
            for point in valid_points:
                target_subdiv.insert(point)
                reference_subdiv.insert(point)

            # 獲取Delaunay三角剖分的三角形
            target_tri = target_subdiv.getTriangleList()
            reference_tri = reference_subdiv.getTriangleList()

            # 對每個三角形進行變形
            registered_image = np.zeros_like(target_image)
            for i in range(len(target_tri)):
                # 獲取目標圖像中的三角形頂點坐標
                target_pts = np.float32([target_tri[i][0:2], target_tri[i][2:4], target_tri[i][4:6]])

                # 獲取參考圖像中的三角形頂點坐標
                reference_pts = np.float32([reference_tri[i][0:2], reference_tri[i][2:4], reference_tri[i][4:6]])

                # 建立仿射變換矩陣
                affine_transform = cv2.getAffineTransform(reference_pts, target_pts)

                # 將參考圖像的三角形區域根據仿射變換矩陣應用到目標圖像上
                dst_pts = cv2.transform(reference_pts.reshape(1, -1, 2), affine_transform).reshape(-1, 2)
                registered_image = warp_triangle(reference_image, registered_image, reference_pts, dst_pts)

        # 保存註冊後的圖像
        registered_image_path = os.path.join(registered_folder_path, filename)
        cv2.imwrite(registered_image_path, registered_image)

        print(f'Registered image saved: {registered_image_path}')
        print(registered_image.min())  # 最小值
        print(registered_image.max())  # 最大值


