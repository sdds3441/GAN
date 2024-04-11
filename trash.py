import cv2

# 이미지를 그레이스케일로 읽어들입니다.
image = cv2.imread('images/HyoChanKim_IDPhoto.jpg', cv2.IMREAD_GRAYSCALE)
# Canny 엣지 검출을 수행합니다.
edges = cv2.Canny(image, 100, 200)

# 엣지가 포함된 이미지를 출력합니다.
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.imwrite('images/gray.jpg',image)
cv2.destroyAllWindows()