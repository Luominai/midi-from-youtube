import cv2 as cv

image = cv.imread('frames/birdbrain_crop.jpg', cv.IMREAD_GRAYSCALE)
# image = cv.imread('frames/perspective.jpg')
# image = cv.imread('frames/black_bg.jpg')
# image = cv.imread('frames/gogh.jpg')
if image is None:
    print('Error opening image!')

(height, width, *rest) = image.shape # type: ignore
# scale = 720 / height
scale = 1.5 * 1280 / width
image = cv.resize(image, None, fx=scale, fy=scale) # type: ignore
times = 1
size = 5

for i in range(times):
    image = cv.blur(image, (size, size))

times = 3
size = 3
for i in range(times):
    image = cv.blur(image, (size, size))

# image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
threshold, output = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
image = cv.Canny(image, threshold / 2, threshold, None, 3)
cv.imshow("a", image)
cv.waitKey(-1)