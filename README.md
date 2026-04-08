import numpy as np
import matplotlib.pyplot as plt

def convolve2d(image, kernel):
    kh, kw = kernel.shape
    h, w = image.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), "edge")
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)
    return output

h, w = 200, 300
img = np.zeros((h, w), dtype=np.float64)
img[30:80, 20:100] = 100
img[110:170, 150:260] = 255

noise = np.random.normal(0, 15, img.shape)
img_noisy = np.where(img > 0, np.clip(img + noise, 0, 255), 0)

gauss_kernel = np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]
]) / 273.0
img_smoothed = convolve2d(img_noisy, gauss_kernel)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
gx = convolve2d(img_smoothed, sobel_x)
gy = convolve2d(img_smoothed, sobel_y)

mag = np.sqrt(gx**2 + gy**2)
theta = np.arctan2(gy, gx) * 180 / np.pi
theta[theta < 0] += 180

nms = np.zeros_like(mag)
for i in range(1, h - 1):
    for j in range(1, w - 1):
        angle = theta[i, j]
        q, r = 255, 255
        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
            q, r = mag[i, j+1], mag[i, j-1]
        elif (22.5 <= angle < 67.5):
            q, r = mag[i+1, j-1], mag[i-1, j+1]
        elif (67.5 <= angle < 112.5):
            q, r = mag[i+1, j], mag[i-1, j]
        elif (112.5 <= angle < 157.5):
            q, r = mag[i-1, j-1], mag[i+1, j+1]
        if mag[i, j] >= q and mag[i, j] >= r:
            nms[i, j] = mag[i, j]

low_th, high_th = 20, 50
res_thresh = np.zeros_like(nms)
res_thresh[nms >= high_th] = 255
res_thresh[(nms >= low_th) & (nms < high_th)] = 100

final = np.copy(res_thresh)
for i in range(1, h - 1):
    for j in range(1, w - 1):
        if res_thresh[i, j] == 100:
            if 255 in res_thresh[i-1:i+2, j-1:j+2]:
                final[i, j] = 255
            else:
                final[i, j] = 0

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
steps = [
    (img, "Оригинал", 'gray'),
    (img_noisy, "Гаус шум", 'gray'),
    (gx, "Sobel X", 'gray'),
    (gy, "Sobel Y", 'gray'),
    (mag, "Мощность", 'gray'),
    (theta, "Направление", 'hsv'),
    (res_thresh, "Двойной порог", 'gray'),
    (final, "Финал (Canny)", 'gray')
]

for i, (data, title, cmap) in enumerate(steps):
    ax = axes[i // 4, i % 4]
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    if title == "Направление":
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
