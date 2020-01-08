import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def image_crop(img, height, width, start=(0, 0)):
    return img[start[0]:start[1] + height, start[1]:start[0] + width]


def _color_shift(v, shift):
    if shift >= 0:
        r_lim = 255 - shift
        v[v > r_lim] = 255
        v[v <= r_lim] = (v[v <= r_lim] + shift).astype(v.dtype)
    else:
        r_lim = abs(shift)
        v[v < r_lim] = 0
        v[v >= r_lim] = (v[v >= r_lim] - shift).astype(v.dtype)
    return v


def color_shift(img, r_shift=0, g_shift=0, b_shift=0):
    if -255 <= r_shift <= 255 or -255 <= g_shift <= 255 and -255 <= b_shift <= 255:
        b, g, r = cv2.split(img)
        b = _color_shift(b, b_shift)
        g = _color_shift(g, g_shift)
        r = _color_shift(r, r_shift)
        return cv2.merge((b, g, r))


def rotation(img, center, angle, scale=1, size=(500, 500)):
    m = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, m, size)


def perspective_transform(img, src_points, dst_points):
    m = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]))


def random_warp(img, row, col):
    height, width, channels = img.shape
    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    return perspective_transform(img, pts1, pts2)


img_ori = cv2.imread("lenna.jpg", 1)
# img_crop1 = image_crop(img_ori, 100, 200)
# img_crop2 = image_crop(img_ori, 400, 200)
# img_crop3 = image_crop(img_ori, 600, 200)
# plt.subplot(141)
# plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
# plt.subplot(142)
# plt.imshow(cv2.cvtColor(img_crop1, cv2.COLOR_BGR2RGB))
# plt.subplot(143)
# plt.imshow(cv2.cvtColor(img_crop2, cv2.COLOR_BGR2RGB))
# plt.subplot(144)
# plt.imshow(cv2.cvtColor(img_crop3, cv2.COLOR_BGR2RGB))
#
# plt.show()

# img_shift = color_shift(img_ori, 255, 255, 255)
# img_shift1 = color_shift(img_ori, -255, -255, -255)
# img_shift2 = color_shift(img_ori, 20, -23, 255)
# plt.subplot(141)
# plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
# plt.subplot(142)
# plt.imshow(cv2.cvtColor(img_shift, cv2.COLOR_BGR2RGB))
# plt.subplot(143)
# plt.imshow(cv2.cvtColor(img_shift1, cv2.COLOR_BGR2RGB))
# plt.subplot(144)
# plt.imshow(cv2.cvtColor(img_shift2, cv2.COLOR_BGR2RGB))
#
# plt.show()

# img_rotation = rotation(img_ori, (0, 0), 67, 1, (img_ori.shape[1], img_ori.shape[0]))
# plt.imshow(cv2.cvtColor(img_rotation, cv2.COLOR_BGR2RGB))
# plt.show()

img_pt = random_warp(img_ori, img_ori.shape[0], img_ori.shape[1])
plt.imshow(cv2.cvtColor(img_pt, cv2.COLOR_BGR2RGB))
plt.show()
