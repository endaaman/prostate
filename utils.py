import sys
import datetime
import numpy as np
import cv2
import torch


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def pack(arr):
    return [x for x in arr if x]

def curry(*args, **kwds):
    def callit(*moreargs, **morekwds):
        kw = kwds.copy()
        kw.update(morekwds)
        return args[0](*(args[1:]+moreargs), **kw)
    return callit

last_message = None
def pp(message):
    global last_message
    if last_message:
        sys.stdout.write('\r' * len(last_message))
    last_message = message
    sys.stdout.write(message)
    sys.stdout.flush()

def revert_onehot(t):
    t = t.permute(0, 2, 3, 1)
    num_classes = t.size(-1)
    _, img = torch.max(t.contiguous().view(-1, num_classes), 1)
    return img

def similarity_index(a, b, smooth=1.):
    a = a.contiguous().view(-1)
    b = b.contiguous().view(-1)
    A = a.sum().item()
    B = b.sum().item()
    inter = (a * b).sum().item()
    dice = (inter * 2.0 + smooth) / (A + B + smooth)
    jaccard = (inter + smooth) / (A + B - inter + smooth)
    return dice, jaccard

def pixel_similarity_index(a, b, smooth=1.):
    A = a.size(0)
    B = b.size(0)
    inter = (a == b).sum().item()
    dice = (inter * 2.0 + smooth) / (A + B + smooth)
    jaccard = (inter + smooth) / (A + B - inter + smooth)
    return dice, jaccard

def inspection_accuracy(pr_arr, gt_arr, smooth=1):
    pr_arr = pr_arr.view(-1)
    gt_arr = gt_arr.view(-1)
    U = gt_arr.size(0)
    pr = pr_arr.sum().item()
    gt = gt_arr.sum().item()
    tp = (pr_arr * gt_arr).sum().item()
    sensitivity = (tp + smooth) / (gt + smooth)
    specificity = (U - pr - gt + tp + smooth) / (U - gt + smooth)
    return sensitivity, specificity

def to_heatmap(org, base_color=[0, 255, 0]):
    z = np.zeros([*org.shape], dtype=np.uint8)
    c255 = np.full([*org.shape], 255, dtype=np.uint8)
    c200 = np.full([*org.shape], 200, dtype=np.uint8)
    vc = ((1 - org) * 170).astype(np.uint8)
    img = np.dstack((vc, c255, c255))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    va = np.full([*org.shape], 127, dtype=np.uint8)
    va[org < 0.1] = 0
    img = np.dstack((img, va))
    return img

def overlay_transparent(background_img, img_to_overlay_t, x=0, y=0, overlay_size=None):
    bg_img = background_img.copy()
    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.medianBlur(a, 5)
    h, w, _ = overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask = cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask = mask)
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
    return bg_img
