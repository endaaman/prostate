import sys
import datetime
import numpy as np
import cv2
import torch

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

last_message = None
def pp(message):
    global last_message
    if last_message:
        sys.stdout.write('\r' * len(last_message))
    last_message = message
    sys.stdout.write(message)
    sys.stdout.flush()

def dice_coef(a, b, smooth = 1.):
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a * b).sum()
    return ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth)).item()

def argmax_acc(a, b):
    num_classes = a.size(1)
    num_channels = a.nelement()
    _, a = torch.max(a.permute(0, 2, 3, 1).contiguous().view(-1, num_classes), 1)
    _, b = torch.max(b.permute(0, 2, 3, 1).contiguous().view(-1, num_classes), 1)
    return (a == b).sum().item() / (num_channels // num_classes)

def curry(*args, **kwds):
    def callit(*moreargs, **morekwds):
        kw = kwds.copy(  )
        kw.update(morekwds)
        return args[0](*(args[1:]+moreargs), **kw)
    return callit

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

def save_report(output_dir, input_img, mask_img):
    cv2.imwrite(f'{output_dir}/org.jpg', input_img)
    cv2.imwrite(f'{output_dir}/out.png', mask_img)
    masked_img = overlay_transparent(input_img, mask_img)
    cv2.imwrite(f'{output_dir}/masked.png', masked_img)
    for i in range(NUM_CLASSES):
        img = to_heatmap(mask_arr[:, :, i])
        cv2.imwrite(f'{output_dir}/heat_{i}.png', img)
        fused = overlay_transparent(input_img, img)
        cv2.imwrite(f'{output_dir}/fused_{i}.png', fused)
