import datetime

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def dice_coef(input, target, smooth = 1.):
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

def argmax_acc(input, target, smooth = 1.):
    iflat = input.view(-1)
    tflat = target.view(-1)
