export CUDA_VISIBLE_DEVICES="0"

epoch=20
tile=512
dest="weights/$tile"

python train.py -m unet11    -b  6 -e $epoch --tile $tile --dest $dest
python train.py -m unet11b   -b  4 -e $epoch --tile $tile --dest $dest
python train.py -m unet11n   -b  4 -e $epoch --tile $tile --dest $dest
python train.py -m unet16    -b  6 -e $epoch --tile $tile --dest $dest
python train.py -m unet16b   -b  4 -e $epoch --tile $tile --dest $dest
python train.py -m unet16n   -b  4 -e $epoch --tile $tile --dest $dest
python train.py -m albunet   -b  8 -e $epoch --tile $tile --dest $dest
python train.py -m albunet_b -b  4 -e $epoch --tile $tile --dest $dest
python train.py -m albunet_n -b  4 -e $epoch --tile $tile --dest $dest


tile=256
dest="weights/$tile"
python train.py -m unet11    -b 20 -e $epoch --tile $tile --dest $dest
python train.py -m unet11b   -b 12 -e $epoch --tile $tile --dest $dest
python train.py -m unet11n   -b 12 -e $epoch --tile $tile --dest $dest
python train.py -m unet16    -b 20 -e $epoch --tile $tile --dest $dest
python train.py -m unet16b   -b 12 -e $epoch --tile $tile --dest $dest
python train.py -m unet16n   -b 12 -e $epoch --tile $tile --dest $dest
python train.py -m albunet   -b 32 -e $epoch --tile $tile --dest $dest
python train.py -m albunet_b -b 16 -e $epoch --tile $tile --dest $dest
python train.py -m albunet_n -b 16 -e $epoch --tile $tile --dest $dest
