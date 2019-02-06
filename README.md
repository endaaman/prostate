# todo

1. crop/mix image
1. build U-Net
1. learn

## data augmentation

- scramble
- rotate
- flip
- overwraping slide

## MEMO

### crop

```
$ python crop.py '224/18_2500/x' 224 inputs/18_2500/1_0_tile.jpg
```

### export env

```
$ conda env export | grep -v "^prefix: " > ./env.yml
```

### load env

```
$ conda env create --name prostate --file ./env.yml
```

