# todo

- calc acc by dice coef or jaccard
- fix RGB channel changing
- fix flip/rotate result of inference

## data augmentation

- universal rotating

## recipe

### crop

```
$ python crop.py '224/18_2500/x' 224 inputs/18_2500/1_0_tile.jpg
```

### train

put files in appropriate dir and

```
$ python train.py
```

### inference

```
$ python infer.py ./weights/100.pt tmp/hoge.jpg
```

### export env

```
$ conda env export | grep -v "^prefix: " > ./env.yml
```

### load env

```
$ conda env create --name prostate --file ./env.yml
```

