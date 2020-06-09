## Usage

Entrypoints are `train.py` are `infer.py`. Please check options for example by `python train.py -h`.

## Env

### export env

```
$ conda env export | grep -v "^prefix: " > ./environment.yml
```

### update env

```
$ conda env update
```

### create env

```
$ conda env create --name prostate --file ./environment.yml
```
