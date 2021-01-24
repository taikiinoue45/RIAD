# RIAD
PyTorch re-implementation of [Reconstruction by Inpainting for Visual Anomaly Detection](https://www.sciencedirect.com/science/article/abs/pii/S0031320320305094)

<br>

## 1. AUROC Scores

| category   | Paper | My Implementation |
| :-         | :-    | :-                |
| zipper     | 0.981 | 0.975             |
| wood       | 0.930 | 0.965             |
| transistor | 0.909 | 0.918             |
| toothbrush | 1.000 | 0.972             |
| tile       | 0.987 | 0.997             |
| screw      | 0.845 | 0.799             |
| pill       | 0.838 | 0.786             |
| metal_nut  | 0.885 | 0.920             |
| leather    | 1.000 | 1.000             |
| hazelnut   | 0.833 | 0.890             |
| grid       | 0.996 | 0.983             |
| carpet     | 0.842 | 0.781             |
| capsule    | 0.884 | 0.731             |
| cable      | 0.819 | 0.655             |
| bottle     | 0.999 | 0.971             |

<br>

## 2. Graphical Results

### zipper
<p align="left">
  <img src=assets/zipper.gif width="100%" />
</p>

### wood
<p align="left">
  <img src=assets/wood.gif width="100%" />
</p>

### transistor
<p align="left">
  <img src=assets/transistor.gif width="100%" />
</p>

### toothbrush
<p align="left">
  <img src=assets/toothbrush.gif width="100%" />
</p>

### tile
<p align="left">
  <img src=assets/tile.gif width="100%" />
</p>

### screw
<p align="left">
  <img src=assets/screw.gif width="100%" />
</p>

### pill
<p align="left">
  <img src=assets/pill.gif width="100%" />
</p>

### metal_nut
<p align="left">
  <img src=assets/metal_nut.gif width="100%" />
</p>

### leather
<p align="left">
  <img src=assets/leather.gif width="100%" />
</p>

### hazelnut
<p align="left">
  <img src=assets/hazelnut.gif width="100%" />
</p>

### grid
<p align="left">
  <img src=assets/grid.gif width="100%" />
</p>

### carpet
<p align="left">
  <img src=assets/carpet.gif width="100%" />
</p>

### capsule
<p align="left">
  <img src=assets/capsule.gif width="100%" />
</p>

### cable
<p align="left">
  <img src=assets/cable.gif width="100%" />
</p>

### bottle
<p align="left">
  <img src=assets/bottle.gif width="100%" />
</p>

<br>

## 3. Requirements
- CUDA 10.2
- nvidia-docker2

<br>

## 4. Usage (WIP)
```
git clone git@github.com:TaikiInoue/RIAD.git
cd RIAD
```

```
docker pull taikiinoue45/mvtec:riad
```

```
docker run --runtime nvidia -it --rm taikiinoue45/mvtec:riad /usr/bin/zsh
```

```
sh run.sh
```

```
mlflow ui
```

<br>

## 5. Contacts

- github: https://github.com/taikiinoue45/
- twitter: https://twitter.com/taikiinoue45/
- linkedin: https://www.linkedin.com/in/taikiinoue45/
