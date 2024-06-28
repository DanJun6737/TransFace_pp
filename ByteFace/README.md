# ByteFace
This is the official Pytorch implementation of ByteFace

## Requirements
* Install Pytorch (torch>=1.9.0)
* ```pip install -r requirement.txt```

## Datasets
You can download the training dataset MS1MV2:
* MS1MV2: [Google Drive](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view)

You can download the test dataset IJB-C as follows:
* IJB-C: [Google Drive](https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view) 

## How to Train Models
1. You need to modify the path of training dataset in every configuration file in folder **configs**.

2. To run on a machine with 8 GPUs:
   
a. TIFF
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_tiff_byteface.py 
```

b. PNG
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_png_byteface.py 
```

c. fCHW
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_fchw_byteface.py 
```

d. fHWC
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_fhwc_byteface.py 
```

## ByteFace Pretrained Models 

* You can **download** the ByteFace models reported in our paper as follows:

**Verification accuracy (%) on the IJB-C benchmark.**
| Training Data | Model (Link) | Data Format | IJB-C(1e-6) | IJB-C(1e-5) | IJB-C(1e-4) | IJB-C(1e-3) | IJB-C(1e-2) | IJB-C(1e-1) |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| MS1MV2 | [**ByteFace-S**](https://drive.google.com/file/d/1KpDUdQTIH8-ToRaw8_4yYenc_1f7hs0t/view?usp=sharing) | fHWC | 84.76 | 93.20 | 95.81 | 97.24 | 98.30 | 99.11 |
| MS1MV2 | [**ByteFace-S**](https://drive.google.com/file/d/18ZOBV5K_9U3vap94RMjrGTEk0sSzR3cZ/view?usp=sharing) | fCHW | 83.36 | 93.17 | 95.55 | 97.10 | 98.21 | 99.01 |
| MS1MV2 | [**ByteFace-S**](https://drive.google.com/file/d/1KVjMjHPZMAdDP-bmWYhsyypACTPzjJDt/view?usp=sharing) | TIFF | 85.65 | 93.37 | 95.87 | 97.26 | 98.14 | 98.99 |
| MS1MV2 | [**ByteFace-S**](https://drive.google.com/file/d/1BG-f68K8ZILU9QBqOQo1F8c2lpIjUoHA/view?usp=sharing) | PNG | 76.55 | 89.59 | 93.84 | 96.11 | 97.49 | 98.56 |
| MS1MV2 | [**ByteFace-B**](https://drive.google.com/file/d/1c5zNffl57m0WZNAHflTcTNuxkIjp6NR0/view?usp=sharing) | fHWC | 86.43 | 93.94 | 96.27 | 97.65 | 98.46 | 99.05 |
| MS1MV2 | [**ByteFace-B**](https://drive.google.com/file/d/10BdJKs1DLPIaUfHb5kcs4JQUl22nEV1x/view?usp=sharing) | fCHW | 85.36 | 93.72 | 96.14 | 97.48 | 98.37 | 99.07 |
| MS1MV2 | [**ByteFace-B**](https://drive.google.com/file/d/15SadWtrFY_BrRculQcI5vBlRyE4jWn4Z/view?usp=sharing) | TIFF | 88.83 | 94.24 | 96.41 | 97.83 | 98.50 | 99.18 |
| MS1MV2 | [**ByteFace-B**](https://drive.google.com/file/d/1doqnC1TWgqxRww5do1r6KYBhGAxzq_NR/view?usp=sharing) | PNG | 80.66 | 91.85 | 95.43 | 97.24 | 98.26 | 99.04 |

* You can test the accuracy of these model.

## How to Test Models
1. You need to modify the path of IJB-C dataset in eval_ijbc_ms1mv2_tiff.py, eval_ijbc_ms1mv2_png.py, eval_ijbc_ms1mv2_fhwc.py, and eval_ijbc_ms1mv2_fchw.py.

2. Run:

a. TIFF
```
python eval_ijbc_ms1mv2_tiff.py --model-prefix work_dirs/tiff_byteface/model.pt --result-dir work_dirs/tiff_byteface --network vit_s_dp005_mask_0 > ijbc_ms1mv2_tiff_byteface_s.log 2>&1 &
```

b. PNG
```
python eval_ijbc_ms1mv2_png.py --model-prefix work_dirs/png_byteface/model.pt --result-dir work_dirs/png_byteface --network vit_s_dp005_mask_0 > ijbc_ms1mv2_png_byteface_s.log 2>&1 &
```

c. fHWC
```
python eval_ijbc_ms1mv2_fhwc.py --model-prefix work_dirs/fHWC_byteface/model.pt --result-dir work_dirs/fHWC_byteface --network vit_s_dp005_mask_0 > ijbc_ms1mv2_fhwc_byteface_s.log 2>&1 &
```

d. fCHW
```
python eval_ijbc_ms1mv2_fchw.py --model-prefix work_dirs/fCHW_byteface/model.pt --result-dir work_dirs/fCHW_byteface --network vit_s_dp005_mask_0 > ijbc_ms1mv2_fchw_byteface_s.log 2>&1 &
```

Thanks!
