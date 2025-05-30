# TransFace++
This is the official Pytorch implementation of TransFace++: Rethinking the Face Recognition Paradigm with a Focus on Efficiency, Security, and Precision.

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
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_tiff_transface++.py 
```

b. PNG
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_png_transface++.py 
```

c. fCHW
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_fchw_transface++.py 
```

d. fHWC
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_fhwc_transface++.py 
```

## How to Test Models
1. You need to modify the path of IJB-C dataset in [eval_ijbc_ms1mv2_tiff.py](https://github.com/DanJun6737/TransFace_pp/blob/main/TransFace%2B%2B/eval_ijbc_ms1mv2_tiff.py), [eval_ijbc_ms1mv2_png.py](https://github.com/DanJun6737/TransFace_pp/blob/main/TransFace%2B%2B/eval_ijbc_ms1mv2_png.py), [eval_ijbc_ms1mv2_fhwc.py](https://github.com/DanJun6737/TransFace_pp/blob/main/TransFace%2B%2B/eval_ijbc_ms1mv2_fhwc.py), and [eval_ijbc_ms1mv2_fchw.py](https://github.com/DanJun6737/TransFace_pp/blob/main/TransFace%2B%2B/eval_ijbc_ms1mv2_fchw.py).

2. Run:

a. TIFF
```
python eval_ijbc_ms1mv2_tiff.py --model-prefix work_dirs/tiff_transface++/model.pt --result-dir work_dirs/tiff_transface++ --network vit_s_dp005_mask_0 > ijbc_ms1mv2_tiff_transface++_s.log 2>&1 &
```

b. PNG
```
python eval_ijbc_ms1mv2_png.py --model-prefix work_dirs/png_transface++/model.pt --result-dir work_dirs/png_transface++ --network vit_s_dp005_mask_0 > ijbc_ms1mv2_png_transface++_s.log 2>&1 &
```

c. fHWC
```
python eval_ijbc_ms1mv2_fhwc.py --model-prefix work_dirs/fHWC_transface++/model.pt --result-dir work_dirs/fHWC_transface++ --network vit_s_dp005_mask_0 > ijbc_ms1mv2_fhwc_transface++_s.log 2>&1 &
```

d. fCHW
```
python eval_ijbc_ms1mv2_fchw.py --model-prefix work_dirs/fCHW_transface++/model.pt --result-dir work_dirs/fCHW_transface++ --network vit_s_dp005_mask_0 > ijbc_ms1mv2_fchw_transface++_s.log 2>&1 &
```

Thanks!
