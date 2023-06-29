# **Project of  MAPLE**
MAPLE: Masked Pseudo-Labeling autoEncoder for Semi-supervised Point Cloud Action Recognition. (ACM MM 2022 Oral)

Our MAPLE Project is based on [FastReID](https://github.com/JDAI-CV/fast-reid)、[P4Transformer (2021 version)](https://github.com/hehefan/P4Transformer/tree/832b0529bf8f2ac941f5871f3d94c04325506a63)、[3DV](https://github.com/3huo/3DV-Action)、[PointNet++](https://github.com/facebookresearch/votenet/tree/master/pointnet2)

## **Requirements of Computer Hardware**
 - GPU>=16GB memory
 - RAM>=32GB memory

## **Installation**
 - Linux with CUDA 10.2 and cuDNN v7.6
 - python ≥ 3.6
 - PyTorch == 1.8.0
 - torchvision == 0.9.0
 - Compile the CUDA layers for PointNet++, which we used for furthest point sampling (FPS) and radius neighbouring search:
    ```bash
    mv modules-pytorch-1.8.1 modules
    cd modules
    python setup.py install
    ```

## **Data loading acceleration**
 - We use DataloaderX instead of torch.nn.Dataloader for data loading acceleration
 - We use @functools.lru_cache(100) in [NTU60](./datasets/ntu60.py) for data loading acceleration with RAM. The size of the cache depends on the size of RAM, for example:
    ```text
    @functools.lru_cache(5000) need about 160GB RAM for data loading acceleration
    @functools.lru_cache(500) need about 16GB RAM for data loading acceleration
    ```

## **Dataset preparation**
 - The preparation of MSR-Action3D dataset
    
    1. Download MSR-Action3D from [url](http://wangjiangb.github.io/my_data.html)
    2. move file:
    
        ```bash
        mv Depth.rar ./data/MSRAction3D/
        ```

    3. unrar the `Depth.rar` file and preprocess the MSRAction3D dataset:
        ```bash
        cd ./data/MSRAction3D/
        # unrar the zip file
        unrar e Depth.rar
        # mkdir
        mkdir ./point
        # preprocess
        python preprocess_file.py --input_dir ./Depth --output_dir ./point --num_cpu 8
        ```
    4. make them look like this:
        ```text
        MAPLE
        ├── datasets
        ├── modules
        `── data
            │── MSRAction3D
                │-- preprocess_file.py
                │-- Depth
                `-- point
                    │-- a01_s01_e01_sdepth.npz
                    │-- a01_s01_e02_sdepth.npz
                    │-- a01_s01_e03_sdepth.npz
                    │-- ...

        ```
    
 - The preparation of NTU RGBD 60/120 dataset
     1. Download NTU RGBD dataset from [rose_lab](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
     2. move file and unzip:
        ```bash
        # mkdir
        mkdir ./data/ntu/npy_faster
        mkdir ./data/ntu/npy_faster/point_reduce_without_sample
        # mv
        mv nturgbd_depth_masked_s0??.zip ./data/ntu/
        # cd 
        cd ./data/ntu/
        # unzip
        unzip nturgbd_depth_masked_s0??.zip
        ```
     3. preprocess the NTU dataset from png to npy
        ```bash
        # runs in the background for 2~4 hours
        bash depth2point4ntu.sh
        # After 2~4 hours, check whether the number of files is 114480
        ls ./npy_faster/point_reduce_without_sample/ -l | grep "^-" | wc -l
        ```
    4. make them look like this:
        ```text
        MAPLE
        ├── datasets
        ├── modules
        `── data
            │── ntu
                │-- depth2point4ntu.py
                │-- depth2point4ntu.sh
                │-- nturgb+d_depth_masked
                `-- npy_faster
                    `-- point_reduce_without_sample
                        │-- S001C001P001R001A001.npy
                        │-- S001C001P001R001A002.npy
                        │-- ...

        ```


 ## Train baseline of 5%/10%/20%/30%/40% superveised baseline model

```bash
# train baseline of MSR-Action dataset
bash ./train-msr-baseline.sh
# train baseline of NTU-60 dataset
bash ./train-ntu-baseline.sh
# train baseline of NTU-120 dataset
bash ./train-ntu-120-baseline.sh
```

 ## Training for pseudo label baseline
```bash
# train pseudo label of MSR-Action dataset
bash ./pseudo_labels/train_msr_pseudo.sh
# train pseudo label of NTU-60 dataset
bash ./pseudo_labels/train_ntu_pseudo.sh
# train pseudo label of NTU-120 dataset
bash ./pseudo_labels/train_ntu120_pseudo.sh
```

 ## Training for VAT and VAT+EntMin
```bash
# use MSR-Action dataset as example, bash file of other datasets is under ./vat
# train VAT + Entmin
bash vat/train_vat_msr_gpu0.sh
# or train VAT + Entmin + resume from pretrained model
bash vat/train_vat_msr_gpu0_resume.sh
# if you need to training for VAT, just remove --vat-EntMin in bash file, such as ./vat/train_vat_ntu.sh
```

 ## Training for MAPLE
```bash
# use MSR-Action dataset as example, bash file of other datasets is under ./z_mask
# train MAPLE
bash ./z_mask/train_mse_msr_gpu0.sh
```

 ## Training for VAT+Entmin+MAPLE   
```bash
# use MSR-Action dataset as example, bash file of other datasets is under ./z_mask
# train VAT+Entmin+MAPLE, put the best pretrained VAT+Entmin model of MSR-Action under ./output_msr/entmin/, such as ./output_msr/entmin/model_best_1.pth
bash ./z_mask/train_mse_msr_gpu0_resume_from_entmin_mask.sh
```

## **Citing MAPLE**

If you use MAPLE in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@inproceedings{chen2022MAPLE,
title={MAPLE: Masked Pseudo-Labeling autoEncoder for Semi-supervised Point Cloud Action Recognition},
author={Xiaodong Chen and Wu Liu and Xinchen Liu and Yongdong Zhang and Jungong Han and Tao Mei},
booktitle={ACM Multimedia (ACM MM)},
year={2022}
}
```
