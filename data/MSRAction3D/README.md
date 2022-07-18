

## Data Processing

1. Download MSR-Action3D from [url](http://wangjiangb.github.io/my_data.html) or [google_drive](https://drive.google.com/file/d/1djwAK3oZTAIFbCz531eClxINmsZgGO_H/view?usp=sharing)
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
    # preprocess,  `--num_cpu` flag is used to specify the number of CPUs to use during parallel processing.
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
