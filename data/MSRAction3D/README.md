

## Data Processing

Download raw MSRAction3D dataset from <a href="https://drive.google.com/file/d/1djwAK3oZTAIFbCz531eClxINmsZgGO_H/view?usp=sharing">here</a> (~62MB). Extract the `.zip` file to get `Depth.rar` file and extract it in `Depth` directory. Then in this directory, run the following command to preprocess the data. `--num_cpu` flag is used to specify the number of CPUs to use during parallel processing.

```
python preprocess_file.py --input_dir /path/to/Depth --output_dir processed_data --num_cpu 11
```

