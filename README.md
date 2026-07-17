# TOF-MRA-Synth

MICCAI 2025 early accpet

Official code for [Diffusion-based Multi-modal MR Fusion for TOF-MRA Image Synthesis](https://papers.miccai.org/miccai-2025/paper/0751_paper.pdf)



## Data preprocessing

> All the data was registered first by SimpleITK
>
> We use **hdf5** as the data storage method

- data: 3D data array (shape: z, x, y)
- max: max intensity
- max_len: num of slice (Axial)

## Usage

### Requirements

`pip install -r requirements.txt`

### Data storage
```
/hdf5_root/case_names/modality.dhf5  
/dcm_root/case_name/...  # TOF only, for dicom templete 
```
> 'modality' contains '3D-T1W', '3D-T2W', '3D-FLAIR', '3D-TOF-MRA'

### Train

```
python train.py
```  
> options.yaml: Adjust parameters
> 
> 3data.json: data split
### Inference

```python test_all.py```  
options.data_root: hdf5_root  
options.dcm_root: dicom_root

## Acknowledgements

Thanks to Shanghai Sixth People's Hospital and Subtle Medical Inc.

Our code is based on [BBDM](https://github.com/xuekt98/BBDM)



