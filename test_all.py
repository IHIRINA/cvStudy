# 修改对应-> dcm_root就是我们的tofniigz文件夹的路径，为了获取信息生成对比数据

import os
import numpy as np
import yaml
from utils.dataset import dict_as_namespace
import pydicom
from pydicom import dcmread
import json
import torch
import h5py
import time
from model.BrownianBridge.BrownianBridge import BrownianBridgeModel
import SimpleITK as sitk
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd
from pydicom.uid import ExplicitVRLittleEndian


def pd_toExcel(data, fileName):  # pandas2excel
    with pd.ExcelWriter(fileName, engine='openpyxl') as writer:
        df = pd.DataFrame(data)
        df.to_excel(writer, index=False)


def createMIP(np_img, dim):
    ''' create the mip image from original image, slice_num is the number of 
    slices for maximum intensity projection'''
    shape = np_img.shape
    np_mip = np.zeros(shape)
    
    if dim == 0:
       for i in range(shape[0]):
           np_mip[i,:,:] = np.amax(np_img[0:i+1], 0)
    if dim == 1:
       for i in range(shape[1]):
           np_mip[:,i,:] = np.amax(np_img[:,0:i+1], 1)
    if dim == 2:
       for i in range(shape[2]):
           np_mip[:,:,i] = np.amax(np_img[:,:,0:i+1], 2)
    return np_mip


def hdf52data(root, name, modal):
    tempo_dict = {}
    with h5py.File(f'{root}{name}/{modal}.hdf5', 'r') as f:
        data = f['data'][:]
        tempo_dict['data'] = torch.tensor(data).float()
        tempo_dict['max'] = f['max'][()]
    return tempo_dict


def get_img(root, name):
    root = f'{root}/{name}/{name}_TOF_1x1x1.nii.gz'   
    '''
    reader = sitk.ImageSeriesReader()  # 读取3D array
    seriesIDs = reader.GetGDCMSeriesIDs(root)
    N = len(seriesIDs)
    lens = np.zeros([N])
    for i in range(N):
        dicom_names = reader.GetGDCMSeriesFileNames(root, seriesIDs[i])
        lens[i] = len(dicom_names)
    dicom_names = reader.GetGDCMSeriesFileNames(root)

    N_MAX = np.argmax(lens)
    dicom_names = reader.GetGDCMSeriesFileNames(root, seriesIDs[N_MAX])
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    '''
    img = sitk.ReadImage(root)
    print(f"Original size: {img.GetSize()}")  # (x, y, z) 即 (宽度, 高度, 深度)
    arr = sitk.GetArrayFromImage(img)
    print(f"Original array shape: {arr.shape}")  # (深度, 高度, 宽度)
    return img


def save_dicom_nii(name, out, gt, dcm_root):
    '''
    if not os.path.exists(f'./inference/{name}/dicom'):
        os.makedirs(f'./inference/{name}/dicom')
    dicom_files = out['dcm_files']
    '''

    if not os.path.exists(f'/root/autodl-tmp/results/inference/{name}'):
        os.makedirs(f'/root/autodl-tmp/results/inference/{name}')
    img = get_img(dcm_root, name)

    

    #data_nii = out['synthed'] * gt['max']/2 + gt['max']/2  # l, 290, 320
    # data_nii[:, 288:290] = 0
    # data_nii = data_nii.astype(np.int16)

    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()

    data_nii = ((out['synthed'] + 1.0) / 2.0 * 4095).clip(0, 4095).astype(np.int16)
    
    print(f"[{name}] Original TOF nii real_max = 1.00 (已归一化)")
    print(f"[{name}] 保存到 nii 的实际强度范围 → min={data_nii.min()}  max={data_nii.max()}")
    
    # 保存nii
    #data = np.transpose(data_nii, (0, 2, 1))  # for nii: l, 320, 290
    data = data_nii
    print(f"out shape = {data.shape}")
    res = sitk.GetImageFromArray(data)
    res.SetSpacing(spacing)
    res.SetDirection(direction)
    res.SetOrigin(origin)
    sitk.WriteImage(res, f'/root/autodl-tmp/results/inference/{name}/predict-3D-TOF-MRA.nii.gz')

    # 保存mip_nii
    for i in range(3):
       mip_img = createMIP(data, i)
       mip_img = sitk.GetImageFromArray(mip_img)
       mip_img.SetSpacing(spacing)
       mip_img.SetDirection(direction)
       mip_img.SetOrigin(origin)
       sitk.WriteImage(mip_img, f'/root/autodl-tmp/results/inference/{name}/MIP{i}.nii.gz')

    # 验证加了一行
    print(f"[{name}] 保存到 nii 的实际强度范围 → min={data_nii.min():.1f}  max={data_nii.max():.1f}   (gt_max={gt['max']:.1f})")

    
    # 保存dicom
    '''
    series_uid = pydicom.uid.generate_uid()
    data_dcm = np.transpose(data_nii, (2, 1, 0))  # for dcm: 320, 290, l  
    uid_pool = set()
    uid_pool.add(series_uid)
    for i in range(len(dicom_files)):

        sop_uid = pydicom.uid.generate_uid()
        while sop_uid in uid_pool:
            sop_uid = pydicom.uid.generate_uid()
        uid_pool.add(sop_uid)

        src_dcm = dicom_files[i]
        TransferSyntaxUID = getattr(src_dcm.file_meta, 'TransferSyntaxUID', None)
        if TransferSyntaxUID:
            pass
        else:
            src_dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        src_dcm.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian  # 原本压缩改为未压缩
        src_dcm.SeriesDescription = 'TOF_synth'
        src_dcm.SeriesInstanceUID = series_uid
        src_dcm.SOPInstanceUID = sop_uid
        src_dcm.DerivationDescription = ExplicitVRLittleEndian
        src_dcm.PixelData = data_dcm[:, :, i].tobytes()
        src_dcm.save_as(f'./inference/{name}/dicom/ImageFileName{i:03d}.dcm', write_like_original=True)  # 存为新的dicom文件
    '''

class Infer:
    def __init__(self, options, model_cofig, start, end):
        self.dcm_root = options.dcm_root
        self.hdf5_root = options.hdf5_root
        self.cuda_num = options.cuda_num
        self.num_slice = options.num_slice
        self.use_modality = options.use_modality
    
        with open("./3data.json", 'r') as load_f:
            load_dict = json.load(load_f)
            self.names = load_dict['valid'] + load_dict['test']
            self.names = self.names[start:end]

        net = BrownianBridgeModel(model_cofig)
        net.eval()
        ckpt = torch.load(f'/root/autodl-tmp/logs/checkpoints/epoch_{options.last_num}.ckpt', map_location=f'cuda:{self.cuda_num}')
        z = dict()
        for k, v in ckpt['state_dict'].items():
            k = k[5:]
            z[k] = v
        net.load_state_dict(z)
        self.net = net.cuda(self.cuda_num)

    def read(self, name):
        data = {}
        for modal in self.use_modality:
            data[modal] = hdf52data(self.hdf5_root, name, modal)
            max_ = data[modal]['max']
            data[modal]['data'] = (data[modal]['data'] - max_/2) / (max_/2)

        '''
        dicom_name = os.listdir(f'{self.dcm_root}{name}/3D-TOF-MRA/')  # 读取root下所有文件名
        dicom_names = [na for na in dicom_name if na[0] != '.']
        dicom_names.sort()
    
        data['dcm_files'] = [dcmread(f'{self.dcm_root}/{name}/3D-TOF-MRA/' + dn) for dn in dicom_names]  # 读取所有dicom文件
        '''
        return data

    def use_net(self, name):
        # use with self.read
        data = self.read(name)
        #l = data[self.use_modality[0]]['data'].shape[0]
        l, h, w = data[self.use_modality[0]]['data'].shape   # 改成动态获取尺寸

        # 加载到GPU处理
        for modal in self.use_modality:
            data[modal]['data'] = data[modal]['data'].cuda(self.cuda_num)
    
        data_nii = np.zeros([l, h, w], dtype=np.float32)

        '''
        for modal in self.use_modality:
            data[modal]['data'] = data[modal]['data'].cuda(self.cuda_num)
        data_nii = np.zeros([l, 290, 320])
        '''
       
        for i in range(l-self.num_slice+1):
            input = {}
            for modal in self.use_modality:
                input[modal] = data[modal]['data'][i:i+self.num_slice].unsqueeze(0)

            out, _, _, _ = self.net.sample_faster(input)
            # out = torch.log(out)
            out = out.squeeze(0)
            out = out.data.cpu().numpy()
        
            data_nii[i:i+self.num_slice] += out   
        
        for i in range(self.num_slice - 1):
            data_nii[i] = data_nii[i] / (i + 1)
            data_nii[l-i-1] = data_nii[l-i-1] / (i+1)
        for i in range(self.num_slice - 1, l - self.num_slice + 1):
            data_nii[i] = data_nii[i] / self.num_slice
        
        out = {}
        out['synthed'] = data_nii
        # out['dcm_files'] = data['dcm_files']

        # 验证加了一行
        print(f"[{name}] model output ([-1,1] normalized) → min={data_nii.min():.4f}  max={data_nii.max():.4f}")

        return out
    
    def save_cacu(self, name, gt):
        out = self.use_net(name)
        save_dicom_nii(name, out, gt, self.dcm_root)

        '''
        denoised = (out['synthed'] * gt['max']/2 + gt['max']/2) / gt['max']
        denoised[:, 288:290] = 0
        gt_array = gt['data'] / gt['max']
        target = np.zeros_like(denoised)
        target[:, :288] = gt_array
        '''
        denoised = (out['synthed'] * gt['max']/2 + gt['max']/2) / gt['max']
        gt_array = (gt['data'] / gt['max']).cpu().numpy()

        max_val = gt_array.max()
        mse = np.mean((gt_array - denoised)**2)
        psnr = 10 * np.log10(max_val**2 / mse)
        ssim = compare_ssim(denoised, gt_array, data_range=max_val)
        
        '''
        max_val = target.max()
        mse = np.sum(np.abs(target-denoised)**2)/target.size
        psnr = 10*np.log10(max_val ** 2 / mse)
        ssim = compare_ssim(denoised, target, data_range=max_val)
        '''

        return psnr, ssim
    
    def infer_all(self):
        result = {'case': [], 'psnr': [], 'ssim': []} 
        # result = {'case': [], 'psnr': [], 'ssim': [], 'mip_psnr': [], 'mip_ssim': []}
        for name in self.names:
            print(f'start infer {name}')
            gt = hdf52data(self.hdf5_root, name, '3D-TOF-MRA')
            psnr, ssim = self.save_cacu(name, gt)
            print(f'psnr: {psnr}, ssim: {ssim}\n')
            result['case'].append(name)
            result['psnr'].append(psnr)
            result['ssim'].append(ssim)
        pd_toExcel(result, f'/root/autodl-tmp/results/infer_epoch100.xlsx')


def main(cuda_num, last_num, start, end):
    options = {
        'cuda_num': cuda_num,
        'last_num': last_num,
        'dcm_root': '/root/autodl-tmp/data/Tof/preprocessed_tof/',         # 这个地方改成tof的niigz的路径
        'hdf5_root': '/root/autodl-tmp/data/hdf5/',       # 这个地方改成hdf5的路径
        'use_modality': ['3D-T1W', '3D-T2W', '3D-FLAIR'],
        'num_slice': 5
    }
    with open('./options.yaml', 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    model_config = dict_as_namespace(dict_config)
    options = dict_as_namespace(options)
    infer_pet_t2 = Infer(options, model_config, start, end)
    infer_pet_t2.infer_all()


if __name__ == '__main__':
   
    main(0, 100, 0, 24)
            
            
        


