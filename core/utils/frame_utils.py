import numpy as np
from PIL import Image
from os.path import *
import re
import json
import imageio
import cv2
import h5py
from scipy.ndimage import binary_dilation

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-np

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writePFM(file, array):
    import os
    assert type(file) is str and type(array) is np.ndarray and \
           os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())



def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    #disp = getNormalizedDisp(disp)
    
    #print(disp.max(),disp.min())
    # valid = disp > 0.0
    valid = (disp > 0.01) & (disp < 80.0)
    return disp, valid

# Method taken from /n/fs/raft-depth/RAFT-Stereo/datasets/SintelStereo/sdk/python/sintel_io.py
def readDispSintelStereo(file_name):
    a = np.array(Image.open(file_name))
    d_r, d_g, d_b = np.split(a, axis=2, indices_or_sections=3)
    disp = (d_r * 4 + d_g / (2**6) + d_b / (2**14))[..., 0]
    mask = np.array(Image.open(file_name.replace('disparities', 'occlusions')))
    valid = ((mask == 0) & (disp > 0))
    return disp, valid

# Method taken from https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt
def readDispFallingThings(file_name):
    a = np.array(Image.open(file_name))
    with open('/'.join(file_name.split('/')[:-1] + ['_camera_settings.json']), 'r') as f:
        intrinsics = json.load(f)
    fx = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']
    disp = (fx * 6.0 * 100) / a.astype(np.float32)
    valid = disp > 0
    return disp, valid

def getNormalizedDisp(depth):
    eps = 1e-3
    mask = depth > 0
    disp = np.zeros_like(depth, dtype=np.float32)
    disp[mask] = 1.0 / (depth[mask] + eps)
    
    # normalize to 0~1
    if np.any(mask):
        disp_min = np.min(disp[mask])
        disp_max = np.max(disp[mask])
        disp[mask] = (disp[mask] - disp_min) / (disp_max - disp_min + eps)

    return disp

def Disp_to_NormalizedDisp(disp):
    eps = 1e-3
    mask = disp > 0
    
    # normalize to 0~1
    if np.any(mask):
        disp_min = np.min(disp[mask])
        disp_max = np.max(disp[mask])
        disp[mask] = (disp[mask] - disp_min) / (disp_max - disp_min + eps)

    return disp

# Method taken from https://github.com/castacks/tartanair_tools/blob/master/data_type.md
def readDispTartanAir(file_name):
    depth = np.load(file_name)
    disp = getNormalizedDisp(depth)
    valid = disp > 0

    return disp, valid

# Method taken from https://www.cs.cornell.edu/projects/megadepth/
def readDispMegaDepth(file_name):
    hdf5_file_read = h5py.File(file_name,'r')
    depth = hdf5_file_read.get('/depth')
    depth = np.array(depth)
    hdf5_file_read.close()

    disp = getNormalizedDisp(depth)
    valid = disp > 0

    return disp, valid

# Method taken from https://kexianhust.github.io/Structure-Guided-Ranking-Loss/
def readDispHRWSI(file_name):
    disp = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH) / 255

    mask_file_name = file_name.replace('gts', 'valid_masks')
    mask = cv2.imread(mask_file_name, cv2.IMREAD_GRAYSCALE)
    valid_disp = np.where(mask == 255, disp, 0)
    valid = valid_disp > 0

    return disp, valid

# Method taken from https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/
def readDispVKITTI(file_name):
    depth = cv2.imread(file_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0 
    disp = getNormalizedDisp(depth)
    valid = depth <= 80

    return disp, valid


import OpenEXR
import Imath

def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if (CNum > 1):
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1],Size[0],CNum),dtype=np.float32)
    if (CNum == 1):
        hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
    else:
        hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
        hdr[:,:,1] = np.reshape(Pixels[1],(Size[1],Size[0]))
        hdr[:,:,2] = np.reshape(Pixels[2],(Size[1],Size[0]))
    return hdr

def load_exr(filename):
	hdr = exr2hdr(filename)
	h, w, c = hdr.shape
	if c == 1:
		hdr = np.squeeze(hdr)
	return hdr

def readDispIRS(file_name):
    gt_disp = load_exr(file_name)
    disp = Disp_to_NormalizedDisp(gt_disp)
    valid = disp > 0

    return disp, valid


def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth


# Method taken from https://github.com/apple/ml-hypersim
def readDispHypersim(file_name):
    depth_fd = h5py.File(file_name, "r")
    distance_meters = np.array(depth_fd['dataset'])
    depth = hypersim_distance_to_depth(distance_meters)
    disp = getNormalizedDisp(depth)

    
    valid= ~np.isnan(depth)

    return disp, valid


# Method taken from https://github.com/YoYo000/BlendedMVS
def readDispBlendedMVS(file_name):
    def expandMask(depth):
        mask = (depth == 0)
        # expand masks by 3px
        dilated_mask = binary_dilation(mask, structure=np.ones((7, 7)))
        depth[dilated_mask] = 0

        return depth

    def getFilteredDisp(depth):
        positive_depth = depth[depth > 0]

        q1 = np.percentile(positive_depth, 25)
        q3 = np.percentile(positive_depth, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        valid_depth = positive_depth[(positive_depth >= lower_bound) & (positive_depth <= upper_bound)]    
        mask = np.isin(depth, valid_depth)    
        disp = np.zeros_like(depth, dtype=np.float32)
        eps = 1e-3
        disp[mask] = 1.0 / (depth[mask] + eps)

        if np.any(mask):
            disp_min = np.min(disp[mask])
            disp_max = np.max(disp[mask])
            disp[mask] = (disp[mask] - disp_min) / (disp_max - disp_min + eps)

        return disp

    depth = readPFM(file_name)
    masked_depth = expandMask(depth)
    # disp = getNormalizedDisp(depth)
    disp = getFilteredDisp(masked_depth)
    valid = disp > 0

    return disp, valid

def readDispMiddlebury(file_name):
    if basename(file_name) == 'disp0GT.pfm':
        disp = readPFM(file_name).astype(np.float32)
        assert len(disp.shape) == 2
        nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
        assert exists(nocc_pix)
        nocc_pix = imageio.imread(nocc_pix) == 255
        assert np.any(nocc_pix)
        return disp, nocc_pix
    elif basename(file_name) == 'disp0.pfm':
        disp = readPFM(file_name).astype(np.float32)
        valid = disp < 1e3
        return disp, valid

def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])
    

def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []

import numpy as np
import imageio.v2 as imageio
import cv2

def save_and_print_stats(disp, valid, gt_disp):
    # 정규화를 위한 min-max 값
    disp_min, disp_max = disp.min(), disp.max()
    gt_disp_min, gt_disp_max = gt_disp.min(), gt_disp.max()
    
    # disp와 gt_disp 정규화 (0~255)
    disp_norm = ((disp - disp_min) / (disp_max - disp_min + 1e-8) * 255).astype(np.uint8)
    gt_disp_norm = ((gt_disp - gt_disp_min) / (gt_disp_max - gt_disp_min + 1e-8) * 255).astype(np.uint8)
    
    # valid mask는 0 또는 255로
    valid_mask = (valid.astype(np.uint8)) * 255

    # 이미지 저장
    imageio.imwrite("disp.png", disp_norm)
    imageio.imwrite("gt_disp.png", gt_disp_norm)
    imageio.imwrite("valid.png", valid_mask)

    # min-max 출력
    print(f"disp: min={disp_min:.4f}, max={disp_max:.4f}")
    print(f"gt_disp: min={gt_disp_min:.4f}, max={gt_disp_max:.4f}")
    print(f"valid: unique values={np.unique(valid)}")


if "__main__" == __name__:
    exr = "/home/wodon326/data2/IRS/Home/ArchVizInterior03Data/d_00001.exr"
    disp, valid = readDispIRS(exr)  # file_name에 맞게 변경
    gt_disp = load_exr(exr)

    save_and_print_stats(disp, valid, gt_disp)
