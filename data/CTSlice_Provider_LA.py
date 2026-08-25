import os
import glob
import math
import numpy as np
import torch
import pydicom
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import odl
from odl.contrib import torch as odl_torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class LimitedAngleCT_Provider(Dataset):
    """
    Dataset loader for Limited-Angle CT (LA-CT) Reconstruction.
    Supports AAPM Mayo Clinic LDCT DICOM dataset.
    """
    def __init__(
        self,
        base_path,
        start_ang=-np.pi / 3,       # e.g., -60 deg for 120 deg range [-60, +60]
        end_ang=np.pi / 3,          # e.g., +60 deg
        num_view=64,                # Number of sampled projection angles in limited range
        num_detectors=512,
        poission_level=1e6,
        gaussian_level=0.05,
        test=False,
        valid=False,
        input_size=256,
        transform=None,
        use_precomputed=False,
        precomputed_setting=None
    ):
        self.base_path = base_path
        self.input_size = input_size
        self.transform = transform
        self.poission_level = poission_level
        self.gaussian_level = gaussian_level
        self.num_view = num_view
        self.num_detectors = num_detectors
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.use_precomputed = use_precomputed
        self.precomputed_setting = precomputed_setting

        # Standard AAPM Mayo LDCT splits
        patients_training = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L506"]
        patients_validation = ["L333"]
        patients_test = ["L310"]

        paths = []
        if valid:
            for patient_id in patients_validation:
                pattern = glob.glob(os.path.join(base_path, "train", patient_id, "full_3mm", f"{patient_id}_FD_3_1.CT.*.*.*.*.*.*.*.*.*.IMA"))
                paths.extend(pattern)
            split_name = "train"
        elif test:
            for patient_id in patients_test:
                pattern = glob.glob(os.path.join(base_path, "test", patient_id, "full_3mm", f"{patient_id}_FD_3_1.CT.*.*.*.*.*.*.*.*.*.IMA"))
                paths.extend(pattern)
            split_name = "test"
        else:
            for patient_id in patients_training:
                pattern = glob.glob(os.path.join(base_path, "train", patient_id, "full_3mm", f"{patient_id}_FD_3_1.CT.*.*.*.*.*.*.*.*.*.IMA"))
                paths.extend(pattern)
            split_name = "train"

        self.slices_path = sorted(paths)
        print(f"[{'VALID' if valid else ('TEST' if test else 'TRAIN')}] Loaded {len(self.slices_path)} CT slices.")

        # If precomputed npy files exist
        if self.use_precomputed and self.precomputed_setting:
            self.sino_dir = os.path.join(base_path, split_name, self.precomputed_setting, "sino")
            self.fbp_dir = os.path.join(base_path, split_name, self.precomputed_setting, "fbp_u")
        else:
            # Build ODL Limited-Angle Radon and FBP operators
            self.radon_op, self.fbp_op = self._build_limited_angle_operators(
                start_ang=self.start_ang,
                end_ang=self.end_ang,
                num_view=self.num_view,
                num_detectors=self.num_detectors
            )

    def _build_limited_angle_operators(self, start_ang, end_ang, num_view, num_detectors):
        xx = 200
        space = odl.uniform_discr([-xx, -xx], [xx, xx], [512, 512], dtype='float32')
        angles = np.array(num_view).astype(int)

        # Limited-Angle angular partition
        angle_partition = odl.uniform_partition(start_ang, end_ang, angles)
        detector_partition = odl.uniform_partition(-480, 480, num_detectors)

        # Fan-Beam Geometry
        geometry = odl.tomo.FanBeamGeometry(
            angle_partition,
            detector_partition,
            src_radius=600,
            det_radius=290
        )

        operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
        fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)

        op_layer = odl_torch.operator.OperatorModule(operator)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_fbp

    def __len__(self):
        return len(self.slices_path)

    def __getitem__(self, index):
        slice_path = self.slices_path[index]
        file_stem = os.path.basename(slice_path).split(".IMA")[0]

        # Load from precomputed cache if enabled
        if self.use_precomputed and self.precomputed_setting:
            sino_file = os.path.join(self.sino_dir, f"{file_stem}.npy")
            fbp_file = os.path.join(self.fbp_dir, f"{file_stem}.npy")
            
            sino = torch.from_numpy(np.load(sino_file)).float()
            fbp_u = torch.from_numpy(np.load(fbp_file)).float()

            # Read phantom ground truth
            dcm = pydicom.read_file(slice_path)
            data_slice = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            data_slice = data_slice.astype(float)
            data_slice = (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice) + 1e-8)
            phantom = torch.from_numpy(data_slice).unsqueeze(0).float()
        else:
            # Read and calibrate DICOM
            dcm = pydicom.read_file(slice_path)
            data_slice = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            data_slice = data_slice.astype(float)
            data_slice = (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice) + 1e-8)

            phantom = torch.from_numpy(data_slice).unsqueeze(0).float()

            # Forward projection on limited angle range
            sino = self.radon_op(phantom)

            # Add Poisson noise (low-dose photon simulation)
            if self.poission_level > 0:
                scale_val = torch.tensor(float(self.poission_level))
                norm_sino = torch.exp(-sino / (sino.max() + 1e-8))
                th_data = np.random.poisson((scale_val * norm_sino).cpu().numpy())
                sino_noisy = -torch.log(torch.from_numpy(th_data).float() / scale_val + 1e-8) * sino.max()
            else:
                sino_noisy = sino

            # Add Gaussian noise (electronic noise simulation)
            if self.gaussian_level > 0:
                noise = float(self.gaussian_level) * torch.randn_like(sino_noisy)
                sino_noisy = sino_noisy + noise

            # Filtered Backprojection reconstruction (with missing wedge artifacts)
            fbp_u = self.fbp_op(sino_noisy)
            sino = sino_noisy

        # Resize to network resolution (e.g., 256x256)
        if self.transform is not None:
            phantom = self.transform(phantom)
            fbp_u = self.transform(fbp_u)
        elif self.input_size != 512:
            resizer = transforms.Resize((self.input_size, self.input_size))
            phantom = resizer(phantom)
            fbp_u = resizer(fbp_u)

        return slice_path, phantom, fbp_u, sino
