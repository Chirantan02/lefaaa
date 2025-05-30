import cv2
import numpy as np
import os
import torch
import onnxruntime as ort
import platform
from .onnxdet import inference_detector
from .onnxpose import inference_pose

class Wholebody:
    def __init__(self, model_root, device):
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            # Add cuDNN path to environment
            if platform.system() == 'Windows':
                cudnn_path = os.path.join(os.environ.get('CUDA_PATH', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2'), 'bin')
                if os.path.exists(cudnn_path):
                    os.add_dll_directory(cudnn_path)

            # Check if CUDA provider is available
            available_providers = ort.get_available_providers()
            providers = []

            # Add CUDA provider
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': torch.cuda.current_device(),
                    'arena_extend_strategy': 'kNextPowerOfTwo',  # More efficient memory allocation
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit for RTX 4060
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',  # Find most memory-efficient algorithm
                    'do_copy_in_default_stream': True
                    # Removed unsupported option: 'gpu_force_graph_level'
                }))

            # Always add CPU as fallback
            providers.append('CPUExecutionProvider')

            if not providers:
                print("WARNING: No GPU providers available. Using CPU provider.")
                providers = ['CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.enable_cpu_mem_arena = False  # Disable memory arena to reduce memory usage
        session_options.enable_mem_pattern = False
        session_options.enable_mem_reuse = True
        session_options.intra_op_num_threads = 1  # Reduce thread count to minimize memory usage
        session_options.inter_op_num_threads = 1
        session_options.enable_profiling = False
        # Optimize memory allocation
        session_options.add_session_config_entry('session.use_memory_efficient_allocation', '1')
        session_options.add_session_config_entry('session.use_arena_allocation', '0')
        # Set smaller memory limits
        session_options.add_session_config_entry('session.max_mem_alloc_size', str(512 * 1024 * 1024))  # 512MB limit
        if device.startswith('cuda'):
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC  # Less aggressive optimization
        # Check if the model_root contains FitDiT subdirectory
        # Check multiple possible model locations
        possible_paths = [
            os.path.join(model_root, 'mask', 'model', 'FitDiT', 'dwpose'),
            os.path.join(model_root, 'FitDiT', 'dwpose'),
            os.path.join(model_root, 'dwpose')
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'yolox_l.onnx')):
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError("Could not find model files in any of: " + str(possible_paths))

        onnx_det = os.path.join(model_path, 'yolox_l.onnx')
        onnx_pose = os.path.join(model_path, 'dw-ll_ucoco_384.onnx')

        # Try to load models with GPU, fall back to CPU if memory error occurs
        try:
            self.session_det = ort.InferenceSession(onnx_det, sess_options=session_options, providers=providers)
            self.session_pose = ort.InferenceSession(onnx_pose, sess_options=session_options, providers=providers)
        except Exception as e:
            if 'bad allocation' in str(e) and 'CPUExecutionProvider' not in providers[0]:
                print("WARNING: GPU memory allocation failed. Falling back to CPU.")
                # Fall back to CPU
                cpu_providers = ['CPUExecutionProvider']
                self.session_det = ort.InferenceSession(onnx_det, sess_options=session_options, providers=cpu_providers)
                self.session_pose = ort.InferenceSession(onnx_pose, sess_options=session_options, providers=cpu_providers)
            else:
                raise e

    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]

        return keypoints, scores