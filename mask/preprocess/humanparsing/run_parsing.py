import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference
import torch


class Parsing:
    def __init__(self, model_root, device):
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            # Check if CUDA provider is available
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB to avoid memory issues
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider'
                ]
            else:
                print("WARNING: CUDA provider not available for ONNX Runtime. Using CPU provider.")
                providers = ['CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = False
        session_options.enable_mem_reuse = True
        session_options.intra_op_num_threads = 2
        session_options.inter_op_num_threads = 2
        session_options.add_session_config_entry('session.arena_extend_strategy', 'kSameAsRequested')
        
        try:
            # Check multiple possible model locations - order is important
            possible_paths = [
                os.path.join(model_root, 'mask', 'model', 'FitDiT', 'humanparsing'),  # First check our known path
                os.path.join(model_root, 'FitDiT', 'humanparsing'),
                os.path.join(model_root, 'humanparsing')
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(os.path.join(path, 'parsing_atr.onnx')):
                    model_path = path
                    break
                    
            if model_path is None:
                raise FileNotFoundError("Could not find parsing models in any of: " + str(possible_paths))
                
            atr_path = os.path.join(model_path, 'parsing_atr.onnx')
            lip_path = os.path.join(model_path, 'parsing_lip.onnx')

            # Single initialization of sessions
            self.session = ort.InferenceSession(atr_path, sess_options=session_options, providers=providers)
            self.lip_session = ort.InferenceSession(lip_path, sess_options=session_options, providers=providers)
            
        except Exception as e:
            print(f"Failed to initialize human parsing model: {e}")
            raise
        

    def __call__(self, input_image):
#         torch.cuda.set_device(self.gpu_id)
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
