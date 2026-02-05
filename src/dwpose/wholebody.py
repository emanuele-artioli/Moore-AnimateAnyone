# https://github.com/IDEA-Research/DWPose
from pathlib import Path

import cv2
import numpy as np
# Try importing torch first so CUDA libraries are loaded (workaround for some ONNX Runtime CUDA init failures)
try:
    import torch  # noqa: F401
except Exception:
    torch = None
import onnxruntime as ort

from .onnxdet import inference_detector
from .onnxpose import inference_pose

ModelDataPathPrefix = Path("./pretrained_weights")


class Wholebody:
    def __init__(self, device="cuda:0"):
        """Create ONNXRuntime sessions for detection and pose models.

        Attempts to use the CUDAExecutionProvider; if initialization fails (for
        example due to missing system CUDA libraries like libcurand), this will
        retry after importing torch (which often brings in CUDA libraries) and
        then gracefully fall back to CPUExecutionProvider with a clear warning
        and actionable instructions.
        """

        def _create_session(onnx_path, try_cuda=True):
            if not try_cuda:
                return ort.InferenceSession(path_or_bytes=onnx_path, providers=["CPUExecutionProvider"])

            # First try: attempt CUDA execution provider
            try:
                return ort.InferenceSession(path_or_bytes=onnx_path, providers=["CUDAExecutionProvider"])
            except Exception as e:
                # Second try: import torch (if available) to ensure CUDA libs are loaded
                try:
                    import torch  # noqa: F401
                except Exception:
                    pass
                try:
                    return ort.InferenceSession(path_or_bytes=onnx_path, providers=["CUDAExecutionProvider"])
                except Exception as e2:
                    import warnings
                    warnings.warn(
                        "CUDA Execution Provider failed to initialize for ONNXRuntime; falling back to CPU. "
                        f"Error: {e2}. "
                        "To enable GPU support, ensure you have a matching `onnxruntime-gpu` wheel "
                        "for your CUDA driver (or install via `pip uninstall onnxruntime onnxruntime-gpu && "
                        "pip install 'optimum[onnxruntime-gpu]'`), and verify system CUDA libraries such as `libcurand` "
                        "are available. See: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements"
                    )
                    return ort.InferenceSession(path_or_bytes=onnx_path, providers=["CPUExecutionProvider"])

        onnx_det = ModelDataPathPrefix.joinpath("DWPose/yolox_l.onnx")
        onnx_pose = ModelDataPathPrefix.joinpath("DWPose/dw-ll_ucoco_384.onnx")

        use_cuda = device != "cpu"
        self.session_det = _create_session(onnx_det, try_cuda=use_cuda)
        self.session_pose = _create_session(onnx_pose, try_cuda=use_cuda)

    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3
        ).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        return keypoints, scores
