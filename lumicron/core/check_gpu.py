import cv2
import numpy as np

def verify_setup():
    print("--- Environment Hardware Audit ---")
    
    # Check OpenCV Version
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Check for OpenCL (GPU) Support
    gpu_available = cv2.ocl.haveOpenCL()
    print(f"Hardware Acceleration (OpenCL): {'ACTIVE' if gpu_available else 'INACTIVE'}")
    
    if gpu_available:
        # Attempt to enable OpenCL
        cv2.ocl.setUseOpenCL(True)
        print(f"OpenCL Device in use: {cv2.ocl.useOpenCL()}")
    else:
        print("WARNING: System is running on CPU only. Image processing may be throttled.")

if __name__ == "__main__":
    verify_setup()
