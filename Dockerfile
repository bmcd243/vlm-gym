# 1. Start from NVIDIA CUDA image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# --- NEW: Force MuJoCo to use EGL (Headless GPU Rendering) ---
ENV MUJOCO_GL=egl

# 3. Install Python, pip, and GRAPHICS LIBRARIES
# We added: libgl1-mesa-glx, libglew-dev, libosmesa6-dev, etc.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# 4. Set up working directory
WORKDIR /app

# 5. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy code and setup output
COPY run_vlm.py .
RUN mkdir -p /app/output

# 7. Default command
CMD ["python3", "run_vlm.py"]