# 1. Start from an official NVIDIA CUDA image.
# This image has Ubuntu 22.04, CUDA 12.1, and cuDNN 8 pre-installed.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. Set environment variables to prevent prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 3. Install Python, pip, and other system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Set up a working directory
WORKDIR /app

# 5. Copy the requirements file and install packages
# This is done before copying the code for better Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your Python script into the container
COPY run_vlm.py .

# 7. Create the output directory we referenced in the script
RUN mkdir -p /app/output

# 8. Set the default command to run when the container starts
CMD ["python3", "run_vlm.py"]