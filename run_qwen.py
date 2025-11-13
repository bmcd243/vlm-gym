import gymnasium as gym
import gymnasium_robotics
from PIL import Image
import torch
import os
import numpy as np  # Needed for defining specific actions
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# --- 1. Setup Output Directory ---
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Environment Setup ---
print("Setting up the Gymnasium environment...")
env = gym.make("FetchPush-v4", render_mode="rgb_array")
observation, info = env.reset(seed=42)

# --- 3. Define Specific Actions (The "Physics" Part) ---
# Action format: [x_vel, y_vel, z_vel, gripper_vel]
# Values are typically between -1.0 and 1.0

# Example: Move RIGHT (positive Y), slightly DOWN (negative Z) to get near table
# We keep X (forward/back) mostly static.
specific_action = np.array([0.0, 0.8, -0.2, 0.0], dtype=np.float32)

frames = []
num_frames = 8  # How many frames you want to generate

print(f"Simulating {num_frames} frames of specific movement...")

# Capture initial state
frames.append(Image.fromarray(env.render()))

for i in range(num_frames):
    # Apply the SAME action repeatedly to create a continuous trajectory
    observation, reward, terminated, truncated, info = env.step(specific_action)
    
    # Render and store as PIL Image
    frame_array = env.render()
    img = Image.fromarray(frame_array)
    frames.append(img)
    
    # Save individual frames to disk to verify movement
    img.save(os.path.join(OUTPUT_DIR, f"frame_{i}.png"))

env.close()
print(f"Captured {len(frames)} frames.")

# --- 4. Load the VLM (Qwen2-VL-2B) ---
print("Loading Qwen2-VL-2B...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

model_id = "Qwen/Qwen2-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto", 
)

processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=1280*28*28)

# --- 5. Prepare Inputs ---

user_prompt = (
    "These images show a robot arm moving over time. "
    "Describe the direction the arm is moving."
)

# Dynamically build the message content based on how many frames we generated
content_list = []
for img in frames:
    content_list.append({"type": "image", "image": img})

# Add the text prompt at the end
content_list.append({"type": "text", "text": user_prompt})

messages = [
    {
        "role": "user",
        "content": content_list,
    }
]

# --- 6. Generate Description ---
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

print("Generating description...")
generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("\n" + "="*30)
print("   VLM DESCRIPTION:")
print("="*30)
print(output_text)