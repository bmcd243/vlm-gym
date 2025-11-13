import gymnasium as gym
import gymnasium_robotics
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
import warnings
import os

# --- 1. Setup Output Directory ---
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Environment Setup ---
print("Setting up the Gymnasium environment...")
# We use 'rgb_array' so we can capture frames
env = gym.make("FetchPush-v4", render_mode="rgb_array")

# --- 3. Define the Three Frame Variables ---
# We use env.render() to get the pixel data (numpy array)
observation, info = env.reset(seed=42)
frame1 = env.render()
print(f"Frame 1 captured (shape: {frame1.shape})")

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
frame2 = env.render()
print(f"Frame 2 captured (shape: {frame2.shape})")

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
frame3 = env.render()
print(f"Frame 3 captured (shape: {frame3.shape})")

env.close()

# --- 4. Load the VLM (With 4-bit Quantization) ---
print("Loading VLM with 4-bit quantization...")

# Define the quantization config (CRITICAL FOR 8GB GPU)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

processor = LlavaNextProcessor.from_pretrained(model_id)

# Load model with the config
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config, 
    device_map="auto",
    low_cpu_mem_usage=True,
)

print("VLM model loaded successfully (4-bit).")
device = model.device 

# --- 5. Prepare Inputs and Save Frames ---
# Convert numpy arrays to PIL Images
img1 = Image.fromarray(frame1)
img2 = Image.fromarray(frame2)
img3 = Image.fromarray(frame3)

# Save the frames
img1.save(os.path.join(OUTPUT_DIR, "frame1.png"))
img2.save(os.path.join(OUTPUT_DIR, "frame2.png"))
img3.save(os.path.join(OUTPUT_DIR, "frame3.png"))
print("Frames saved to /app/output/")

user_prompt = (
    "These three images show a robot arm and a cube on a table, "
    "in chronological order. What is the robot arm doing?"
)
full_vlm_prompt = f"[INST] <image>\n<image>\n<image>\n{user_prompt} [/INST]"

inputs = processor(
    text=full_vlm_prompt,
    images=[img1, img2, img3],
    return_tensors="pt"
).to(device)

# --- 6. Generate and Save Description ---
print("Generating description from VLM...")
output_ids = model.generate(**inputs, max_new_tokens=100)

vlm_description_raw = processor.batch_decode(
    output_ids, skip_special_tokens=True
)[0]

vlm_description = vlm_description_raw.split("[/INST]")[-1].strip()

output_path = os.path.join(OUTPUT_DIR, "vlm_description.txt")
with open(output_path, "w") as f:
    f.write(vlm_description)

print("\n" + "="*30)
print(f"   VLM DESCRIPTION (saved to {output_path}):")
print("="*30)
print(vlm_description)
print("\nScript finished.")