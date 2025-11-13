import gymnasium as gym
import gymnasium_robotics
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig # <--- Add BitsAndBytesConfig
import torch
import warnings
import os

# --- 1. Setup Output Directory ---
# We will save all output to /app/output
# This path is *inside* the Docker container
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Environment Setup ---
print("Setting up the Gymnasium environment...")
env = gym.make("FetchPush-v4", render_mode="rgb_array")

# --- 3. Define the Three Frame Variables ---
observation, info = env.reset(seed=42)
frame1 = env.render()

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
frame2 = env.render()

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
frame3 = env.render()

env.close()

# --- 4. Load the VLM (LLaVA) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    warnings.warn("CUDA (GPU) not available. Running on CPU.")
else:
    print("CUDA is available! Running on GPU.")

print("Loading VLM with 4-bit quantization...")

# Define the quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

processor = LlavaNextProcessor.from_pretrained(model_id)

# Load model with the config
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config, # <--- This does the magic
    device_map="auto",                       # <--- Automatically puts it on the GPU
    low_cpu_mem_usage=True,
)

print("VLM model loaded successfully (4-bit).")


# --- 5. Prepare Inputs and Save Frames ---
img1 = Image.fromarray(frame1)
img2 = Image.fromarray(frame2)
img3 = Image.fromarray(frame3)

# Save the frames instead of showing them
img1.save(os.path.join(OUTPUT_DIR, "frame1.png"))
img2.save(os.path.join(OUTPUT_DIR, "frame2.png"))
img3.save(os.path.join(OUTPUT_DIR, "frame3.png"))
print("Frames saved to /app/output/")

user_prompt = (
    "These three images show a robot arm and a cube on a table, "
    "in chronological order. What is the robot arm doing?"
)
full_vlm_prompt = f"[INST] <image>\n<image>\<image>\n{user_prompt} [/INST]"

inputs = processor(
    text=full_vlm_prompt, 
    images=[img1, img2, img3], 
    return_tensors="pt"
).to(device) # Use the device we grabbed earlier

# --- 6. Generate and Save the VLM's Description ---
print("Generating description from VLM...")
output_ids = model.generate(**inputs, max_new_tokens=100)
vlm_description_raw = processor.batch_decode(
    output_ids, skip_special_tokens=True
)[0]
vlm_description = vlm_description_raw.split("[/INST]")[-1].strip()

# Save the description to a text file
output_path = os.path.join(OUTPUT_DIR, "vlm_description.txt")
with open(output_path, "w") as f:
    f.write(vlm_description)

print("\n" + "="*30)
print(f"   VLM DESCRIPTION (saved to {output_path}):")
print("="*30)
print(vlm_description)
print("\nScript finished.")