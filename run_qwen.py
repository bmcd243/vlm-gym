import gymnasium as gym
import gymnasium_robotics
from PIL import Image
import torch
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# --- 1. Setup Output Directory ---
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Environment Setup ---
print("Setting up the Gymnasium environment...")
env = gym.make("FetchPush-v4", render_mode="rgb_array")

# --- 3. Define the Three Frame Variables ---
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

# --- 4. Load the VLM (Qwen2-VL-2B) ---
print("Loading Qwen2-VL-2B with 4-bit quantization...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

# This model is much lighter (2B params vs 7B)
model_id = "Qwen/Qwen2-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto", 
)

processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=1280*28*28)

print("VLM model loaded successfully.")

# --- 5. Prepare Inputs ---
# Convert numpy arrays to PIL Images
img1 = Image.fromarray(frame1)
img2 = Image.fromarray(frame2)
img3 = Image.fromarray(frame3)

# Save the frames for inspection
img1.save(os.path.join(OUTPUT_DIR, "frame1.png"))
img2.save(os.path.join(OUTPUT_DIR, "frame2.png"))
img3.save(os.path.join(OUTPUT_DIR, "frame3.png"))

user_prompt = (
    "These three images show a robot arm and a cube on a table, "
    "in chronological order. Describe the movement of the robot arm."
)

# Qwen uses a specific list-of-dicts format which is very clean for multi-image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img1},
            {"type": "image", "image": img2},
            {"type": "image", "image": img3},
            {"type": "text", "text": user_prompt},
        ],
    }
]

# Prepare inputs for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# process_vision_info handles extraction of images from the message list
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# --- 6. Generate and Save Description ---
print("Generating description from VLM...")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

output_path = os.path.join(OUTPUT_DIR, "vlm_description.txt")
with open(output_path, "w") as f:
    f.write(output_text)

print("\n" + "="*30)
print(f"   VLM DESCRIPTION (saved to {output_path}):")
print("="*30)
print(output_text)
print("\nScript finished.")