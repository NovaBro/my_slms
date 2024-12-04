"""
Able to download models and run inference.
Theoretically able to finetune,through this tutorial: https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md
"""
from mlx_lm import load, generate


# model, tokenizer = load("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
# model, tokenizer = load("bigcode/starcoderbase-1b") #NOTE: non-instruct model, so human readable prompts yield poor results.

# response = generate(model, tokenizer, prompt="Print all primes between 1 and n", verbose=True)


from datasets import load_dataset
# raw_data = load_dataset('smangrul/hf-stack-v1', )
# print(raw_data['train']['repo_id'][:4])

DATASET = "smangrul/hf-stack-v1"  # Dataset on the Hugging Face Hub
dataset = load_dataset(
    DATASET,
    data_dir="data",
    split="train",
    streaming=True,
)

valid_data = dataset.take(4000)
train_data = dataset.skip(4000)
# train_data = train_data.shuffle(buffer_size=5000, seed=SEED)

print(valid_data['repo_id'])
print(train_data)