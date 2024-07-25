import torch
import os

# Define the parameters
bs = 2    # Batch size
nh = 3    # Number of heads
qlen = 4  # Query length
klen = 3  # Key length
dim = 5   # Dimension of each code
bits = 4  # Number of bits per element

# Calculate K
K = 2 ** bits

# Initialize the tensors
query_codes = torch.randint(0, 2 ** bits, (bs, nh, qlen, dim), dtype=torch.uint8).to('cuda')
key_codes = torch.randint(0, 2 ** bits, (bs, nh, klen, dim), dtype=torch.uint8).to('cuda')
sd_lut = torch.randn((nh, K, K), dtype=torch.float32).to('cuda') + torch.ones((nh, K, K), dtype=torch.float32).to('cuda')

# Print tensor shapes to verify initialization
print(f"query_codes shape: {query_codes.shape}")
print(f"key_codes shape: {key_codes.shape}")
print(f"sd_lut shape: {sd_lut.shape}")

# Define the path for saving tensors
file_path = 'test_bench.pt'

# Save tensors to a file
torch.save({
    'query_codes': query_codes.cpu(),
    'key_codes': key_codes.cpu(),
    'sd_lut': sd_lut.cpu()
}, file_path)

print(f"Tensors saved to {file_path}")
