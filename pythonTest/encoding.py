import torch

# Define the dimensions
bs = 2    # Batch size
nh = 3    # Number of heads
qlen = 64  # Query length
klen = 64  # Key length
dim = 5   # Dimension of each code
bits = 4  # Number of bits per element

K = 2 ** bits


# Initialize the tensors
query_codes = torch.randint(0, 2 ** bits,(bs, nh, qlen, dim), dtype=torch.uint8)
key_codes = torch.randint(0, 2 ** bits,(bs, nh, klen, dim), dtype=torch.uint8)
sd_lut = torch.randn((nh, K, K), dtype=torch.float32) + torch.ones((nh, K, K), dtype=torch.float32)
sd_store = torch.randint(0, 2 ** bits,(bs, nh, qlen, klen), dtype=torch.float32)

import time
start_time = time.time()

query_codes = query_codes.to('cuda')
key_codes = key_codes.to('cuda')
sd_lut = sd_lut.to('cuda')
sd_store = sd_store.to('cuda')
# print(sd_lut)
# exit()

# print(sd_lut.shape)



##### debug
# query_codes = torch.ones((bs, nh, qlen, dim), dtype=torch.uint8).to('cuda')
# key_codes = torch.ones((bs, nh, klen, dim), dtype=torch.uint8).to('cuda')
# cartesian_index = torch.ones((bs, nh, qlen, klen , 2 , 1), dtype=torch.uint8).to('cuda')
# sd_lut = torch.full((nh, K, K) , 3 , dtype=torch.uint8).to('cuda')
# sd_store = torch.full((bs, nh, qlen, klen), 2, dtype=torch.uint8).to('cuda')

# print(sd_lut.shape)
# exit()



def vector_to_scalar(tensor):

    tensor = tensor.to(torch.uint8)  
    binary_str = ''.join(f'{x:0{bits}b}' for x in tensor)    
    scalar_value = int(binary_str, 2)

    return scalar_value



def MyMean(tensor):
    temp = 0
    for i in range(dim):
        temp += tensor[i]
    temp /= dim
    return temp


for b in range(bs):
    for n in range(nh):

        # print("test = " , query_codes[b, n, :, :].shape)
        # exit()
        for q in range(qlen):
            q_vector = query_codes[b, n, q, :]
            tensor_float = q_vector.to(torch.float32)
            tensor_float = tensor_float.mean()
            # q_encoding_value = MyMean(q_vector)
            q_encoding_value = tensor_float.to(torch.uint8)

            # print(q_encoding_value)
            # exit()

            # print(q_vector)
            # print(vector_to_scalar(q_vector ))
            # exit()

            for k in range(klen):
                k_vector = key_codes[b, n, k, :]
                tensor_float = k_vector.to(torch.float32)
                tensor_float = tensor_float.mean()
                k_encoding_value = tensor_float.to(torch.uint8)

                # # k_encoding_value = MyMean(k_vector)
                # print(q_encoding_value)
                # print(k_encoding_value)
                # exit()

                find_value = sd_lut[n , q_encoding_value.item() , q_encoding_value.item()]

                sd_store[b , n , q , k] = find_value



                # print(find_value.item())
                # exit()


                ##### find element in LUT:


                # cartesian_index[b , n , q , k , 0 , 0] = q_encoding_value
                # cartesian_index[b , n , q , k , 1 , 0] = k_encoding_value


end_time = time.time()
elapsed_time_ms = (end_time - start_time) * 1000
print(f"Elapsed time: {elapsed_time_ms:.2f} ms")



# print(sd_store.shape)
# print(sd_store)
                

# exit()
       





