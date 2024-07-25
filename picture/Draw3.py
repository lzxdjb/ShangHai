import matplotlib.pyplot as plt

# Data
ql_nh_values = [4, 8, 16, 32]
sequential = [73.896317, 211.430496, 790.458130, 3732.801758]
stream = [0.393440, 99.160835, 590.692383, 3217.209229]
morefancy = [0.424800, 210.150589, 825.458679, 3949.333740]
ultra = [0.417376, 0.956000, 2.335008, 7.954624]

# Create a plot
plt.figure(figsize=(10, 6))

# Plot each line
plt.plot(ql_nh_values, sequential, label='Sequential Transfer and Execute (ms)', marker='o')
plt.plot(ql_nh_values, stream, label='Stream Execute (ms)', marker='o')
plt.plot(ql_nh_values, morefancy, label='MoreFancy Execute (ms)', marker='o')
plt.plot(ql_nh_values, ultra, label='Ultra Execute (ms)', marker='o')

# Add title and labels
plt.title('Execution Time vs. ql and nh Values')
plt.xlabel('ql = nh')
plt.ylabel('Time (ms)')

# Add legend
plt.legend()

# Set logarithmic scale for y-axis to better visualize data
plt.yscale('log')

# Show the plot
plt.grid(True)
plt.show()
