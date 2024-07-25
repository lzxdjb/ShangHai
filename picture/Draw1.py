import matplotlib.pyplot as plt

# Data
ql_nh_values = [4, 8, 16]
sequential = [75.158401, 231.537628, 872.630005]
stream = [0.382784, 104.765282, 636.395386]
morefancy = [0.404832, 224.317978, 915.603516]

# Create a plot
plt.figure(figsize=(10, 6))

# Plot each line
plt.plot(ql_nh_values, sequential, label='Sequential Transfer and Execute (ms)', marker='o')
plt.plot(ql_nh_values, stream, label='Stream Execute (ms)', marker='o')
plt.plot(ql_nh_values, morefancy, label='MoreFancy Execute (ms)', marker='o')

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
