import matplotlib.pyplot as plt

# Data
ql_values = [16, 32, 64]
sequential_times = [219.953598, 426.429443, 870.881531]
stream_times = [1.393504, 2.900448, 33.405376]
more_fancy_times = [0.532352, 0.839200, 1.307680]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(ql_values, sequential_times, marker='o', label='Sequential Transfer and Execute')
plt.plot(ql_values, stream_times, marker='o', label='Stream Execute Time')
plt.plot(ql_values, more_fancy_times, marker='o', label='MoreFancy Execute Time')

# Labels and Title
plt.xlabel('ql')
plt.ylabel('Time (ms)')
plt.title('Execution Times for Different Methods')
plt.legend()
plt.grid(True)

# Show Plot
plt.show()
