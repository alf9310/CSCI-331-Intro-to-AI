import matplotlib.pyplot as plt
import numpy as np
import math

# Read file
filename = "C:\\Users\\alynf\\OneDrive\\Documents\\CSCI-331-Intro-to-AI\\project1\\centipede_score.txt"
lines = np.genfromtxt(filename, delimiter=",")
int_lines = lines.astype(np.int32)
unzip = list(zip(*int_lines))
print(unzip)

# Plot histogram
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
plt.ylabel("Number of Runs")
axs[0].hist(unzip[0], color = "blue")
axs[0].set_title('Score')
axs[1].hist(unzip[1], color = "red")
axs[1].set_title("Levels Reached")
x_ticks = np.arange(min(unzip[0]) - 2, max(unzip[0]) + 3, 1)
axs[0].set_xticks(x_ticks)
plt.show()