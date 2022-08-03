import matplotlib.pyplot as plt
import matplotlib.colors

matrix = [[0, 2, 2, 3, 4, 5],
[6, 7, 8, 9, 10, 11],
[12, 13, 14, 15, 16, 17],
[18, 19, 20, 21, 22, 23],
[24, 25, 26, 27, 0, 0]
]
matrix = np.array(matrix)
custom_cmap = ["#8fd3ff", "#eaaded", "#a884f3", "#905ea9", "#6b3e75", "#45293f", "#6e2727", "#547e64", "#374e4a", "#cddf6c", "#91db69", "#1ebc73", "#239063", "#b33831", "#ea4f36", "#b2ba90", "#fbff86", "#165a4c", "#d5e04b", "#a2a947", "#676633", "#4c3e24", "#f79617", "#30e1b9", "#0eaf9b", "#0b8a8f", "#0b5e65" ]
custom_cmap = matplotlib.colors.ListedColormap(custom_cmap)

plt.imshow(matrix, custom_cmap)
plt.show()