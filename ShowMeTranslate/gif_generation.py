import imageio
import os
images = []
for filename in os.listdir('./imgs'):
    images.append(imageio.imread('./imgs/'+filename))
imageio.mimsave('./gif/gif.gif', images)