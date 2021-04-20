import scipy.io
import matplotlib.pyplot as plt
import numpy as np

n_lighting = 9

base_img_tensor = scipy.io.loadmat('Test_Data_128/input.mat')
base_img_tensor = np.expand_dims(base_img_tensor['input'], 0)

lighting_tensor = None
sets_v = list()
for i in range(1, n_lighting + 1):
    sets_v.append(np.expand_dims(scipy.io.loadmat('Test_Data_128/output_' + str(i) + '.mat')['output'], 0))
lighting_tensor = np.concatenate(sets_v)

cs_base = base_img_tensor.shape
cs_lighting = lighting_tensor.shape

base_img_tensor = base_img_tensor.reshape(1, -1, cs_base[3], cs_base[4])
lighting_tensor = lighting_tensor.reshape(n_lighting, -1, cs_lighting[3], cs_lighting[4])

base_img_tensor = base_img_tensor*(base_img_tensor > 0)
lighting_tensor = lighting_tensor*(lighting_tensor > 0)

base_img_tensor = 2*(base_img_tensor / np.amax(base_img_tensor, axis=(2,3), keepdims=True))-1
lighting_tensor = 2*(lighting_tensor / np.amax(lighting_tensor, axis=(2,3), keepdims=True))-1

base_img_tensor = np.expand_dims(base_img_tensor, 2)
lighting_tensor = np.expand_dims(lighting_tensor, 2)

# Indices: Lighting Condition, Augmented Bottle Example, Channel, Pixels, Pixels
print(base_img_tensor.shape)
print(lighting_tensor.shape)

np.save('test_base_img_arr.npy', base_img_tensor)
np.save('test_lighting_arr.npy', lighting_tensor)
