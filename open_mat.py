from scipy.io import loadmat
import matplotlib.pyplot as plt
from pdb import set_trace

mat_data = loadmat('/mnt/c/Users/tnandi/Downloads/modulus/files/anl_domain.mat')
print(mat_data.keys())  

gray_matter = mat_data['gray_matter']  
set_trace()

plt.imshow(gray_matter, aspect='auto')
plt.colorbar()  
plt.show()


set_trace()

