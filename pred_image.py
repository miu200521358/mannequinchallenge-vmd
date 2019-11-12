import numpy as np
from matplotlib import pyplot as plt

pred_depth = np.loadtxt("E:/MMD/MikuMikuDance_v926x64/Work/201805_auto/04/yoiyoi/yoiyoi_edit_20191110_124137/yoiyoi_edit_json_20191111_161727_depth/depth/depth_pred/pred_000000000000.txt")

# Plot result
plt.cla()
plt.clf()
ii = plt.imshow(pred_depth, interpolation='nearest')
plt.colorbar(ii)

plt.show()



disparity = 1. / pred_depth
disparity = disparity / np.max(disparity)
disparity = np.tile(np.expand_dims(disparity, axis=-1), (1, 1, 3))
saved_imgs = np.concatenate((saved_img, disparity), axis=1)
saved_imgs = (saved_imgs*255).astype(np.uint8)

