import matplotlib.pyplot as plt
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib 
#%matplotlib inline
#from __future__ import division, print_function
from tf_unet import unet, util, image_util
from tf_unet import unet, util, image_util
#from __future__ import division, print_function
#get_ipython().magic('matplotlib inline')


#load training images
data_provider = image_util.ImageDataProvider(search_path = "train_data_pre/*", data_suffix="train.jpg", mask_suffix="mask.jpg")
#x_test, y_test = data_provider(1)

#load model
net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)

#train the model
trainer = unet.Trainer(net)
path = trainer.train(data_provider, "./temp", training_iters=10, epochs=10)
prediction = net.predict("./predicted/", data)
"""
unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
img = util.combine_img_prediction(data, label, prediction)
util.save_image(img, "prediction.jpg")"""


# In[8]:


# x_test, y_test = data_provider(1)
# print(y_test.shape)

# fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
# ax[0].imshow(x_test[0,...,0], aspect="auto")
# ax[1].imshow(y_test[0,...,1], aspect="auto")


# In[15]:


#load testing images 
data_provider = image_util.ImageDataProvider(search_path = "train_data_pre/*", data_suffix="train.jpg", mask_suffix="mask.jpg")


net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)


# In[21]:

#load one testing image
x_test, y_test = data_provider(1)


# In[22]:

#get prediction
prediction = net.predict("./temp/model.cpkt", x_test)


# In[23]:

unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))


# In[24]:



fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
#mask = prediction[0,...,1] > 0.9
mask = prediction[0,...,1] > 0.5
ax[2].imshow(mask, aspect="auto")
ax[0].set_title("Input")
ax[1].set_title("Ground truth")
ax[2].set_title("Prediction")
fig.tight_layout()


# In[25]:

#get accuracy

import numpy as np
mask_predict = np.resize(prediction, (1, 572, 572, 2))
a = np.ndarray.flatten(mask_predict)
b = np.ndarray.flatten(y_test)
a = 1. * (a>0.5)
c = np.add(a, b)




# In[26]:

for i in range(len(c)):
    if(c[i] == 2):
        a[i] = 1
    else:
        a[i] = 0


# In[27]:

for i in range(len(c)):
    if(c[i] > 0):
        b[i] = 1
    else:
        b[i] = 0


# In[28]:

inter = np.count_nonzero(a)
union = np.count_nonzero(b)


# In[29]:

print(float(inter/union))
