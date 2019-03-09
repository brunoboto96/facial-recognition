import numpy as np
from keras.preprocessing import image
from keras.models import load_model



test_image = image.load_img('test.jpg', target_size = (68,77))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)/255.
model = load_model('CNN/model.h5')
result = model.predict(test_image)
#trainingset.class_indices
print(np.argmax(result))
