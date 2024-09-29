import coremltools as ct
import numpy as np
import PIL.Image
from IPython.display import display

# Load a model whose input type is "Image".
model = ct.models.MLModel('/Users/bslota/IdeaProjects/summer-challenge-ia/pouet/240924/caipirinia/weights/best.mlpackage')

Height = 640  # use the correct input image height
Width = 640  # use the correct input image width


# Scenario 1: load an image from disk.
def load_image(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to)
    img_np = np.array(img).astype(np.float32)
    return img_np, img


# Load the image and resize using PIL utilities.
_, img = load_image('/Users/bslota/IdeaProjects/summer-challenge-ia/caipirinia/images/images_IMG_7010.jpg', resize_to=(Width, Height))
out_dict = model.predict({'image': img})

# Scenario 2: load an image from a NumPy array.
shape = (Height, Width, 3)  # height x width x RGB
data = np.zeros(shape, dtype=np.uint8)
# manipulate NumPy data
pil_img = PIL.Image.fromarray(data)
out_dict = model.predict({'image': pil_img})
display(out_dict['var_914'])
# out_dict.save("pouet.png")

