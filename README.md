# Image-Transformation

## Aim:

To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:

Anaconda - Python 3.7 

## Algorithm:

### Step1:

Import the necessary libraries and read the original image and save it as a image variable.

### Step2:

Translate the image using M=np.float32([[1,0,20],[0,1,50],[0,0,1]]) translated_img=cv2.warpPerspective(input_img,M,(cols,rows)).

### Step3:

Scale the image using M=np.float32([[1.5,0,0],[0,2,0],[0,0,1]]) scaled_img=cv2.warpPerspective(input_img,M,(cols,rows)).

### Step4:

Shear the image using M_x=np.float32([[1,0.2,0],[0,1,0],[0,0,1]]) sheared_img_xaxis=cv2.warpPerspective(input_img,M_x,(cols,rows)).

### Step5:

Reflection of image can be achieved through the code M_x=np.float32([[1,0,0],[0,-1,rows],[0,0,1]]) reflected_img_xaxis=cv2.warpPerspective(input_img,M_x,(cols,rows)).

### Step6:

Rotate the image using angle=np.radians(45) M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]) rotated_img=cv2.warpPerspective(input_img,M,(cols,rows)).

### Step7:

Crop the image using cropped_img=input_img[20:150,60:230].

### Step8:

Display all the Transformed images and end the program.


## Program:

```python

Developed By: Guru Prasad.B
Register Number: 212221230032

```

i) Image Translation

```python

import numpy as np
import cv2
import matplotlib.pyplot as plt

input_image = cv2.imread ("cars.png")

input_image = cv2. cvtColor (input_image, cv2. COLOR_BGR2RGB)

plt. axis('off')

plt.imshow(input_image)
plt. show()

rows, cols, dim = input_image.shape

M = np. float32([[1, 0, 50],
                 [0, 1, 100],
                 [0, 0, 1]])

translated_image = cv2.warpPerspective (input_image, M, (cols, rows))

plt.axis("off")

plt.imshow(translated_image)
plt.show()

```

ii) Image Scaling

```python

import numpy as np
import cv2
import matplotlib.pyplot as plt

input_image = cv2.imread ("cars.png")

plt. axis('off')

plt.imshow(input_image)

plt. show()

rows, cols, dim = input_image.shape

M = np. float32([[1.9, 0, 0],
                 [0, 1.9, 0],
                 [0, 0, 1]])

scaled_image = cv2.warpPerspective (input_image, M, (cols*2, rows*2))

plt.axis("off")
plt.imshow(scaled_image)
plt.show()

```

iii) Image shearing

```python

import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("cars.png")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()
rows,cols,dim = input_image.shape
M_x = np.float32([[1,0.3,0],
                 [0,1,0],
                 [0,0,1]])

M_y = np.float32([[1,0,0],
                 [0.3,1,0],
                 [0,0,1]])

sheared_xaxis = cv2.warpPerspective(input_image,M_x,(int(cols*1.5),int(rows*1.5)))
sheared_yaxis = cv2.warpPerspective(input_image,M_y,(int(cols*1.5),int(rows*1.5)))
plt.axis('off')
plt.imshow(sheared_xaxis)
plt.show()
plt.axis('off')
plt.imshow(sheared_yaxis)
plt.show()

```


iv) Image Reflection

```python

import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("cars.png")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()
rows,cols,dim = input_image.shape
M_x = np.float32([[1,0,0],
                 [0,-1,rows],
                 [0,0,1]])

M_y = np.float32([[-1,0,cols],
                 [0,1,0],
                 [0,0,1]])

reflected_xaxis = cv2.warpPerspective(input_image,M_x,(int(cols),int(rows)))
reflected_yaxis = cv2.warpPerspective(input_image,M_y,(int(cols),int(rows)))
plt.axis('off')
plt.imshow(reflected_xaxis)
plt.show()
plt.axis('off')
plt.imshow(reflected_yaxis)
plt.show()

```


v) Image Rotation

```python

import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("cars.png")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()
rows,cols,dim = input_image.shape
angle = np.radians(20)

M = np.float32([[np.cos(angle),-(np.sin(angle)),0],
               [np.sin(angle),np.cos(angle),0],
               [0,0,1]])

rotated_image = cv2.warpPerspective(input_image,M,(int(cols),int(rows)))
plt.axis('off')
plt.imshow(rotated_image)
plt.show()

```


vi) Image Cropping

```python

import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("cars.png")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()
rows,cols,dim = input_image.shape

cropped_image = input_image[500:600,600:650]

plt.axis('off')
plt.imshow(cropped_image)
plt.show()

```


## Output:

### i) Image Translation:

![o1](https://user-images.githubusercontent.com/95342910/230917709-32b3ad83-57ee-4148-901c-36da2abb7d71.png)

### ii) Image Scaling:

![o2](https://user-images.githubusercontent.com/95342910/230917717-a91a3d31-5010-4553-8a89-a66faa6d70a2.png)

### iii)Image shearing:

![o3](https://user-images.githubusercontent.com/95342910/230917726-10ba3a4e-1498-4ea5-a85d-39c411524f0a.png)

### iv) Image Reflection:

![o4](https://user-images.githubusercontent.com/95342910/230917729-be3bc0de-922e-47ac-8607-1f6065d12d70.png)


### v) Image Rotation:

![o5](https://user-images.githubusercontent.com/95342910/230917734-671beb61-0285-41cc-b90d-f6c4ee06ecb2.png)


### vi) Image Cropping:

![o6](https://user-images.githubusercontent.com/95342910/230917747-45b937bc-4339-48e5-919f-f3b3bf7b4c8e.png)


## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.


