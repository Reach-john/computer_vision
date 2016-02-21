from PIL import Image
import  numpy
from pylab import *
from scipy.ndimage import filters
import rof

im = array(Image.open("1.jpg").convert('L'))
u, t = rof.denoise(im, im)
subplot(1, 2, 1)
imshow(im)
gray()
subplot(1, 2, 2)
imshow(u)
gray()
show()