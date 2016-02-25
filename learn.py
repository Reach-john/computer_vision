from PIL import Image
import harris
from numpy import array
from pylab import *
import sift
import harris

imname = '3.jpg'
im = array(Image.open(imname).convert('L'))
sift.process_image(imname, '3.sift')
locs, desc = sift.read_features_from_file('3.sift')
harrisim = harris.compute_harris_response(im)
harris_coords = harris.get_harris_points(harrisim)
figure()
gray()
subplot(1, 3, 1)
imshow(im)
axis('off')
subplot(1, 3, 2)
sift.plot_feature(im, locs)
subplot(1, 3, 3)
harris.plot_harris_points(im, harris_coords)
show()




