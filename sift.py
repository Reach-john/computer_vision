from PIL import Image
from os import system
from numpy import loadtxt, savetxt, hstack, arange, pi, cos, sin
from pylab import plot, imshow, axis


def process_image(imagename, resultname, parames="--edge-thresh 10 --peak-thresh 5"):
    if imagename[-3:] != 'pgm':
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'
    cmd = str("sift "+ imagename+ " --output="+resultname+" "+parames)
    system(cmd)
    print 'processed ', imagename, ' to ', resultname
    return

def read_features_from_file(filenames):
    f = loadtxt(filenames)
    return f[:, :4], f[:, 4:]

def write_features_to_file(filename, locs, desc):
    savetxt(filename, hstack((locs, desc)))
    return

def plot_feature(im, locs, circle=False):
    def draw_circle(c, r):
        t = arange(0, 1.01, .01) * 2 * pi
        x = r*cos(t) + c[0]
        y = r*sin(t) + c[1]
        plot(x, y, 'b', linewidth=2)
    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plot(locs[:, 0], locs[:, 1], 'ob')
    axis('off')
    return
