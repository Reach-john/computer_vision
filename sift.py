from PIL import Image
from os import system
from numpy import loadtxt, savetxt, hstack, arange, pi, cos, sin, array, linalg, zeros, dot, argsort, arccos
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

def match(desc1, desc2):
    desc1 = array(d/linalg.norm(d) for d in desc1)
    desc2 = array(d/linalg.norm(d) for d in desc2)

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = zeros((desc1_size[0], 1), 'int')
    desc2t = desc2.T
    for i in range(desc1_size):
        dotprods = dot(desc1[i, :], desc2t)
        dotprods = 0.999*dotprods
        indx = argsort(arccos(dotprods))
        if arccos(dotprods)[indx[0]] < dist_ratio*arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])
    return matchscores

def match_twosided(desc1, desc2):
    match12 = match(desc1, desc2)
    match21 = match(desc2, desc1)
    ndx_12 = match12.nonzero()[0]
    for n in ndx_12:
        if match21[int(match12[n])] != n:
            match12[n] = 0
    return match12
