from numpy import  *

def denoise(im, u_init, tolerance = 0.1, tau = 0.125, tv_weight = 100):
    m, n = im.shape
    u = u_init
    px = im
    py = im
    error = 1
    while(error > tolerance):
        uold = u
        grandux = roll(u, -1, axis=1) - u
        granduy = roll(u, -1, axis=0) - u
        pxnew = px + (tau / tv_weight)*grandux
        pynew = py + (tau / tv_weight)*granduy
        normnew = maximum(1, sqrt(pxnew**2 + pynew**2))
        px = pxnew/normnew
        py = pynew/normnew
        rxpx = roll(px, 1, axis=1)
        rypy = roll(py, 1, axis=0)
        divp = (px-rxpx) + (py-rypy)
        u = im + tv_weight*divp
        error = linalg.norm(u-uold) / sqrt(n*m)
    return u, im-u
