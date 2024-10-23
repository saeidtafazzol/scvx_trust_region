import numpy as np

def coe_from_sv(R, V, mu):
    """
    This function computes the classical orbital elements (coe)
    from the state vector (R,V) using a specified algorithm.
    
    Parameters:
    - mu: gravitational parameter (km^3/s^2)
    - R: position vector in the geocentric equatorial frame (km)
    - V: velocity vector in the geocentric equatorial frame (km/s)
    
    Returns:
    - coe: vector of orbital elements [a, e, incl, RA, w, TA]
    """
    eps = 1.e-10
    pi = np.pi

    r = np.linalg.norm(R)
    v = np.linalg.norm(V)
    vr = np.dot(R, V) / r
    H = np.cross(R, V)
    h = np.linalg.norm(H)
    incl = np.arccos(H[2] / h)
    N = np.cross([0, 0, 1], H)
    n = np.linalg.norm(N)

    if incl != 0:
        RA = np.arccos(N[0] / n)
        if N[1] < 0:
            RA = 2 * pi - RA
    else:
        RA = 0

    E = (1 / mu) * ((v**2 - mu / r) * R - r * vr * V)
    e = np.linalg.norm(E)

    if incl != 0:
        if e > eps:
            w = np.arccos(np.dot(N, E) / n / e)
            if E[2] < 0:
                w = 2 * pi - w
        else:
            w = 0
    else:
        if e > eps:
            w = np.arccos(E[0] / e)
            if E[1] < 0:
                w = 2 * pi - w
        else:
            w = 0

    if incl != 0:
        if e > eps:
            TA = np.arccos(np.dot(E, R) / e / r)
            if vr < 0:
                TA = 2 * pi - TA
        else:
            TA = np.arccos(np.dot(N, R) / n / r)
            if R[2] < 0:
                TA = 2 * pi - TA
    else:
        if e > eps:
            TA = np.arccos(np.dot(E, R) / e / r)
            if vr < 0:
                TA = 2 * pi - TA
        else:
            TA = np.arccos(R[0] / r)
            if R[1] < 0:
                TA = 2 * pi - TA

    a = h**2 / mu / (1 - e**2)

    coe = np.array([a, e, incl, RA, w, TA])
    return coe

def mee_from_sv(R, V, mu):
    """
    Converts state vector to modified equinoctial elements (MEE).
    
    Parameters:
    - R: numpy array, position vector in km.
    - V: numpy array, velocity vector in km/s.
    - mu: float, gravitational parameter in km^3/s^2.
    
    Returns:
    - mee: numpy array, modified equinoctial elements [p, f, g, h, k, L].
    """
    coe = coe_from_sv(R, V, mu)
    a, e, incl, RA, w, TA = coe

    p = a * (1 - e**2)
    f = e * np.cos(w + RA)
    g = e * np.sin(w + RA)
    h = np.tan(incl / 2) * np.cos(RA)
    k = np.tan(incl / 2) * np.sin(RA)
    L = RA + w + TA

    mee = np.array([p, f, g, h, k, L])
    return mee

def sv_from_mee(mee, mu):
    """
    Converts modified equinoctial elements (MEE) to state vector.
    
    Parameters:
    - mee: numpy array, modified equinoctial elements [p, f, g, h, k, L].
    - mu: float, gravitational parameter in km^3/s^2.
    
    Returns:
    - R: numpy array, position vector in km.
    - V: numpy array, velocity vector in km/s.
    """
    p, f, g, h, k, L = mee

    s_L = np.sin(L)
    c_L = np.cos(L)
    a_s = h**2 - k**2
    s_s = 1 + h**2 + k**2
    w = 1 + f * c_L + g * s_L
    r = p / w
    s_mu_p = np.sqrt(mu / p)

    cart = np.array([
        r / s_s * (c_L + a_s * c_L + 2 * h * k * s_L),
        r / s_s * (s_L - a_s * s_L + 2 * h * k * c_L),
        2 * r / s_s * (h * s_L - k * c_L),
        -1 / s_s * s_mu_p * (s_L + a_s * s_L - 2 * h * k * c_L + g - 2 * f * h * k + a_s * g),
        -1 / s_s * s_mu_p * (-c_L + a_s * c_L + 2 * h * k * s_L - f + 2 * g * h * k + a_s * f),
        2 / s_s * s_mu_p * (h * c_L + k * s_L + f * h + g * k)
    ])
    
    return cart



def sph_from_cart(cart):
    x = cart[0]
    y = cart[1]
    z = cart[2]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)

    sph = np.array([r, theta, phi])

    if len(cart) > 3:
        dx = cart[3]
        dy = cart[4]
        dz = cart[5]

        dr = (2*x*dx + 2*y*dy + 2*z*dz) / (2 * (x**2 + y**2 + z**2)**(1/2))
        dtheta = -(dz/(x**2 + y**2 + z**2)**(1/2) - (z * (2*x*dx + 2*y*dy + 2*z*dz)) / (2 * (x**2 + y**2 + z**2)**(3/2))) / np.sqrt(1 - z**2 / (x**2 + y**2 + z**2))
        dphi = (dy / x - (y * dx) / x**2) / (y**2 / x**2 + 1)

        sph = np.concatenate((sph, [dr, dtheta, dphi]))

    return sph


def cart_from_sph(sph):
    r = sph[0]
    theta = sph[1]
    phi = sph[2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    cart = np.array([x, y, z])

    if len(sph) > 3:
        dr = sph[3]
        dtheta = sph[4]
        dphi = sph[5]

        dx = np.cos(phi) * np.sin(theta) * dr + np.cos(phi) * np.cos(theta) * r * dtheta - np.sin(phi) * np.sin(theta) * r * dphi
        dy = np.sin(phi) * np.sin(theta) * dr + np.cos(phi) * np.sin(theta) * r * dphi + np.cos(theta) * np.sin(phi) * r * dtheta
        dz = np.cos(theta) * dr - np.sin(theta) * r * dtheta

        cart = np.concatenate((cart, [dx, dy, dz]))

    return cart