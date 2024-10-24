import numpy as np
import coordinate_conversion
earth_to_mars = {}

earth_to_mars['name'] = 'Earth to Mars'
earth_to_mars['origin'] = 'Earth'
earth_to_mars['destination'] = 'Mars'
earth_to_mars['AU'] = 1.496e+8  # km
earth_to_mars['AUm'] = earth_to_mars['AU'] * 1e3  # meters
earth_to_mars['TU'] = 31536000 / (2 * np.pi)  # time unit (s)

# Gravitational parameter of the Sun (mu)
earth_to_mars['mu'] = 132712440018 / (earth_to_mars['AU'] ** 3) * (earth_to_mars['TU'] ** 2)

# Spacecraft parameters
earth_to_mars['m_0'] = 659.3  # initial mass (kg)
earth_to_mars['g0'] = 9.8065  # gravitational acceleration (m/s^2)
earth_to_mars['I_sp'] = 3300  # specific impulse (s)

# Thrust calculations
earth_to_mars['T_max'] = 0.55 / earth_to_mars['AUm'] * (earth_to_mars['TU'] ** 2)

# Effective exhaust velocity
earth_to_mars['c'] = earth_to_mars['I_sp'] * earth_to_mars['g0'] / earth_to_mars['AUm'] * earth_to_mars['TU']

# Initial position and velocity vectors (in AU and AU/TU)
# earth_to_mars['r_i'] = np.array([-140699693, -51614428, 980]) / earth_to_mars['AU']
# earth_to_mars['v_i'] = np.array([9.774596, -28.07828, 4.337725e-4]) / earth_to_mars['AU'] * earth_to_mars['TU']

# Final position and velocity vectors (in AU and AU/TU)
# earth_to_mars['r_f'] = np.array([-172682023, 176959469, 7948912]) / earth_to_mars['AU']
# earth_to_mars['v_f'] = np.array([-16.427384, -14.860506, 9.21486e-2]) / earth_to_mars['AU'] * earth_to_mars['TU']
earth_to_mars['rho0'] = 0.03  # initial mass (kg)
earth_to_mars['rho1'] = 0.4  # gravitational acceleration (m/s^2)
earth_to_mars['rho2'] = 0.9  # specific impulse (s)

earth_to_mars['C'] = 5.0
# Mode of operation
earth_to_mars['mode'] = 'mee'

earth_to_mars['t_f_days'] = 253
earth_to_mars['t_f'] = earth_to_mars['t_f_days'] * 24 * 3600 / earth_to_mars['TU']

earth_to_mars['N'] = 600
earth_to_mars['dt'] = earth_to_mars['t_f']/(earth_to_mars['N']-1)



earth_to_mars['x_i'] = np.array([1.0, 0.0 ,0.0 ,0.0 , 0.0 ,0.0])
earth_to_mars['x_f'] = np.array([1.5237, 0.0, 0.0, 0.0, 1.6146e-2, 3.1416])


earth_to_mars['init_traj'] = np.linspace(earth_to_mars['x_i'],earth_to_mars['x_f'],earth_to_mars['N'])
earth_to_mars['m_f_estimated'] = 500

earth_to_mars['z_0'] = np.log(earth_to_mars['m_0'])
earth_to_mars['z_f_estimated'] = np.log(earth_to_mars['m_f_estimated'])

earth_to_mars['init_z_map'] = np.linspace(earth_to_mars['z_0'],earth_to_mars['z_f_estimated'],earth_to_mars['N'])
earth_to_mars['init_control'] = np.einsum('N,d->Nd',earth_to_mars['T_max']*np.exp(-earth_to_mars['init_z_map'])/2 , np.ones((3,))/np.sqrt(3)) 


###########
###########
###########

earth_to_dionysus = {}

earth_to_dionysus['name'] = 'Earth to Dionysus'
earth_to_dionysus['origin'] = 'Earth'
earth_to_dionysus['destination'] = 'Dionysus'
earth_to_dionysus['AU'] = 1.496e+8  # km
earth_to_dionysus['AUm'] = earth_to_dionysus['AU'] * 1e3  # meters
earth_to_dionysus['TU'] = 31536000 / (2 * np.pi)  # time unit (s)

# Gravitational parameter of the Sun (mu)
earth_to_dionysus['mu'] = 132712440018 / (earth_to_dionysus['AU'] ** 3) * (earth_to_dionysus['TU'] ** 2)

# Spacecraft parameters
earth_to_dionysus['m_0'] = 4000  # initial mass (kg)
earth_to_dionysus['g0'] = 9.8065  # gravitational acceleration (m/s^2)
earth_to_dionysus['I_sp'] = 3000  # specific impulse (s)

# Thrust calculations
earth_to_dionysus['T_max'] = 0.32 / earth_to_dionysus['AUm'] * (earth_to_dionysus['TU'] ** 2)

# Effective exhaust velocity
earth_to_dionysus['c'] = earth_to_dionysus['I_sp'] * earth_to_dionysus['g0'] / earth_to_dionysus['AUm'] * earth_to_dionysus['TU']

# Initial position and velocity vectors (in AU and AU/TU)
earth_to_dionysus['r_i'] = np.array([-3637871.081, 147099798.784, -2261.441]) / earth_to_dionysus['AU']
earth_to_dionysus['v_i'] = np.array([-30.265097, -0.8486854 ,  0.0000505]) / earth_to_dionysus['AU'] * earth_to_dionysus['TU']

# Final position and velocity vectors (in AU and AU/TU)
earth_to_dionysus['r_f'] = np.array([-302452014.884,  316097179.632,  82872290.0755]) / earth_to_dionysus['AU']
earth_to_dionysus['v_f'] = np.array([-4.533473, -13.110309, 0.656163]) / earth_to_dionysus['AU'] * earth_to_dionysus['TU']

# Mode of operation
earth_to_dionysus['mode'] = 'mee'

earth_to_dionysus['t_f_days'] = 3534
earth_to_dionysus['t_f'] = earth_to_dionysus['t_f_days'] * 24 * 3600 / earth_to_dionysus['TU']

earth_to_dionysus['N'] = 600
earth_to_dionysus['dt'] = earth_to_dionysus['t_f']/(earth_to_dionysus['N']-1)



earth_to_dionysus['x_i'] = coordinate_conversion.mee_from_sv(earth_to_dionysus['r_i'],earth_to_dionysus['v_i'],earth_to_dionysus['mu'])
earth_to_dionysus['x_f'] = coordinate_conversion.mee_from_sv(earth_to_dionysus['r_f'],earth_to_dionysus['v_f'],earth_to_dionysus['mu'])
earth_to_dionysus['x_f'][-1] += 12*np.pi


earth_to_dionysus['init_traj'] = np.linspace(earth_to_dionysus['x_i'],earth_to_dionysus['x_f'],earth_to_dionysus['N'])
earth_to_dionysus['m_f_estimated'] = 2500

earth_to_dionysus['z_0'] = np.log(earth_to_dionysus['m_0'])
earth_to_dionysus['z_f_estimated'] = np.log(earth_to_dionysus['m_f_estimated'])

earth_to_dionysus['init_z_map'] = np.linspace(earth_to_dionysus['z_0'],earth_to_dionysus['z_f_estimated'],earth_to_dionysus['N'])
earth_to_dionysus['init_control'] = np.zeros((earth_to_dionysus['N'],3))#np.einsum('N,d->Nd',earth_to_dionysus['T_max']*np.exp(-earth_to_dionysus['init_z_map'])/2 , np.ones((3,))/np.sqrt(3)) 
