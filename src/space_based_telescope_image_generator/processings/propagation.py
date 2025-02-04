import re
import datetime
import math as m
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta
from numpy.typing import NDArray

MU = 3.986004418e14     # [m3/s2] Standard gravitational constant of the Earth
J2 = 1082.62668e-6 		# [-] Perturbation constant from gravitational potential
RE = 6378.137e3 		# [m] Equatorial radius of the Earth
DAY_SECONDS = 86400 	# [s] Seconds in a solar day
PI = m.pi               # Pi

TLE = ["1 55044U 23001AM  25005.04515951  .02585999  11606-1  26070-2 0  9995",
    "2 55044  97.3788  79.4722 0006145 286.1743  73.8866 16.13983607112084"]

# Cleans the TLE from all multiple spaces, replaces them with single spaces
for j in range(len(TLE)):                               # Runs through all line of the TLE
    count = 0                                           # While loop safety
    while "  " in TLE[j] and count < 100:
        TLE[j] = re.sub(r'\s+', " ", TLE[j]).strip()    # Detects and replaces multiple spaces
        count += 1                                      # Avoids infinite while loop in case of problem

def tle2keplerian(TLE: tuple[str, str])->tuple[float, float, float, float, float, float, datetime]:
    '''
    This function extracts the Keplerian elements and date of a space object from a TLE
    
    Args:
        TLE: tuple[str, str]
    
    Returns:
        a: float, Semi-major axis (m)
        e: float, Eccentricity (-)
        i: float, Inclination (rad)
        Omega: float, Longitude of the Right Ascension of the Ascending Node (rad)
        omega: float, Argument of perigee (rad)
        nu: float, True anomaly (rad)
        date: datetime, Date of the TLE
    '''
    TLE1 = TLE[0].split(' ')
    TLE2 = TLE[1].split(' ')

    e = float(str('0.') + TLE2[4])          # [-] Eccentricity
    i = float(TLE2[2])                      # [deg] Inclination
    Omega = float(TLE2[3])                  # [deg] Longitude of RAAN
    omega = float(TLE2[5])                  # [deg] Argument of perigee
    nu = float(TLE2[6])                     # [deg] True anomaly
    MM = float(TLE2[7])                     # [rev/day] Mean motion
    T = 86400/MM                            # [s] Orbital period
    a = (MU*(T/(2*PI))**2)**(1/3)           # [m] Semi-major axis

    year = 2000 + int(float(TLE1[3])/1000)  # Associated date
    doy = float(TLE1[3])%1000               # Day of the year

    month, day = getDate(year, doy)
    hour, minute, second = s2hms(doy)

    date = datetime.datetime(year, month, day, hour, minute, second)
    return a, e, m.radians(i), m.radians(Omega), m.radians(omega), m.radians(nu), date

def getDate(y: float, doy: float)->tuple[int, int]:
    '''
    This function finds the date according to the year and the day of the year (doy)

    Args:
        y: float, year (2000+)
        doy: float, day of the year
    
    Returns:
        month: int, number of the month from 1 to 12
        day: int, number of the date from 1 to 31
    '''
    doy = int(doy)          # Takes the integer part: the fractional part corresponds to hours, minute, second
    dayPerMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    if y%4 == 0:            # Leap year
        dayPerMonth[1] += 1 # Adds one day in February
    
    countDay = 0            # Counts the number of days passed after each month
    count = 0               # While loop safety
    j = 0                   # Iteration
    while countDay < doy and count < 100:
        countDay += dayPerMonth[j]
        j += 1
        count += 1
    month = j
    day = doy - countDay + dayPerMonth[month]
    return month, day

def s2hms(doy: float)->tuple[int, int, int]:
    '''
    This function converts the day of the year into the corresponding hour, minute and second
    
    Args:
        doy: float, day of the year between 0 and 366
    
    Returns:
        hour: int, number of hours passed since the beginning of the say in UTC
        minute: int, number of minutes passed
        second: int, number of whole seconds passed
    '''
    fractional = doy%1          # Takes the fractional part of the DOY (corresponding to the advancement in the day)
    secondDoy = fractional*86400# Converts the fractional part in seconds
    
    # Initializing the variables
    hour = 0
    minute = 0
    second = 0
    
    count = 0                               # While loop safety
    while secondDoy > 3600 and count < 100: # Counts the hours
        hour += 1
        secondDoy -= 3600
        count += 1
    
    count = 0                               # While loop safety
    while secondDoy > 60 and count < 100:   # Counts the minutes
        minute += 1
        secondDoy -= 60
        count += 1
    second = int(secondDoy)                  # The remainder is the number of seconds

    return hour, minute, second

def Rot(a: float, u: int)->NDArray:	
	"""
	Rotation matrices method

	Args:
		a: float, rotation angle [rad]
		u: int, rotation axis (0 = x, 1 = y, 2 = z)

	Returns:
		Rot: NDArray, 3x3 matrix
	"""
	c = np.cos(a)
	s = np.sin(a)
	if u == 0:						# Rotation about the x-axis
		Rot = np.array([[1, 0, 0], 
						[0, c, -s], 
						[0, s, c]])
	elif u == 1:					# Rotation about the y-axis
		Rot = np.array([[c, 0, s], 
						[0, 1, 0], 
						[-s, 0, c]])
	elif u == 2:					# Rotation about the z-axis
		Rot = np.array([[c, -s, 0], 
						[s, c, 0], 
						[0, 0, 1]])
	else:							# Wrong axis number: error
		Rot = np.array([[1, 0, 0],
						[0, 1, 0], 
						[0, 0, 1]])
		print("Error: wrong axis number provided for rotation matrix")
	return Rot

def radius(a: float, e: float, nu: float)->float:
	"""
	Instantaneous radius of the orbit, given for the current Keplerian elements a, e and nu

	Args:
		a: float, semi major axis [m]
		e: float, eccentricity (distance between focal point / half major axis)
		nu: float, angle between the direction of the periapsis and the current position of an object [rad]
	
	Returns: 
		Position radius for a non-circular orbit, float
	"""
	return a * (1 - e**2)/(1 + e * m.cos(nu))

def keplerian2cartesian(a: float, e: float, i: float, Omega: float, omega: float, nu: float)->tuple[NDArray,NDArray]:
	"""
	Converts the Keplerian elements into state vectors

	Args:
		a: semi major axis [m], float
		e: eccentricity (=distance between focal point / half major axis), float
		i: inclination of the orbit [rad], float
		Omega: longitude of the right ascension of the ascending node [rad], float
		omega: argument of perigee [rad], float
		nu: angle between the direction of the periapsis and the current position of an object [rad], float

	Returns: 
		r: ECI position vector 3x1, NDArray
		rdot: ECI velocity vector 3x1, NDArray
	"""
	rc = radius(a, e, nu)
	E = 2*m.atan2(m.tan(nu/2), m.sqrt((1 + e)/(1 - e)))
	o = rc*np.array([m.cos(nu), m.sin(nu), 0])
	odot = m.sqrt(MU*a)/rc*np.array([-m.sin(E), m.sqrt(1 - e**2)*m.cos(E), 0])

	r = np.matmul(Rot(Omega, 2), np.matmul(Rot(i, 0), np.matmul(Rot(omega, 2), o)))
	rdot = np.matmul(Rot(Omega, 2), np.matmul(Rot(i, 0), np.matmul(Rot(omega, 2), odot)))
	return r, rdot

def perturbations(pos: NDArray, vel: NDArray) -> tuple[NDArray, NDArray]:
    """
    Calculates the instantaneous acceleration and jerk vectors

	Args:
    	pos: ECI position vector, NDArray 3x1 
    	vel: ECI velocity vector, NDArray 3x1 

    Returns: 
		acc: NDArray, ECI acceleration vector 3x1 (m/s2)
		jerk: NDArray, ECI jerk vector 3x1 first time deritative of acceleration (m/s3)
    """
    pos_norm = np.linalg.norm(pos)      # [m] Norm of the positio vector
    
    if pos_norm == 0:
        return np.zeros(3), np.zeros(3) # Special case : position at the origin of the reference frame

    acc2B = - MU * pos / pos_norm**3  # [m/s2] 2-body acceleration

    accJ2 = [                         # [m/s2] J2 acceleration
        MU / pos_norm**2 * (RE / pos_norm)**2 * J2 * (15 / 2 * (pos[2] / pos_norm)**2 - 3 / 2) * pos[0] / pos_norm,
        MU / pos_norm**2 * (RE / pos_norm)**2 * J2 * (15 / 2 * (pos[2] / pos_norm)**2 - 3 / 2) * pos[1] / pos_norm,
        MU / pos_norm**2 * (RE / pos_norm)**2 * J2 * (15 / 2 * (pos[2] / pos_norm)**2 - 9 / 2) * pos[2] / pos_norm,
    ]

    acc = acc2B + accJ2
    jerk = - MU * (vel / pos_norm**3 - 3 * pos * np.dot(pos, vel) / pos_norm**5)    # [m/s3] 2-body jerk
    return acc, jerk

a, e, i, Omega, omega, nu, date = tle2keplerian(TLE)
pos, vel = keplerian2cartesian(a, e, i, Omega, omega, nu)
acc, jerk = perturbations(pos, vel)

# Simulation parameters
nb_days = 1						# [day] Number of days of propagation
time = DAY_SECONDS*nb_days  	# [s] Total time
dt = 5    						# [s] Time step of the simulation

# Initialize arrays to store position and velocity
pos_array = []
vel_array = []

# Main simulation loop
for i in range(int(time / dt)):
    pos_new = pos + vel * dt + acc * dt**2 / 2 + jerk * dt**3 / 6	# [m] Forward Euler, order 3 integration
    vel_new = vel + acc * dt + jerk * dt**2 / 2					# [m/s] Forward Euler, order 2 integration
    date = date + timedelta(seconds=dt)
    acc, jerk = perturbations(pos_new, vel_new)					# [m/s2, m/s3] New acceleration and jerk vectors
    pos_array.append(pos_new)											# [m] Position array
    vel_array.append(vel_new)										# [m/s] Velocity array
    pos = pos_new														# [m] Resetting position vector
    vel = vel_new													# [m/s] Resetting velocity vector

# Convert lists to numpy arrays
pos_array = np.array(pos_array)
vel_array = np.array(vel_array)

# Plotting the orbit in 3D
fig2 = plt.figure()
plt.title("Orbit in ECI")

# Creating a wireframe for the Earth
phi = np.linspace(0, 2 * m.pi, 36)	# Angular meshing
theta = np.linspace(0, m.pi, 18)	# Angular meshing
A, B = np.meshgrid(theta, phi)
X = RE * np.sin(A) * np.cos(B)  # X coordinates
Y = RE * np.sin(A) * np.sin(B)  # Y coordinates
Z = RE * np.cos(A)              # Z coordinates

# Plotting the Earth and the trajectory
eci = fig2.add_subplot(111, projection = '3d')
# Sphere
eci.plot_wireframe(X, Y, Z, color = 'c', zorder = 1, alpha = 0.25)  # type: ignore
# Trajectory
eci.plot3D(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], color = 'm') # type: ignore

# Plotting the ECI coordinates in 2D
plt.figure()
# X_ECI
plt.plot(pos_array[:, 0], color = 'r') # type: ignore
# Y_ECI 
plt.plot(pos_array[:, 1], color = 'g') # type: ignore
# Z_ECI
plt.plot(pos_array[:, 2], color = 'b') # type: ignore
plt.grid()
plt.legend(['$X_{ECI}$', '$Y_{ECI}$', '$Z_{ECI}$'])
plt.title('Coordinates in ECI of the satellite versus time')
plt.xlabel(f"Time, one step = {dt} sec")
plt.ylabel('ECI coordinates in m')
plt.show()