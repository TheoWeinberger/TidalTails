import matplotlib.pyplot as plt
import numpy as np
import math as mth
import scipy.integrate as int
import matplotlib.animation as animation

#program using the Euler method of integration to solve Newton's gravitational equations for two massive bodies

G_const = 1.0  # gravitational constant
Mass_1 = 1.0 # mass of central heavy mass
Mass_2 = 1.0 # mass of second heavy mass

plt.rcParams.update({'font.size': 25})
plt.rcParams["font.family"] = "Times New Roman"

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)

#method to initialise the speed of a particle based
#takes arguments r_0 (radial position vector), G_const, Mass_1
def Init_Speed(r_0):
    z = [0.0, 0.0, 1.0]
    v_0 =  np.sqrt(np.divide(G_const*Mass_1,(np.linalg.norm(r_0, axis = 0))))*np.cross(z, r_0, axis = 0)/np.linalg.norm(r_0, axis = 0)
    return v_0

#method to initialise the positions of the particles
# particles evenly spaced on rings, takes argument R_num
def Init_Positions(R_num):
    Init_Rad = np.arange(2,R_num+2) #matrix to initialise radii
    Init_Number = Init_Rad * 6 #matrix to initialise number of masses per radius
    R_init = np.array([[],[]]) #initialise empty arrays where the initial positions of the particles will be stored
    for i in range(len(Init_Rad)):
        Init_Angle = np.arange(0, Init_Number[i])
        r = np.array ([Init_Rad[i]*np.array([np.cos(Init_Angle*2*mth.pi/Init_Number[i]), np.sin(Init_Angle*2*mth.pi/Init_Number[i])])])
        r_reshape = r.reshape(2, Init_Number[i])
        R_init = np.concatenate([R_init, r_reshape], axis = 1)
    Init_Coords = np.vstack([R_init,np.zeros(shape = [len(R_init[0,:])])])
    return Init_Coords # 3,120 matrix containing position values

#initialise galaxy positions to produce desired orbits
def Init_Galaxies_positions():
    r_0_galaxies = np.array([[0,0,0],[-8,16,0]]).T
    return r_0_galaxies

#initialise galaxy velocities. The first galaxy should initially be stationary.
#the second galaxy velocity is determined such that E_tot = 0
#takes arguments r_0_galaxies (initial galaxy positions)
def Init_Galaxies_speeds(r_0_galaxies):
    v_0 = mth.sqrt(2.0*G_const*Mass_1/np.linalg.norm(r_0_galaxies[:,1]-r_0_galaxies[:,0]))
    v_0_galaxies = np.array([[0,0,0],[v_0,0,0]]).T
    return v_0_galaxies

#method to work out the vector of action of newtonian gravity
#takes arguments r_pos (the current particle position)
def Acceleration(r_pos,r_galaxies):
    r_1 = r_pos.T - r_galaxies[:,0]
    r_2 = r_pos.T - r_galaxies[:,1]
    r_1_mod = np.linalg.norm(r_1, axis = 1)
    r_2_mod = np.linalg.norm(r_2, axis = 1)
    return G_const*Mass_1*np.divide(r_1.T,(r_1_mod)**3, out = np.zeros_like(r_1.T), where=r_1_mod!=0) + G_const*Mass_2*np.divide(r_2.T,(r_2_mod)**3, out = np.zeros_like(r_2.T), where=r_2_mod!=0)

#method to step the velocity
#takes arguments r_pos (the current particle position), velocity and d_t (time step)
def Velocity_Step(r_pos, velocity, r_galaxies, d_t):
    return velocity - d_t*Acceleration(r_pos,r_galaxies)

#method to step the position
#takes arguments r_pos (the current particle position), velocity and d_t (time step)
def Position_Step(r_pos, velocity, r_galaxies, d_t):
    return r_pos + velocity*d_t - 0.5*Acceleration(r_pos,r_galaxies)*(d_t)**2

#method to plot positions of system
#takes argument N(Number of steps), N_snapshot(value of N at which to plot snapshot), d_t(time step), R_num(number of rings)
def Two_Galaxies(N, N_snapshot, d_t, R_num):
    #determine starting configuration for the system
    r_0 = Init_Positions(R_num) 
    v_0 = Init_Speed(r_0) 
    r_0_galaxies = Init_Galaxies_positions()
    v_0_galaxies = Init_Galaxies_speeds(r_0_galaxies)

    #set up arrays where data will be stored
    r_pos = np.zeros((N,3,len(r_0[0,:])))
    velocity = np.zeros((N,3,len(r_0[0,:])))
    r_pos_galaxies = np.zeros((N,3,len(r_0_galaxies[0,:])))
    v_galaxies = np.zeros((N,3,len(r_0_galaxies[0,:])))

    #initialise
    r_pos[0,:,:] = r_0
    velocity[0,:,:] = v_0
    r_pos_galaxies[0,:,:] = r_0_galaxies
    v_galaxies[0,:,:] = v_0_galaxies

    #apply timestepping
    for i in range(0,N-1):
        velocity[i+1,:,:] = Velocity_Step(r_pos[i,:,:], velocity[i,:,:], r_pos_galaxies[i,:,:], d_t)
        r_pos[i+1,:,:] = Position_Step(r_pos[i,:,:], velocity[i,:,:], r_pos_galaxies[i,:,:], d_t)
        v_galaxies[i+1,:,:] = Velocity_Step(r_pos_galaxies[i,:,:], v_galaxies[i,:,:], r_pos_galaxies[i,:,:], d_t)
        r_pos_galaxies[i+1,:,:] = Position_Step(r_pos_galaxies[i,:,:], v_galaxies[i,:,:], r_pos_galaxies[i,:,:], d_t)

    R_pos = np.append(r_pos, r_pos_galaxies, axis = 2)

    c = np.random.random([len(R_pos[0,0,:])])
    s_star = np.ones(len(R_pos[0,0,:])-2)
    s_galaxy = np.array([10,10])
    s = np.append(s_star,s_galaxy)

    N_plot = np.arange(N)

    #plot galactic separation and print off minimum separation
    plt.plot(N_plot*d_t, np.linalg.norm(r_pos_galaxies[:,:,0]-r_pos_galaxies[:,:,1], axis= 1)) #plot evoution of planetary separation with time to ensure it fulfills separation parameters
    plt.title('Galactic Separation as a function of time')
    plt.xlabel('Time/s')
    plt.ylabel('Galaxy Separation/m')
    plt.show()

    print (np.min(np.linalg.norm(r_pos_galaxies[:,:,0]-r_pos_galaxies[:,:,1],axis = 1))) #print value of closest approach to find distance of closest approach

    #plot of arrangement at a given time
    plt.scatter(R_pos[N_snapshot,0,:], R_pos[N_snapshot,1,:], c=c, s=s*20) 
    plt.xlabel('x-distance/m')
    plt.ylabel('y-distance/m')
    
    #centering plot on central galaxy
    x_cen =  R_pos[N_snapshot,0,len(R_pos[0,0,:])-2]
    y_cen = R_pos[N_snapshot,1,len(R_pos[0,1,:])-2]
    plt.xlim(x_cen-40,x_cen+40)
    plt.ylim(y_cen-40,y_cen+40)
    t = mth.floor(N_snapshot*d_t)
    plt.title('N-body Orbital plots at time {}s'.format(t))
    plt.show()
    return R_pos

#method for updating simulation
#takes arguments i, x, y, scat, d_t, time, ax, and Acc_Facc
def update(i,x,y,scat,d_t,time,ax, Acc_Fac):
    #update data
    scat.set_offsets(np.vstack((x[Acc_Fac*i,:],y[Acc_Fac*i,:])).T)

    #update axes
    x_cen =  x[Acc_Fac*i,len(x[0,:])-2]
    y_cen = y[Acc_Fac*i,len(x[0,:])-2]
    ax.set_xlim(x_cen-30, x_cen+30)
    ax.set_ylim(y_cen-30, y_cen+30)

    #update timer
    t = mth.floor(d_t*Acc_Fac*i)
    time.set_text('Time = {}s'.format(t))
    return scat,time,

#method for running simulation
#takes arguments N_steps, N_snapshot(N value for which snapshot is printed) d_t, Acc_Facc(the acceleration factor for the movie) and R_num (number of rings)
def Simulation(N_steps, N_snapshot, d_t, Acc_Fac, R_num):
    R_pos = Two_Galaxies(N_steps, N_snapshot, d_t, R_num)

    x = R_pos[:,0,:] #x coordinates of planets
    y = R_pos[:,1,:] #y coordinates of planets

    fig,ax = plt.subplots()
    plt.xlabel('x-distance/m')
    plt.ylabel('y-distance/m')

    time = plt.title('Time = {}s', x = 0.90) #updating timer

    c = np.random.random([len(x[0,:])])
    s_star = np.ones(len(x[0,:])-2)
    s_galaxy = np.array([10,10])
    s = np.append(s_star,s_galaxy)
    scat = plt.scatter(x[0,:],y[0,:], c = c, s = s*20)

    ani = animation.FuncAnimation(fig,update, frames = mth.floor(N_steps/Acc_Fac), interval = 0.01, fargs = (x,y,scat,d_t,time,ax,Acc_Fac))

    #ani.save('TidalTails.mp4', writer = writer)
    
    plt.show()

#run simulation
#takes arguments N_steps, N_snapshot(N value for which snapshot is printed) d_t, Acc_Facc(the acceleration factor for the movie) and R_num (number of rings)
Simulation(10000, 9000, 0.001, 1000, 5)


