import matplotlib.pyplot as plt
import numpy as np
import math as mth
import scipy.integrate as int
import matplotlib.animation as animation
import time

#program using the velcoity verlet method of integration to solve Newton's gravitational equations for one massive body

G_const = 1.0  # gravitational constant
Mass_1 = 1.0 # mass of central heavy mass

plt.rcParams.update({'font.size': 45})
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
#particles evenly spaced on rings, takes argument R_num
def Init_Positions(R_num):
    Init_Rad = np.arange(2,R_num +2 ) #matrix to initialise radii
    Init_Number = Init_Rad * 6 #matrix to initialise number of masses per radius
    R_init = np.array([[],[]]) #initialise empy arrayn where the initial positions of the particles will be stored
    for i in range(len(Init_Rad)):
        Init_Angle = np.arange(0, Init_Number[i])
        r = np.array ([Init_Rad[i]*np.array([np.cos(Init_Angle*2*mth.pi/Init_Number[i]), np.sin(Init_Angle*2*mth.pi/Init_Number[i])])])
        r_reshape = r.reshape(2, Init_Number[i])
        R_init = np.concatenate([R_init, r_reshape], axis = 1)
    Init_Coords = np.vstack([R_init,np.zeros(shape = [len(R_init[0,:])])])
    return Init_Coords # 3,120 matrix containing position values

#method to work out the vector of action of newtonian gravity
#takes arguments r_pos (the current particle position)
def Acceleration(r_pos):
    return G_const*Mass_1*np.divide(r_pos,(np.linalg.norm(r_pos, axis = 0)**3))

#method to step the velocity
#takes arguments r_pos_1 (the current particle position), r_pos (old position), velocity and d_t (time step)
def Velocity_Step(r_pos_1, r_pos, velocity, d_t):
    return velocity - 0.5*d_t*(Acceleration(r_pos)+Acceleration(r_pos_1))

#method to step the position
#takes arguments r_pos (the current particle position), velocity and d_t (time step)
def Position_Step(r_pos, velocity, d_t):
    return r_pos + velocity*d_t - 0.5*Acceleration(r_pos)*(d_t)**2

#test function to see if a circular orbit for a single mass is formed
#takes arguments N (number of steps) and d_t (time interval)
def Single_Body(N, d_t):
    r_0 = [3,0,0]
    v_0 = Init_Speed(r_0)
    r_pos = np.zeros((N,3))
    velocity = np.zeros((N,3))

    #initialise
    r_pos[0,:] = r_0 
    velocity[0,:] = v_0


    #implement velocity verlet
    for i in range(0,N-1):
        r_pos[i+1,:] = Position_Step(r_pos[i,:], velocity[i,:], d_t)
        velocity[i+1,:] = Velocity_Step(r_pos[i+1,:], r_pos[i,:], velocity[i,:], d_t)

    #plot complete evolution of all particles in system
    plt.plot (r_pos[:,0], r_pos[:,1], 'k')
    #plt.title('Path of single mass orbit')
    plt.xlabel('x-distance/m')
    plt.ylabel('y-distance/m')
    plt.show()
    return r_pos, velocity

#method to determine how well integrator conserves energy
def Energy(N, d_t):
    r_pos, velocity = Single_Body(N,d_t)
    Kinetic = 0.5*np.linalg.norm(velocity,axis = 1)**2
    Gravitational = -(G_const*Mass_1)/np.linalg.norm(r_pos,axis = 1)
    Total = Kinetic + Gravitational 
    t = np.linspace(0, N*d_t, N)
    plt.plot(t, Total, 'k')
    #plt.title('Energy conservation as a function of time')
    plt.xlabel('Time/s')
    plt.ylabel('Energy/J')
    plt.show()
    return


#Function to produce N-body orbits
#takes arguments N (number of steps), d_t (time interval) and R_num
def Single_Galaxy(N, d_t, R_num):
    r_0 = Init_Positions(R_num) # initialise positions
    v_0 = Init_Speed(r_0) # initialise velocities

    r_pos = np.zeros((N,3,len(r_0[0,:])))
    velocity = np.zeros((N,3,len(r_0[0,:])))

    #initialise
    r_pos[0,:,:] = r_0
    velocity[0,:,:] = v_0
    r_pos_galaxy = np.zeros_like(r_pos)

    #apply timestepping
    for i in range(0,N-1):
        r_pos[i+1,:,:] = Position_Step(r_pos[i,:,:], velocity[i,:,:], d_t)
        velocity[i+1,:,:] = Velocity_Step(r_pos[i+1,:,:], r_pos[i,:,:], velocity[i,:,:], d_t)

    R_pos = np.append(r_pos, r_pos_galaxy, axis = 2)

    #plot complete evolution of all particles in system
    plt.plot (R_pos[:,0,:], R_pos[:,1,:])
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.xlabel('x-distance/m')
    plt.ylabel('y-distance/m')
    #plt.title('N-body Orbital plots')
    plt.show()
    return R_pos

#method for updating simulation
#takes arguments i, x, y, scat, d_t, time and Acc_fac (acceleration factor for the movie)
def update(i,x,y,scat,d_t,time, Acc_fac):
    scat.set_offsets(np.vstack((x[Acc_fac*i,:],y[Acc_fac*i,:])).T)
    t = mth.floor(Acc_fac*d_t*i)
    time.set_text('Time = {}s'.format(t))
    return scat,time,

#method for running simulation
#takes arguments N_steps, d_t, Acc_fac and R_num
def Simulation(N_steps, d_t, Acc_fac, R_num):
    R_pos = Single_Galaxy(N_steps,d_t, R_num)

    x = R_pos[:,0,:] #x coordinates of planets
    y = R_pos[:,1,:] #y coordinates of planets

    fig = plt.figure()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.xlabel('x-distance/m')
    plt.ylabel('y-distance/m')

    time = plt.title('Time = {}s', x = 0.90)

    c = np.random.random([len(x[0,:])])
    s_star = np.ones(len(R_pos[0,0,:])-1)
    s_galaxy = np.array([10])
    s = np.append(s_star,s_galaxy)
    scat = plt.scatter(x[0,:],y[0,:], c = c, s = s*50)


    ani = animation.FuncAnimation(fig,update, frames = mth.floor(N_steps/Acc_fac), interval = 0.01, fargs = (x,y,scat,d_t,time, Acc_fac))

    #ani.save('SingleModelEuler001.mp4', writer = writer)
    
    plt.show()

#method to determine runtime of integrator
def timer():
    times = np.zeros((100))
    for i in range(100):
        start_time = time.time()
        Single_Body(10000, 0.01)
        times[i] = (time.time()-start_time)
    mean_time = np.average(times)
    std = np.std(times)
    print(mean_time)
    print(std)
    return

Simulation(10000, 0.01, 1, 5)
Energy(100000,0.01)

#timer()