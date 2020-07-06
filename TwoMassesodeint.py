import matplotlib.pyplot as plt
import numpy as np
import math as mth
import scipy.integrate as integrate
import matplotlib.animation as animation

#program using odeint to solve coupled 1st order ODEs to solve for Newtonian gravity for two massive bodies

G_const = 1  # gravitational constant
Mass_1 = 1 # mass of central heavy mass
Mass_2 = 1 # mass of second heavy mass

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
#particles evenly spaced on ring takes argument R_num
def Init_Positions(R_num):
    Init_Rad = np.arange(2, R_num + 2) #matrix to initialise radii
    Init_Number = Init_Rad * 6 #matrix to initialise number of masses per radius
    R_init = np.array([[],[]]) #initialise empy arrayn where the initial positions of the particles will be stored
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

#old acceleration method requiring loopsincluded for comparison
#def Acceleration(y,t):
#   #set up arrays for galaxy positions
#    r_galaxy_1 = np.array([y[0],y[1],y[2]])
#    r_galaxy_2 = np.array([y[6],y[7],y[8]])
#
#    #determine new position after gravitation acceleration
#    pos_vel_step_prime = []
#    N = int(len(y)/6)
#    for i in range(N):
#        r_pos = np.array([y[6*i],y[6*i+1],y[6*i+2]])
#        r_1 = r_pos - r_galaxy_1
#        r_2 = r_pos - r_galaxy_2
#        r_1_mod = np.linalg.norm(r_1, axis = 0)
#        r_2_mod = np.linalg.norm(r_2, axis = 0)
#        Acceleration = -G_const*Mass_1*np.divide(r_1,(r_1_mod)**3, out = np.zeros_like(r_1), where=r_1_mod!=0) - G_const*Mass_2*np.divide(r_2,(r_2_mod)**3, out = np.zeros_like(r_2), where=r_2_mod!=0)
#        pos_vel_step_list = [y[6*i+3], y[6*i+4], y[6*i+5] , Acceleration[0], Acceleration[1], Acceleration[2]]
#        pos_vel_step_prime.append(pos_vel_step_list)
#    pos_vel_step = np.array(pos_vel_step_prime).reshape(6*N)
#    return pos_vel_step


#function which provides coupled 1st order ODEs 
#takes arguments y (array of velocities and positions) and t (time)
def Acceleration(y,t):
    #set up arrays for galaxy positions
    r_galaxy_1 = np.array([y[0],y[1],y[2]])
    r_galaxy_2 = np.array([y[6],y[7],y[8]])

    #determine new position after gravitation acceleration
    N = int(len(y)/6)
    N_array = np.arange(N)
    r_pos = np.array([y[6*N_array],y[6*N_array+1],y[6*N_array+2]])
    r_1 = (r_pos.T - r_galaxy_1).T
    r_2 = (r_pos.T - r_galaxy_2).T
    r_1_mod = np.linalg.norm(r_1, axis = 0)
    r_2_mod = np.linalg.norm(r_2, axis = 0)
    Acceleration = -G_const*Mass_1*np.divide(r_1,(r_1_mod)**3, out = np.zeros_like(r_1), where=r_1_mod!=0) - G_const*Mass_2*np.divide(r_2,(r_2_mod)**3, out = np.zeros_like(r_2), where=r_2_mod!=0)
    pos_vel_step_list = np.array([y[6*N_array+3], y[6*N_array+4], y[6*N_array+5] , Acceleration[0], Acceleration[1], Acceleration[2]])
    pos_vel_step = np.array(pos_vel_step_list.T).reshape(6*N)
    return pos_vel_step

#method to determine how well energy is conserved by this integrator
def Energy(R_pos, t):
    Kinetic = np.sum(0.5*np.linalg.norm(R_pos[:,:2,3:6],axis = 2)**2, axis = 1)
    Gravitational = -G_const*Mass_1*Mass_2/np.linalg.norm(R_pos[:,0,0:3]-R_pos[:,1,0:3], axis= 1)
    Total = Kinetic + Gravitational 
    plt.plot(t, Total, 'k')
    #plt.title('Energy conservation as a function of time')
    plt.xlabel('Time/s')
    plt.ylabel('Energy/J')
    plt.show()
    return

#method to determine how well momentum is conserved by this integrator
def Momentum(velocity, t):
    Mom_1 = Mass_1*(velocity[:,0,:])
    Mom_2 = Mass_2*(velocity[:,1,:])
    Total_x_mom = Mom_1[:,3]+Mom_2[:,3]
    Total_y_mom = Mom_1[:,4]+Mom_2[:,4]
    plt.plot(t, Total_x_mom, 'k')
    #plt.title('Momentum conservation as a function of time')
    plt.xlabel('Time/s')
    plt.ylabel('x-momentum/kgms$^-1$')
    plt.show()
    plt.plot(t, Total_y_mom, 'k')
    #plt.title('Momentum conservation as a function of time')
    plt.xlabel('Time/s')
    plt.ylabel('y-momentum/kgms$^-1$')
    plt.show()
    return

#function to evaluate positions and velocities of particles
#takes arguments T_max(max time), and N_steps(number of points at which the system is evaluated)
def Two_Galaxies(T_max, N_steps, N_snapshot, R_num):
    #initialise matrices to provide for initial value problem
    r_0 = Init_Positions(R_num)
    v_0 = Init_Speed(r_0)
    r_0_galaxies = Init_Galaxies_positions()
    v_0_galaxies = Init_Galaxies_speeds(r_0_galaxies)
    r_0_prime = np.append(r_0_galaxies,r_0, axis = 1)
    v_0_prime = np.append(v_0_galaxies,v_0, axis = 1)

    #combine and reshape matrix to be accepted by Gravity method
    pos_vel_0_prime = np.concatenate([r_0_prime.T, v_0_prime.T], axis = 1)
    pos_vel_0 = pos_vel_0_prime.reshape(6*len(pos_vel_0_prime[:,0]))

    #define time period over which system is solved and solve using odeint
    t = np.linspace(0, T_max, N_steps)
    R_pos_prime = integrate.odeint(Acceleration,pos_vel_0, t)
    R_pos = R_pos_prime.reshape(N_steps,len(r_0_prime[0,:]),6)

    #check energy conservation
    Energy(R_pos,t)

    #check momentum conservation
    Momentum(R_pos,t)


    #plot galactic separation and print off minimum separation
    plt.plot(t, np.linalg.norm(R_pos[:,0,0:3]-R_pos[:,1,0:3], axis= 1), 'k') #plot evoution of planetary separation with time to ensure it fulfills separation parameters
    #plt.title('Galactic Separation as a function of time')
    plt.xlabel('Time/s')
    plt.ylabel('Galaxy Separation/m')
    plt.show()

    print (np.min(np.linalg.norm(R_pos[:,0,0:3]-R_pos[:,1,0:3], axis= 1))) #print value of closest approach to find distance of closest approach

    c = np.random.random([len(R_pos[0,:,0])])
    s_star = np.ones(len(R_pos[0,:,0])-2)
    s_galaxy = np.array([10,10])
    s = np.append(s_galaxy,s_star)


    #plot system arrangement at a given point in time (N_Snapshot)
    plt.scatter(R_pos[N_snapshot,:,0], R_pos[N_snapshot,:,1], c=c, s=s*50)
    plt.xlabel('x-distance/m')
    plt.ylabel('y-distance/m')
    
    #centering plot on central galaxy
    x_cen =  R_pos[N_snapshot,0,0]
    y_cen = R_pos[N_snapshot,0,1]
    plt.xlim(x_cen-30,x_cen+30)
    plt.ylim(y_cen-30,y_cen+30)
    d_t = T_max/N_steps
    t = mth.floor(N_snapshot*d_t)
    plt.title('N-body Orbital plots at time {}s'.format(t))
    plt.show()
    return R_pos

#method for updating simulation
#takes arguments i, x, y, scat, ax, Acc_Facc, d_t and time
def update(i,x,y,scat,ax, Acc_Fac, d_t, time):
    #update data
    scat.set_offsets(np.vstack((x[Acc_Fac*i,:],y[Acc_Fac*i,:])).T)

    #update axes
    x_cen =  x[Acc_Fac*i,0]
    y_cen = y[Acc_Fac*i,0]
    ax.set_xlim(x_cen-30, x_cen+30)
    ax.set_ylim(y_cen-30, y_cen+30)

    #update timer
    t = mth.floor(d_t*Acc_Fac*i)
    time.set_text('Time = {}s'.format(t))
    return scat,

#method for running simulation
#takes arguments Acc_Facc(the acceleration factor for the movie), T_max (max time of simulation), N_steps (Number of steps for integrator), N_snapshot(frame at which snapshot is taken) and R_num (number of rings)
def Simulation(Acc_Fac, T_max, N_steps, N_Snapshot, R_num):
    R_pos = Two_Galaxies(T_max, N_steps, N_Snapshot, R_num)

    x = R_pos[:,:,0] #x coordinates of planets
    y = R_pos[:,:,1] #y coordinates of planets

    fig,ax = plt.subplots()
    plt.xlabel('x-distance/m')
    plt.ylabel('y-distance/m')

    time = plt.title('Time = {}s', x = 0.90) #updating timer
    d_t = T_max/N_steps

    c = np.random.random([len(x[0,:])])
    s_star = np.ones(len(x[0,:])-2)
    s_galaxy = np.array([10,10])
    s = np.append(s_galaxy, s_star)
    scat = plt.scatter(x[0,:],y[0,:], c = c, s = s*50)

    ani = animation.FuncAnimation(fig,update, frames = int(N_steps/Acc_Fac), interval = 0.01, fargs = (x,y,scat,ax,Acc_Fac,d_t, time))

    #ani.save('TidalTailsodeintrev.mp4', writer = writer)
    
    plt.show()

#run simulation
#takes arguments Acc_Facc(the acceleration factor for the movie), T_Max(max time of integration), N_steps (number of steps in simulation), N_snapshot (frame to plot snapshot at) and R_num (number of rings)
Simulation(1000, 100, 1000000, 900000, 5)


