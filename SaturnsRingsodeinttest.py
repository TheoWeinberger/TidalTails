import matplotlib.pyplot as plt
import numpy as np
import math as mth
import scipy.integrate as integrate
import matplotlib.animation as animation

#program using odeint to solve coupled 1st order ODEs to simulate dynamics of Saturn's Rings

G_const = 6.7*10**-29  # gravitational constant scaled for time and distance 
Mass_1 = 5.7*10**26 # mass of Saturn
Mass_2 = 3.7*10**22 # mass of Mimas
Mass_3 = 5.0*10**19 # mass of Pan
density = 400 #surface density of rings

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
#particles evenly spaced on rings
def Init_Positions():
    Init_Rad = np.arange(90, 140) #matrix to initialise radii
    Init_Coords = np.vstack([np.zeros_like(Init_Rad), Init_Rad, np.zeros_like(Init_Rad)])
    return Init_Coords #matrix containing position values

#initialise mass positions to produce desired orbits
def Init_mass_positions():
    r_0_mass = np.array([[0,0,0],[0,185,0],[0,133.6,0]]).T
    return r_0_mass

#initialise mass velocities. The first mass should initially be stationary.
#the other mass velocities are such that they produce circular orbits with initial velocity in the negative x direction
#takes arguments r_0_mass (initial mass positions)
def Init_mass_speeds(r_0_mass):
    v_0_1 = mth.sqrt(G_const*Mass_1/np.linalg.norm(r_0_mass[:,1]-r_0_mass[:,0]))
    v_0_2 = mth.sqrt(G_const*Mass_1/np.linalg.norm(r_0_mass[:,2]-r_0_mass[:,0]))
    v_0_mass = np.array([[0,0,0],[-v_0_1,0,0],[-v_0_2,0,0]]).T
    return v_0_mass

#function which provides coupled 1st order ODEs 
#takes arguments t(time) and y(array containing velocities and positions)
def Acceleration(y,t):
    #set up arrays for mass positions
    r_mass_1 = np.array([y[0],y[1],y[2]])
    r_mass_2 = np.array([y[6],y[7],y[8]])
    r_mass_3 = np.array([y[12],y[13],y[14]])

    #determine new position after gravitation acceleration

    N = int(len(y)/6)
    N_array = np.arange(N) #array to solve for each particle simulatneously

    r_pos = np.array([y[6*N_array],y[6*N_array+1],y[6*N_array+2]]) #old positions

    #separation from masses (vector)
    r_1 = (r_pos.T - r_mass_1).T 
    r_2 = (r_pos.T - r_mass_2).T 
    r_3 = (r_pos.T - r_mass_3).T

    #distance from masses
    r_1_mod = np.linalg.norm(r_1, axis = 0)
    r_2_mod = np.linalg.norm(r_2, axis = 0)
    r_3_mod = np.linalg.norm(r_3, axis = 0)

    #scaled distance to outside of ring for self gravity
    r_a = np.divide(r_1_mod, r_1_mod[N-1], out =np.zeros_like(r_1_mod), where=r_1_mod!=r_1_mod[1])

    #contritbutions to acceleration
    Self_Acceleration = 2*G_const*density*mth.pi*np.divide(r_1, r_1_mod, out = np.zeros_like(r_1), where=r_1_mod!= r_1_mod[0])*(0.5*r_a + 0.1875*np.power(r_a, 3) + 0.1171*np.power(r_a, 5))
    Mass_1_Acceleration = G_const*Mass_1*np.divide(r_1,(r_1_mod)**3, out = np.zeros_like(r_1), where=r_1_mod!=0)
    Mass_2_Acceleration = G_const*Mass_2*np.divide(r_2,(r_2_mod)**3, out = np.zeros_like(r_2), where=r_2_mod!=0)
    Mass_3_Acceleration = G_const*Mass_3*np.divide(r_3,(r_3_mod)**3, out = np.zeros_like(r_3), where=r_3_mod!=0)

    #combined acceleration
    Acceleration = -Mass_1_Acceleration - Mass_2_Acceleration - Mass_3_Acceleration - Self_Acceleration 

    #new positions and velocities
    pos_vel_step_prime = np.array([y[6*N_array+3], y[6*N_array+4], y[6*N_array+5] , Acceleration[0], Acceleration[1], Acceleration[2]])
    pos_vel_step = np.array(pos_vel_step_prime.T).reshape(6*N) #new positions
    print(t)
    return pos_vel_step

#method to determine how well energy is conserved by this integrator, considering energy of first two masses
#takes arguments t and R_pos(velocity and position array)
def Energy(R_pos, t):
    Kinetic = np.sum(0.5*np.linalg.norm(R_pos[:,:3,3:6],axis = 2)**2, axis = 1)
    GP_12 = -G_const*Mass_1*Mass_2/np.linalg.norm(R_pos[:,0,0:3]-R_pos[:,1,0:3], axis= 1)
    GP_13 = - G_const*Mass_1*Mass_3/np.linalg.norm(R_pos[:,0,0:3]-R_pos[:,2,0:3], axis= 1)
    GP_23 = -G_const*Mass_3*Mass_2/np.linalg.norm(R_pos[:,2,0:3]-R_pos[:,1,0:3], axis= 1)
    Gravitational = GP_12 + GP_23 + GP_13
    Total = Kinetic + Gravitational 
    plt.plot(t, Total, 'k')
    #plt.title('Energy conservation as a function of time')
    plt.xlabel('Time/s')
    plt.ylabel('Energy/J')
    plt.show()
    return

#method to determine how well momentum is conserved by this integrator, considering momemtum of first two masses 
#takes arguments velocity and t
def Momentum(velocity, t):
    Mom_1 = Mass_1*(velocity[:,0,:])
    Mom_2 = Mass_2*(velocity[:,1,:])
    Mom_3 = Mass_3*(velocity[:,2,:])
    Total_x_mom = Mom_1[:,3]+Mom_2[:,3] + Mom_3[:,3]
    Total_y_mom = Mom_1[:,4]+Mom_2[:,4] + Mom_3[:,4]

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
def Two_mass(T_max, N_steps):
    #intialise matrices to provide for initial value problem
    r_0 = Init_Positions()
    v_0 = Init_Speed(r_0)
    r_0_mass = Init_mass_positions()
    v_0_mass = Init_mass_speeds(r_0_mass)
    r_0_prime = np.append(r_0_mass,r_0, axis = 1)
    v_0_prime = np.append(v_0_mass,v_0, axis = 1)

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


    #plot distance of ring test masses from centre
    for i in range(len(R_pos[0,:,0])):
        plt.plot(t, np.linalg.norm(R_pos[:,i,0:3]-R_pos[:,0,0:3], axis= 1)) #plot evoution of planetary separation with time to ensure it fulfills separation parameters
    #plt.title('Separation as a function of time')
    plt.xlabel('Time/s')
    plt.ylabel('Distance from Saturn/Mm')
    plt.show()
    return 

Two_mass(1000000,1000000)


