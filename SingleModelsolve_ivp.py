import matplotlib.pyplot as plt
import numpy as np
import math as mth
import scipy.integrate as int
import matplotlib.animation as animation

#program to show the loss of energy usng solve_ivp making it an unsuitbale integrator to use for this problem

G_const = 1  # gravitational constant
Mass_1 = 1 # mass of central heavy mass

plt.rcParams.update({'font.size': 25})
plt.rcParams["font.family"] = "Times New Roman"

#method to initialise the speed of a particle based
#takes arguments r_0 (radial position vector), G_const, Mass_1
def Init_Speed(r_0):
    z = [0.0, 0.0, 1.0]
    v_0 =  np.sqrt(np.divide(G_const*Mass_1,(np.linalg.norm(r_0, axis = 0))))*np.cross(z, r_0, axis = 0)/np.linalg.norm(r_0, axis = 0)
    return v_0

#method to initialise the positions of the particles
# particles evenly spaced on ring s
def Init_Positions():
    Init_Rad = np.arange(2,7) #matrix to initialise radii
    Init_Number = Init_Rad * 6 #matrix to initialise number of masses per radius
    R_init = np.array([[],[]]) #initialise empy arrayn where the initial positions of the particles will be stored
    for i in range(len(Init_Rad)):
        Init_Angle = np.arange(0, Init_Number[i])
        r = np.array ([Init_Rad[i]*np.array([np.cos(Init_Angle*2*mth.pi/Init_Number[i]), np.sin(Init_Angle*2*mth.pi/Init_Number[i])])])
        r_reshape = r.reshape(2, Init_Number[i])
        R_init = np.concatenate([R_init, r_reshape], axis = 1)
    Init_Coords = np.vstack([R_init,np.zeros(shape = [len(R_init[0,:])])])
    return Init_Coords # 3,120 matrix containing position values

def f_grav(t, y):
    x1, x2, x3, v1, v2, v3 = y
    dydt = [v1, v2, v3 , -x1*G_const*Mass_1/(x1**2+x2**2)**(3/2), -x2*G_const*Mass_1/(x1**2+x2**2)**(3/2), x3]
    return dydt

#test function to see if a circular orbit for a single mass is formed
#takes arguments N (number of steps) and d_t (time interval)
def Single_Body():
    r_0 = [3,0,0]
    v_0 = Init_Speed(r_0)

    pos_vel_0 = np.append(r_0, v_0, axis = None)

    t = np.linspace(0, 100, 1000000)
    ans = int.solve_ivp(f_grav, [0,100],  pos_vel_0, t_eval= t)
    print(ans)

    plt.plot(ans.y[0,:], ans.y[1,:])
    plt.show()
    return

Single_Body()

