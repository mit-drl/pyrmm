#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation
from cvxopt import solvers, matrix

#dynamics
def dynamics(y,t):
    dxdt = y[3]*np.cos(y[2])
    dydt = y[3]*np.sin(y[2])
    dttdt =y[4]
    dvdt = y[5]
    return [dxdt, dydt, dttdt, dvdt, 0, 0]

def map(veh_num, road_num):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.axis('equal')
    veh, veh_un, speed_handle = [], [], []
    for i in range(veh_num):
        if i == 0:
            line, = ax.plot([], [], color='r', linewidth=3.0)
        else:
            line, = ax.plot([], [], color='k', linewidth=3.0)
        veh.append(line)
        if i == 0:
            line, = ax.plot([], [], color='r', linestyle=':', marker='o')
        else:
            line, = ax.plot([], [], color='k', linestyle=':')
        veh_un.append(line)
        text_handle = ax.text([],[], '')
        speed_handle.append(text_handle)
    pred_handle, = ax.plot([], [], color='r', linestyle='--')

    road = []
    for i in range(road_num + 1): 
        if i == 0 or i == road_num:
            line, = ax.plot([], [], color='k', linewidth=4.0)
        else:
            line, = ax.plot([], [], color='k', linestyle='--')
        road.append(line)
    line, = ax.plot([], [], color='m', marker='^', linewidth=60) # one markers
    road.append(line)
    line, = ax.plot([], [], color='m', marker='v', linewidth=60) # one markers
    road.append(line)
    # speed_handle = ax.text([],[], '')
    plt.show(block = False)
    plt.pause(0.02)
    return ax, fig, veh, veh_un, road, speed_handle, pred_handle

def search_the_last(veh_state, active_num, current_id):
    veh_last_pos = veh_state[0,0]
    for i in range(active_num):
        if int(veh_state[i,5]) == current_id:
            veh_last_pos = max(veh_last_pos, veh_state[i,0])
    
    return veh_last_pos

def search_preceding(veh_state, active_num, veh_pos, lane_id):
    veh_pred_pos = veh_pos + 40
    veh_pred_vel = None
    for i in range(active_num):
        if int(veh_state[i,5]) == lane_id:
            if veh_state[i,0] > veh_pos and veh_state[i,0] < veh_pred_pos:
                veh_pred_pos = veh_state[i,0]
                veh_pred_vel = veh_state[i,3]
    
    return veh_pred_pos, veh_pred_vel

def search_vehicles(veh_state, active_num):
    corner = np.array([[-2.5, -1.0], [-2.5, +1.0], [+2.5, +1.0], [+2.5, -1.0]])
    indexes = []
    obs = []
    if(active_num > 1):
        for i in range(1, active_num, 1):
            foot = veh_state[i,0:2] + corner
            within = False
            for j in range(4):
                if(foot[j,0] - veh_state[0,0])**2 + (foot[j,1] - veh_state[0,1])**2 <= 20**2:
                    within = True
                    break
            if within == True:
                indexes.append(i)
                obs.append([veh_state[i,0], veh_state[i,1], veh_state[i,3], veh_state[i,5]])
    return indexes, obs

def sensing(veh_state, indexes):
    num = len(indexes)
    results = []
    for theta in np.arange(0, 2*np.pi, 2*np.pi/100.):
        if theta >= 0 and theta < np.pi/2:
            if np.abs(theta) < 0.001:
                sx, sy = 20, 0
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        if rx > 2.5 and ry >= -1 and ry <= 1:
                            sx = min(rx - 2.5, sx)
                            sy = 0
            else:
                sx, sy = 20*np.cos(theta), 20*np.sin(theta)
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        temp = np.tan(theta)*(rx - 2.5)
                        if rx > 2.5 and temp >= ry - 1 and temp <= ry + 1:
                            if rx - 2.5 < sx:
                                sx = rx - 2.5
                                sy = temp
                        temp = (ry-1)/np.tan(theta)
                        if ry > 1 and temp >= rx - 2.5 and temp <= rx + 2.5:
                            if ry - 1 < sy:
                                sx = temp
                                sy = ry - 1
        elif theta >= np.pi/2 and theta < np.pi:
            if np.abs(theta - np.pi/2) < 0.001:
                sx, sy = 0, 20
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        if ry > 1 and rx >= -2.5 and rx <= 2.5:
                            sx = 0
                            sy = min(ry - 1, sy)
            else:
                sx, sy = 20*np.cos(theta), 20*np.sin(theta)
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        temp = np.tan(theta)*(rx + 2.5)
                        if rx < -2.5 and temp >= ry - 1 and temp <= ry + 1:
                            if rx + 2.5 > sx:
                                sx = rx + 2.5
                                sy = temp
                        temp = (ry-1)/np.tan(theta)
                        if ry > 1 and temp >= rx - 2.5 and temp <= rx + 2.5:
                            if ry - 1 < sy:
                                sx = temp
                                sy = ry - 1
        elif theta >= np.pi and theta < 3*np.pi/2:
            if np.abs(theta - np.pi) < 0.001:
                sx, sy = -20, 0
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        if rx < -2.5 and ry >= -1 and ry <= 1:
                            sx = max(rx + 2.5, sx)
                            sy = 0
            else:
                sx, sy = 20*np.cos(theta), 20*np.sin(theta)
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        temp = np.tan(theta)*(rx + 2.5)
                        if rx < -2.5 and temp >= ry - 1 and temp <= ry + 1:
                            if rx + 2.5 > sx:
                                sx = rx + 2.5
                                sy = temp
                        temp = (ry+1)/np.tan(theta)
                        if ry < -1 and temp >= rx - 2.5 and temp <= rx + 2.5:
                            if ry + 1 > sy:
                                sx = temp
                                sy = ry + 1
        else:
            if np.abs(theta - 3*np.pi/2) < 0.001:
                sx, sy = 0, -20
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        if ry < -1 and rx >= -2.5 and rx <= 2.5:
                            sx = 0
                            sy = max(ry + 1, sy)
            else:
                sx, sy = 20*np.cos(theta), 20*np.sin(theta)
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        temp = np.tan(theta)*(rx - 2.5)
                        if rx > 2.5 and temp >= ry - 1 and temp <= ry + 1:
                            if rx - 2.5 < sx:
                                sx = rx - 2.5
                                sy = temp
                        temp = (ry+1)/np.tan(theta)
                        if ry < -1 and temp >= rx - 2.5 and temp <= rx + 2.5:
                            if ry + 1 > sy:
                                sx = temp
                                sy = ry + 1
        results.append([sx, sy])
    results = np.array(results)

    len0 = results.shape[0]
    dist = []
    for i in range(len0):
        temp = np.sqrt((results[i,0])**2 + (results[i,1])**2)
        dist.append(temp)
    dist = np.array(dist)
    return results, dist

def cvx_solver(Q, p, G, h):
    mat_Q = matrix(Q)
    mat_p = matrix(p)
    mat_G = matrix(G)
    mat_h = matrix(h)

    solvers.options['show_progress'] = False
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    return sol['x']

def CBF_QP(ego, other):
    x, y, theta, v = ego[0], ego[1], ego[2], ego[3]
    x0, y0, theta0, v0 = other[0], other[1], other[2], other[3]

    #############################################safety
    r = 15.5
    b = (x - x0)**2 + (y - (y0 - 12))**2 - r**2

    Lfb = 2*(x - x0)*(v*np.cos(theta) - v0) + 2*(y - (y0 - 12))*v*np.sin(theta)
    Lf2b = 2*(v*np.cos(theta) - v0)**2 + 2*(v*np.sin(theta))**2
    
    LgLfbu1 = 2*(x - x0)*(-v*np.sin(theta)) + 2*(y - (y0 - 12))*v*np.cos(theta)
    LgLfbu2 = 2*(x - x0)*np.cos(theta)  + 2*(y - (y0 - 12))*np.sin(theta)

    b_safe = Lf2b + 2*Lfb + b
    b_safe = np.reshape(b_safe, (1,1))
    A_safe = [-LgLfbu1, -LgLfbu2, 0, 0]
    A_safe = np.array(A_safe)
    A_safe = np.reshape(A_safe, (1,4))
    #############################################convergence
    yd, vd = 6, 17   #desired lane id:3, desired speed:17m/s
    k1, k2 = 1, 1
    V = (y - yd + k1*v*np.sin(theta))**2
    
    LfV = 2*(y - yd + k1*v*np.sin(theta))*v*np.sin(theta)
    LgVu1 = 2*(y - yd + k1*v*np.sin(theta))*(k1*v*np.cos(theta))
    LgVu2 = 2*(y - yd + k1*v*np.sin(theta))*(k1*np.sin(theta))

    A_clf = [LgVu1, LgVu2, -1, 0]
    A_clf = np.array(A_clf)
    A_clf = np.reshape(A_clf, (1,4))
    b_clf = -LfV - 1*V
    b_clf = np.reshape(b_clf, (1,1))

    V = (v - vd)**2
    LfV = 0
    LgVu1 = 0
    LgVu2 = 2*(v - vd)

    A_clfv = [LgVu1, LgVu2, 0, -1]
    A_clfv = np.array(A_clfv)
    A_clfv = np.reshape(A_clfv, (1,4))
    b_clfv = -LfV - 1*V
    b_clfv = np.reshape(b_clfv, (1,1))
    
    A_clf = np.concatenate((A_clf, A_clfv), axis = 0)
    b_clf = np.concatenate((b_clf, b_clfv), axis = 0)
   
    G = np.concatenate((A_safe, A_clf), axis = 0)
    h = np.concatenate((b_safe, b_clf), axis = 0)
    
    ################################ control bounds
    A_bound = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0]])
    b_bound = np.array([[0.5],[0.5],[0.1],[0.1]])

    G = np.concatenate((G, A_bound), axis = 0)
    h = np.concatenate((h, b_bound), axis = 0)

    Q = np.eye(4)
    Q[0,0] = 1000
    Q[2,2] = 1
    Q[3,3] = 1
    p = np.zeros((4,1))

    ################################## solve the QP
    rt = cvx_solver(Q, p, G, h)
    u = np.array([rt[0], rt[1]])
    return u



############################################# main function
veh_num = 50
road_num = 4
ax, fig, veh, veh_un, road, speed_handle, pred_handle = map(veh_num, road_num)

lidar_nb, control_nb, ego_nb, other_nb = [], [], [], []
for nbat in range(1):  #num of batch
                  
    rand_x, rand_y = 20., 6.
    theta0, speed0 = 0., 0.

    veh_state = np.zeros((veh_num, 8)) # x, y, theta, v, psi(unused), lane_id, pen_id, desired_spd, the first vehicle is ego
    veh_state[0, 0:4] = np.array([0, rand_y, theta0, 18+speed0]) 
    veh_state[0, 5:7] = np.array([3, 0])   #initial lane id
    active_num = 1
    pen_id = 1
    marker_pos = veh_state[0, 0] + 26


    state = []
    once = 1
    lidar, control, ego, other = [], [], [], []
    def update(niter):
        global state, ax, veh, veh_un, road, speed_handle, pred_handle, veh_num, road_num, veh_state, active_num, pen_id, marker_pos, once
        print('nbat: ', nbat, '|', 'niter: ', niter)
        dT = 0.1
        consider_a_single_vehicle_as_obstacle = True
        other_vehicles_consider_safety = False
        if consider_a_single_vehicle_as_obstacle:
            if once == 1:
                once = 0
                lane_id = 3

                speed = 13.5
                veh_state[active_num, 0:4] = np.array([veh_state[0, 0] + rand_x, 4*(lane_id-1) - 2, 0, speed])  # 20
                veh_state[active_num, 5:8] = np.array([lane_id, pen_id, speed])
                active_num += 1
                pen_id += 1
                if pen_id > 49:
                    pen_id = 1
        else:
            number = np.random.rand(1)
            if number[0] < 0.05:   #chance of generating a new vehicle at each iter.
                lane_id = int(np.ceil(np.random.rand(1)[0]*4))
                last_pos = search_the_last(veh_state, active_num, lane_id)
                if veh_state[0, 0] + 40 - last_pos > 16: #only generate a vehicle when there is safe space
                    speed = 14. + np.random.rand(1)[0]*4
                    veh_state[active_num, 0:4] = [veh_state[0, 0] + 40, 4*(lane_id-1) - 2, 0, speed]
                    veh_state[active_num, 5:8] = [lane_id, pen_id, speed]
                    active_num += 1
                    pen_id += 1
                    if pen_id > 49:
                        pen_id = 1
        
        if other_vehicles_consider_safety:
            for i in range(1,active_num,1):  # update speed for safety
                veh_pos = veh_state[i, 0]
                lane_id = int(veh_state[i,5])
                pred_pos, pred_vel = search_preceding(veh_state, active_num, veh_pos, lane_id)
                if pred_vel != None:
                    veh_state[i, 3] = min(veh_state[i, 7], pred_vel + pred_pos - veh_pos - 14)  # CBF for safety distance 6m
        
        # update lane id for ego
        lane_pos = [-2, 2, 6, 10]
        dis = np.array((lane_pos - veh_state[0,1])**2)
        min_dis = np.min(dis)
        min_idx = np.where(min_dis == dis)
        veh_state[0,5] = min_idx[0]+1

        #update marker
        if(marker_pos < veh_state[0,0] - 26):
            marker_pos = veh_state[0,0] + 26
        
        indexes,_ = search_vehicles(veh_state, active_num)  #Lidar
        results, dist = sensing(veh_state, indexes)
        lidar.append(dist)
        
        u = CBF_QP(veh_state[0,0:4], veh_state[1,0:4])
        
        ego_state = veh_state[0,0:4].tolist()
        ego_state.append(u[0])
        ego_state.append(u[1])
        
        ego.append(np.array([veh_state[0,0], veh_state[0,1],  veh_state[0,2],  veh_state[0,3]]))
        other.append(np.array([veh_state[1,0], veh_state[1,1],  veh_state[1,2],  veh_state[1,3]]))
        control.append(np.array([u[0], u[1]]))
        
        pred_state = veh_state[0:1, 0:2]
        
        dt = [0,0.1]
        rt = np.float32(odeint(dynamics,ego_state,dt))
        veh_state[0,0:4] = rt[1][0:4]

        for i in range(1, active_num, 1):
            veh_state[i, 0] += veh_state[i, 3]*dT
        
        
        cir_x = results[:,0]
        cir_y = results[:,1]
        corner = np.array([[-2.5, -1.0], [-2.5, +1.0], [+2.5, +1.0], [+2.5, -1.0], [-2.5, -1.0]])

        for i in range(active_num):
            if i == 0:
                rot = np.array([[np.cos(veh_state[i, 2]), -np.sin(veh_state[i, 2])],[np.sin(veh_state[i, 2]), np.cos(veh_state[i, 2])]])
                rot_corner = rot.dot(corner.transpose()).transpose()
                veh[int(veh_state[i, 6])].set_data(veh_state[i, 0] + rot_corner[:,0], veh_state[i, 1] + rot_corner[:,1])        
                veh_un[int(veh_state[i, 6])].set_data(veh_state[i, 0]  + cir_x, veh_state[i, 1] + cir_y) 
                speed_handle[int(veh_state[i, 6])].set_position((veh_state[i, 0], veh_state[i, 1]))
                speed_handle[int(veh_state[i, 6])].set_text(f"Speed: {veh_state[i,3]:>.2f} m/s")
                pred_handle.set_data(pred_state[:,0], pred_state[:,1]) 
            else:
                veh[int(veh_state[i, 6])].set_data(veh_state[i, 0] + corner[:,0], veh_state[i, 1] + corner[:,1])
                speed_handle[int(veh_state[i, 6])].set_position((veh_state[i, 0], veh_state[i, 1]))
                speed_handle[int(veh_state[i, 6])].set_text(f"Speed: {veh_state[i,3]:>.2f} m/s")

                d = np.sqrt((veh_state[i, 0] - veh_state[0, 0])**2 + (veh_state[i, 1] - veh_state[0, 1])**2)
                dv = np.abs(veh_state[0, 3] - veh_state[i, 3])
                if dv < 0.5:
                    dv = 0.5
                if d < 5.5:
                    d = 5.5
                if d > 20:
                    d = 20
                coe = 0.1*d + 0.5
                coe2 = 0.1*dv+0.95
                veh_un[int(veh_state[i, 6])].set_data(veh_state[i, 0]  + corner[:,0]*coe*coe2, veh_state[i, 1] + corner[:,1]*coe*coe2)
                ang = np.linspace(0,2*np.pi,100)
                r = 15.5
                xx = r*np.cos(ang)
                yy = r*np.sin(ang)
                if int(veh_state[i, 5]) == 1:
                    ox = veh_state[i, 0] + xx
                    oy = veh_state[i, 1] - 12 + yy
                elif int(veh_state[i, 5]) == 4:
                    ox = veh_state[i, 0] + xx
                    oy = veh_state[i, 1] + 12 + yy
                elif veh_state[i, 1] > veh_state[0, 1] + 0.1:
                    ox = veh_state[i, 0] + xx
                    oy = veh_state[i, 1] + 12 + yy
                else:
                    ox = veh_state[i, 0] + xx
                    oy = veh_state[i, 1] - 12 + yy
                veh_un[int(veh_state[i, 6])].set_data(ox, oy)

        for i in range(road_num + 1):
            road[i].set_data(veh_state[0, 0] + [-40, 40], [4*i - 4, 4*i - 4])
        road[-1].set_data(marker_pos, 13)
        road[-2].set_data(marker_pos, -5)

        i = 0
        while(i < active_num):    
            if veh_state[i, 0] < veh_state[0, 0] - 40:
                for j in range(active_num - i):
                    veh_state[i+j, :] = veh_state[i+j+1, :]
                active_num -= 1
                i -= 1
            i += 1
        ax.axis([-24 + veh_state[0, 0], 24 + veh_state[0, 0], -16, 16])
        
    

    ani = animation.FuncAnimation(fig, update, 99, fargs=[],   #change the simulation time step:99 (9.9s) to any other values
                                interval=25, blit=False, repeat = False)  # interval/ms, blit = False/without return

    ani.save('{:03d}'.format(nbat) +'.mp4')
    # plt.show()

    #save lidar info and controls
    lidar = np.array(lidar)
    control = np.array(control)
    ego = np.array(ego)
    other = np.array(other)
     
    tt = np.linspace(0, 9.9, 100)
    plt.figure(2)
    plt.cla()
    plt.plot(tt, control[:,0], 'r-', tt, control[:,1], 'b-')
    plt.savefig('{:03d}'.format(nbat))

    lidar_nb.append(lidar)
    control_nb.append(control)
    ego_nb.append(ego)
    other_nb.append(other)

lidar = np.array(lidar_nb)
control = np.array(control_nb)
ego = np.array(ego_nb)
other = np.array(other_nb)

data = {'lidar':lidar, 'ctrl':control, 'ego':ego, 'other':other}  #save all data
import pickle
output = open('data.pkl', 'wb')
pickle.dump(data, output)
output.close()
