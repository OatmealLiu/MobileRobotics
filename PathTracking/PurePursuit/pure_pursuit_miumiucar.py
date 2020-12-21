"""
Path tracking simulation
    - Bicycle Model
    - Pure pursuit control
    - Stanley(..ongoing)
    - Proportional control
Author: Mingxuan Liu
Ref:
    - <Automatic Steering Methods for Autonomous Automobile Path Tracking>
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class StateTape():
    """
    Record trajectory
    """
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []
        self.rear_x = []
        self.rear_y = []
    
    def append(self, t, x, y, yaw, v, rx, ry):
        self.x.append(x)
        self.y.append(y)
        self.yaw.append(yaw)
        self.v.append(v)
        self.t.append(t)
        self.rear_x.append(rx)
        self.rear_y.append(ry)

class ErrorTape:
    """
    Record errors
    """
    def __init__(self):
        self.errors = []
        self.tick = []
    
    def append(self, tick, error):
        self.errors.append(error)
        self.tick.append(tick)


class MiuMiuCar:
    """
    MiuMiuCar is a Bicycle Model equiped with:
        - Controllers: Pure pursuit steering control / Proportional speed control / ..Stanley(future)
        - Initializer:
            - Noise: make or not
            - Ini-Config: where you want MiuMiuCar start from
            - Steering Controller: Pure pursuit or Stanley(future)
            - Speed Conrtroller: P, PID and Gains

        - Tapes: error and state tapes
        - Target Course: you can feed a target course to MiuMiuCar, he will follow
        - Color painter: you can change the color of MiuMiuCar
    """

    def __init__(self,x=0.0, y=0.0, yaw=0.0,
                      v=0.0, b=2.0, vel_controller='p',
                      ang_controller='pure_pursuit', sensor=None,
                      target_path=None, make_noise=False,
                      ref_point='r', l=2.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.b = b
        self.rear_x = self.x - ((b / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((b / 2) * math.sin(self.yaw))
        self.ref_point = ref_point

        self.vel_controller = vel_controller
        self.ang_controller = ang_controller
        self.k = 0.1
        self.Kp = 1.0

        self.sensor = sensor

        if target_path == None:
            self.commanded_path_x = np.zeros(1)
            self.commanded_path_y = np.zeros(1)
            self.commanded_path_yaw = np.zeros(1)
            self.commanded_length = 1
        elif len(target_path) >= 3:
            self.commanded_length = len(target_path[0])
            self.commanded_path_x = target_path[0]
            self.commanded_path_y = target_path[1]
            self.commanded_path_yaw = target_path[2]
        else:
            self.commanded_length = len(target_path[0])
            self.commanded_path_x = target_path[0]
            self.commanded_path_y = target_path[1]
            self.commanded_path_yaw = np.zeros(self.commanded_path_x.shape)

        self.old_nearest_point_idx = None
        self.const_dist_lookAhead = l

        self.state_tape = StateTape()
        self.error_tape = ErrorTape()

        self.make_noise = make_noise
        self.noise_scale = 0.0
        self.this_noise = 0.0
        self.dt = 0.1

        self.car_length = 0.7,
        self.car_width = 2.0,
        self.car_fc = 'deeppink'
        self.car_ec = 'crimson'


    def show_config(self):
        print("Pose=({},{},{})\nVel={}\nk-gain={}\nKp={}".\
                format(self.x,self.y,self.yaw,
                       self.v,self.k,self.Kp))

    def update(self, acc, delta):
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += self.v / self.b * math.tan(delta) * self.dt 
        self.v += acc * self.dt
        self.rear_x = self.x - ((self.b / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.b / 2) * math.sin(self.yaw))

    def calc_dist(self, point_x, point_y):
        dx = 0.0
        dy = 0.0
        if self.make_noise:
            if self.ref_point == 'f':
                dx = self.x + self.this_noise - point_x
                dy = self.y + self.this_noise - point_y
            else:
                dx = self.rear_x + self.this_noise - point_x
                dy = self.y + self.this_noise - point_y
        else:
            if self.ref_point == 'f':
                dx = self.x - point_x
                dy = self.y - point_y
            else:
                dx = self.rear_x - point_x
                dy = self.y - point_y
        
        return math.hypot(dx, dy)

    def record_state(self, t):
        self.state_tape.append(
            t,
            self.x,
            self.y,
            self.yaw,
            self.v,
            self.rear_x,
            self.rear_y)
    
    def read_state_tape(self):
        return self.state_tape

    def record_error(self):
        for i in range(self.commanded_length):
            dx = [self.commanded_path_x[i] - ix for ix in self.state_tape.x]
            dy = [self.commanded_path_y[i] - iy for iy in self.state_tape.y]
            dists = np.hypot(dx,dy)
            ref_idx = np.argmin(dists)

            alpha_e = math.atan2(self.commanded_path_x[i] - self.state_tape.rear_x[ref_idx],
                                 self.commanded_path_x[i] - self.state_tape.rear_y[ref_idx])\
                                     - self.state_tape.yaw[ref_idx]
                                     
            cross_track_error = dists[ref_idx] * math.sin(alpha_e)
            self.error_tape.append(i, cross_track_error)

    def read_error_tape(self):
        return self.error_tape

    def set_ini_configuration(self, x, y, yaw, v, b, target_const_lookAhead):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.b = b
        self.rear_x = self.x - ((b / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((b / 2) * math.sin(self.yaw))
        self.const_dist_lookAhead = target_const_lookAhead
    
    def search_target_point(self):
        if self.make_noise:
            gaussian_noise = np.random.normal(loc=0.0,
                                              scale=self.noise_scale,
                                              size=1)
            self.this_noise = gaussian_noise
            sensor_rear_x = self.rear_x + gaussian_noise
            sensor_rear_y = self.rear_y + gaussian_noise
        else:
            sensor_rear_x = self.rear_x
            sensor_rear_y = self.rear_y

        if self.old_nearest_point_idx is None:
            dx = [sensor_rear_x - ix for ix in self.commanded_path_x]
            dy = [sensor_rear_y - iy for iy in self.commanded_path_y]
            dists = np.hypot(dx, dy)
            idx = np.argmin(dists)
            self.old_nearest_point_idx = idx
        else:
            idx = self.old_nearest_point_idx
            dist_this_idx = self.calc_dist(self.commanded_path_x[idx],
                                           self.commanded_path_y[idx])

            while True:
                if (1 + idx) >= self.commanded_length:
                    break
                dist_next_idx = self.calc_dist(self.commanded_path_x[1 + idx],
                                               self.commanded_path_y[1 + idx])
                if dist_next_idx > dist_this_idx:
                    break
                idx = idx + 1 if (1 + idx) < self.commanded_length else idx
                dist_this_idx = dist_next_idx

            self.old_nearest_point_idx = idx
        
        dist_lookAhead = self.k * self.v + self.const_dist_lookAhead
        # dist_lookAhead = self.const_dist_lookAhead


        while dist_lookAhead > self.calc_dist(self.commanded_path_x[idx],
                                              self.commanded_path_y[idx]):
            if (1 + idx) >= self.commanded_length:
                break
            idx += 1

        return idx, dist_lookAhead

    def set_ang_controller(self, ang_controller ='pure_pursuit', k=0.1):
        self.ang_controller = ang_controller
        self.k = k

    def pure_pursuit_control(self):
        target_idx, dist_lookAhead = self.search_target_point()
        if self.make_noise:
            sensor_rear_x = self.rear_x + self.this_noise
            sensor_rear_y = self.rear_y + self.this_noise
        else:
            sensor_rear_x = self.rear_x
            sensor_rear_y = self.rear_y

        if self.old_nearest_point_idx >= target_idx:
            target_idx = self.old_nearest_point_idx
        
        if target_idx < self.commanded_length:
            target_x = self.commanded_path_x[target_idx]
            target_y = self.commanded_path_y[target_idx]
        else:
            target_x = self.commanded_path_x[-1]
            target_y = self.commanded_path_y[-1]
        
        alpha = math.atan2(target_y - sensor_rear_y, target_x - sensor_rear_x)\
                - self.yaw

        delta = math.atan2(2.0 * self.b * math.sin(alpha) / dist_lookAhead,\
                            1.0)

        return delta, target_idx
    
    def stanley_control(self):
        pass

    def set_vel_controller(self, vel_controller ='p', Kp=1.0):
        self.vel_controller = vel_controller
        self.Kp = Kp

    def proportional_control(self, target_vel):
        acc = self.Kp * (target_vel - self.v)

        return acc

    def make_some_noise(self,make=False,scale=0.2):
        if make:
            self.make_noise = make
            self.noise_scale = scale

    def paint(self,length=1.5, width=50.0, fc='deeppink',ec='crimson'):
        self.car_length = length
        self.car_width = width
        self.car_fc = fc
        self.car_ec = ec

    def plot_car(self):
        plt.arrow(self.x, self.y,
                  self.car_length * math.cos(self.yaw),
                  self.car_length * math.sin(self.yaw),
                  fc=self.car_fc, ec=self.car_ec,
                  head_width=self.car_width, head_length=self.car_width,
                  alpha=0.5)
        plt.plot(self.x, self.y)

    def feed_target_path(self, target_path):
        if len(target_path) >= 3:
            self.commanded_length = len(target_path[0])
            self.commanded_path_x = target_path[0]
            self.commanded_path_y = target_path[1]
            self.commanded_path_yaw = target_path[2]
        else:
            self.commanded_length = len(target_path[0])
            self.commanded_path_x = target_path[0]
            self.commanded_path_y = target_path[1]
            self.commanded_path_yaw = np.zeros(self.commanded_path_x.shape)

    def calc_dist_2_goal(self):
        d = self.calc_dist(self.commanded_path_x[-1], self.commanded_path_y[-1])
        return d

    def run(self, target_vel, running_time, show_plot=False):
        target_vel = target_vel
        T = running_time
        time = 0.0
        # last_idx = self.commanded_length - 1

        target_idx, _ = self.search_target_point()
        self.record_state(time)
        stop_radius = self.calc_dist_2_goal()
        
        while T >= time and stop_radius > 0.8:
            acc_2bu = self.proportional_control(target_vel)
            delta_2bu, target_idx = self.pure_pursuit_control()


            self.update(acc_2bu, delta_2bu)
            stop_radius = self.calc_dist_2_goal()
            time += self.dt

            self.record_state(time)

            if show_plot:
                plt.cla()
                self.plot_car()
                plt.plot(self.commanded_path_x, self.commanded_path_y,
                         color="darkslateblue",
                         label="course",
                         linestyle='-')
                plt.plot(self.state_tape.x, self.state_tape.y,
                         color="deeppink",
                         label="trajectory",
                         linestyle='-')
                plt.plot(self.commanded_path_x[target_idx], 
                         self.commanded_path_y[target_idx],
                         "bo",
                         label="target",
                         alpha=0.5,
                         mec="navy",
                         mew=3.5,
                         ms=3.5)
                plt.axis("equal")
                plt.grid(True)
                plt.title("Speed[m/s]:{}  K:{}".format(str(target_vel),str(self.k)))
                plt.pause(0.0001)


        self.record_error()

        if show_plot:
            plt.cla()
            plt.plot(self.commanded_path_x, self.commanded_path_y,
                     color="darkslateblue",
                     linestyle='',
                     marker='.',
                     label="course",
                     alpha=0.7)
            plt.plot(self.state_tape.x, self.state_tape.y,
                     color="deeppink",
                     linestyle='-',
                     label="trajectory",
                     alpha=0.7)
            plt.legend()
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.axis("equal")
            plt.grid(True)
            plt.show()


def main():
    # Design Simulation

    # #Create experimental environments
    exp_environments = []

    # Sin-like path
    px_1 = np.arange(0,100,0.5)
    py_1 = np.array([math.sin(ix / 5.0) * ix / 2.0 for ix in px_1])
    px_1 *= 5.0
    exp_environments.append((px_1,py_1))

    # # Sin-like path
    # px_1 = np.arange(0,50,0.5)
    # py_1 = np.array([math.sin(ix / 5.0) * ix / 2.0 for ix in px_1])
    # px_1 *= 1.0
    # exp_environments.append((px_1,py_1))

    # Lane Change path
    px_2_1 = np.arange(0,120.5,0.5)
    px_2_2 = np.full(100,fill_value=120.0)
    px_2_3 = np.arange(120.0,370.5,0.5)
    px_2_4 = np.full(100,fill_value=370.0)
    px_2_5 = np.arange(370.0,500,0.5)
    px_2 = np.hstack((px_2_1, px_2_2,
                      px_2_3, px_2_4,
                      px_2_5))

    py_2_1 = np.zeros(px_2_1.shape)
    py_2_2 = np.arange(0.5,50.5,0.5)
    py_2_3 = np.full(px_2_3.shape,fill_value=50)
    py_2_4 = np.arange(50.0,0.0,-0.5)
    py_2_5 = np.zeros(px_2_5.shape)
    py_2 = np.hstack((py_2_1, py_2_2,
                      py_2_3, py_2_4,
                      py_2_5))
    exp_environments.append((px_2,py_2))

    # Create experimental parameter
    simulation_T = 300
    exp_velocities = [5.0, 10.0, 15.0, 20.0]
    # exp_velocities = [3.0, 6.0, 9.0, 12.0]

    exp_k_gains_1 = [0.1, 0.28, 0.46, 0.64, 0.82, 1.0]


    # exp_k_gains_2 = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    
    error_tape_box = [
                      [[],[],[],[]],
                      [[],[],[],[]]
                     ]
    # show=True
    # Start simulation: Pure Pursuit
    for envi in range(len(exp_environments)):
        for vi in range(len(exp_velocities)):
            for exp_k in exp_k_gains_1:
                print("--->Vel = {}, k-gain = {}".format(exp_velocities[vi],exp_k))
                
                miumiu_bicycle = MiuMiuCar()

                miumiu_bicycle.set_ini_configuration(x=-0.0, y=0.0,
                                                     yaw=0.0, v=0.0,
                                                     b=2.0, target_const_lookAhead=2.0)

                miumiu_bicycle.set_vel_controller(vel_controller='p', Kp=1.0)
                miumiu_bicycle.set_ang_controller(ang_controller='pure_pursuit', k=exp_k)

                miumiu_bicycle.make_some_noise(make=False,scale=0.5)

                if miumiu_bicycle.make_noise:
                    print("yooohoooo~~~~~~~")

                miumiu_bicycle.feed_target_path(exp_environments[envi])
                miumiu_bicycle.paint(length=0.1, width=4.5, fc='deeppink',ec='crimson')
                
                
                if exp_k == 0.28 and envi == 0 and vi == 0:
                    miumiu_bicycle.run(exp_velocities[vi], simulation_T, show_plot=False)
                elif exp_k == 0.64 and envi == 1 and vi == 0:
                    miumiu_bicycle.run(exp_velocities[vi], simulation_T, show_plot=False)
                else:
                    miumiu_bicycle.run(exp_velocities[vi],simulation_T, show_plot=False)

                # miumiu_bicycle.show_config()

                error_tape_box[envi][vi].append(miumiu_bicycle.read_error_tape())


    x_tik_0 = np.arange(0,225,25)
    y_tik_0 = np.arange(-2.5,3.0,0.5)
    x_tik_1 = np.arange(0,1400,200)
    y_tik_1 = np.arange(-8.0,8.5,1.0)
    # Plot comparison
    for envi in range(len(exp_environments)):
        plt.cla()
        canavas = [221, 222, 223, 224]
        k_colors = ['mediumspringgreen','deeppink','mediumturquoise','royalblue','tomato','lawngreen']

        legend_txt = ["k = {}".format(str(k)) for k in exp_k_gains_1]
        patches = []
        for c in k_colors:
            patches.append(mpatches.Patch(color=c, linestyle='-'))

        for vi in range(len(exp_velocities)):
            ax = plt.subplot(canavas[vi])

            for ki in range(len(exp_k_gains_1)):
                plt.plot(error_tape_box[envi][vi][ki].tick,
                         error_tape_box[envi][vi][ki].errors,
                         color=k_colors[ki],
                         linewidth=0.8
                         )

            plt.legend(handles=patches,labels=legend_txt,loc="upper left")
            plt.xlabel("Station[m]")
            plt.ylabel("Cross Track Error[m]")
            plt.title("Pure Pursuit at {} [m/s]".format(exp_velocities[vi]))
            plt.grid(True)
            if envi == 0:
                plt.xticks(x_tik_0)
                plt.yticks(y_tik_0)
            else:
                plt.xticks(x_tik_1)
                plt.yticks(y_tik_1)
            

        plt.show()

if __name__ == '__main__':
    print("---> Go Go Go Miumiu!!!")
    main()