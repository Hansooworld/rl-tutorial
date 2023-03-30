import math,os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as ET

# Convert quaternion to Euler angle 
def quaternion_to_euler_angle(w, x, y, z):
    """
        w, x, y, z -> R, P, Y
    """
    y_sqr = y*y

    t_0 = +2.0 * (w*x + y*z)
    t_1 = +1.0 - 2.0 * (x*x + y_sqr)
    X = math.degrees(math.atan2(t_0, t_1))
	
    t_2 = +2.0 * (w*y - z*x)
    t_2 = +1.0 if t_2 > +1.0 else t_2
    t_2 = -1.0 if t_2 < -1.0 else t_2
    Y = math.degrees(math.asin(t_2))
	
    t_3 = +2.0 * (w*z + x*y)
    t_4 = +1.0 - 2.0 * (y_sqr + z*z)
    Z = math.degrees(math.atan2(t_3, t_4))
	
    return X, Y, Z

# Snapbot (4 legs) Environment
class Snapbot4EnvClass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                VERBOSE     = True,
                name        = 'Snapbot with 4 legs',
                leg_idx     = 1245,
                xml_path    = 'mujoco_random_env/xml/snapbot_4/robot_4_',
                frame_skip  = 5,
                condition   = None, 
                ctrl_coef   = 0,
                head_coef   = 0,
                render_mode = 'human',
                render_w    = 1500,
                render_h    = 1000,
                render_res  = 200
                ):
        """
            Initialize
        """
        self.VERBOSE    = VERBOSE
        self.name       = name
        self.leg_idx    = str(leg_idx)
        self.xml_path   = os.path.abspath(xml_path+'{}.xml'.format(self.leg_idx))
        self.frame_skip = frame_skip
        self.condition  = condition
        self.ctrl_coef  = ctrl_coef
        self.head_coef  = head_coef
        self.rand_mass  = None
        self.k_p = 0.2,
        self.k_i = 0.001,
        self.k_d = 0.01,
        self.joint_pos_deg_min = -np.array([43,40,43,40,43,40,43,40])
        self.joint_pos_deg_max = np.array([43,40,43,40,43,40,43,40])

        # Open xml
        try:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip,
            mujoco_bindings = 'mujoco_py'
            )
        except:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip
            )
        utils.EzPickle.__init__(self)

        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

        if self.VERBOSE:
            print("Snapbot(4legs) Environment")   
            print("Obs Dim: [{}] Act Dim: [{}] dt:[{}] Condition:[{}]".format(self.odim, self.adim, self.dt, condition))
            print("ctrl_coef:[{}] head_coef:[{}]".format(self.ctrl_coef, self.head_coef))

        # Timing
        self.hz = int(1/self.dt)
        # Reset
        self.reset()
        # Viewer setup
        if render_mode is not None:
            self.viewer_custom_setup(
                render_mode = render_mode,
                render_w    = render_w,
                render_h    = render_h,
                render_res  = render_res
                )

    def step(self, a):
        """
            Step forward
        """        
        # Before run
        x_pos_before      = self.get_body_com("torso")[0]
        y_pos_before      = self.get_body_com("torso")[1]
        heading_before    = self.get_heading()
        self.prev_state   = np.concatenate([self.sim.data.qpos.flat[2:], self.sim.data.qvel.flat])
        self.prev_torque  = a
        self.contact_data = np.array(self.sim.data.sensordata[4:])

        # Run sim
        self.do_simulation(a, self.frame_skip)
        x_pos_after   = self.get_body_com("torso")[0]
        y_pos_after   = self.get_body_com("torso")[1]
        heading_after = self.get_heading()
        x_diff        = x_pos_after - x_pos_before
        y_diff        = y_pos_after - y_pos_before
        heading_diff  = heading_after - heading_before
        # Set reward
        reward_forward = (x_pos_after-x_pos_before) / self.dt
        reward_survive = 1
        cost_ctrl      = self.ctrl_coef * np.square(a).sum()
        cost_heading   = self.head_coef * (heading_diff**2+y_diff**2)
            
        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        # self.r    = min(reward_forward, 1) + reward_survive - cost_ctrl - cost_heading
        self.r    = reward_forward
        self.info = dict({'reward_forward': reward_forward, 'cost_ctrl': cost_ctrl})
        
        # Done condition
        state   = self.state_vector()
        r, p, y = quaternion_to_euler_angle(state[3], state[4], state[5], state[6])
        notdone = np.isfinite(state).all and abs(r) < 170
        self.d  = not notdone
        
        return self.o, self.r, self.d, self.info
    
    def _get_obs(self):
        """
            Get observation
        """
        self.index = np.array(self.get_max_leg()).reshape(1)
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            self.prev_state,
            self.prev_torque,
            self.index
        ])
    
    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o

    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[9],q[10],q[13],q[14],q[17],q[18],q[21],q[22]]
            )*180.0/np.pi

    def get_heading(self):
        """
            Get z-axis rotation angle in [Deg]
        """
        q = self.data.get_body_xquat('torso')
        _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
        return z_deg
    
    def get_seonsor_data(self):
        """
            Get sensor data from touchsensors
        """
        l1 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[0]))
        l2 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[1]))
        l3 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[2]))
        l4 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[3]))
        ls = [l1, l2, l3, l4]
        return ls

    def get_max_leg(self):
        lst   = self.get_seonsor_data()
        score = 0
        index = 0
        for i,j in enumerate(lst):
            if j > score : 
                score = j
                index = i+1
        return index

    def get_time(self):
        """
            Get time in [Sec]
        """
        return self.sim.data.time
    
    def viewer_custom_setup(
        self,
        render_mode = 'human',
        render_w    = 1500,
        render_h    = 1000,
        render_res  = 200
        ):
        """
            View setup
        """
        self.render_mode = render_mode
        self.render_w    = render_w
        self.render_h    = render_h
        self.render_res  = render_res
        
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        self.viewer.cam.distance  = 8.0 # distance to plane (1.5)
        self.viewer.cam.elevation = -45 # elevation angle (-30)
        self.viewer.cam.lookat[0] = 0.0 # x-axis (let this follow the robot)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        
    def render_center(self):
        """
            Render with torso-centered
        """
        for d_idx in range(3): # follow the robot torso
            self.viewer.cam.lookat[d_idx] = self.get_body_com("torso")[d_idx]
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        return frame

if __name__ == "__main__":
    env = Snapbot4EnvClass(render_mode=None)
    done = False
    for i in range(1000):
        env.render()
        print(env.get_joint_pos_deg())
        _, _, done, _ = env.step(np.random.standard_normal(8)*1)
        if done:
            break

