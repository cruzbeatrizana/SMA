import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import math
from multiagent.multi_discrete import MultiDiscrete
from .core import Obstacle, Agent

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!


class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True, width=700, height=700):

        self.width = width
        self.height = height
        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(
            world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(
            world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(
                    low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(
                    low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete(
                        [[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # all agents get the same reward in cooperative case
        #reward_n = self._get_reward()
        #reward_n = [reward_n] * self.n
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def agent_rotation(self, v):
        vx, vy = v

        beta = math.atan2(vy, vx) - math.pi/2

        return beta

    def centroid(self, world, previous=False):

        x_avg = y_avg = 0
        n = len(world.agents)
        for a in world.agents:
            if previous:
                x_avg += (a.state.prev_p_pos[0]) / n
                y_avg += (a.state.prev_p_pos[1]) / n
            else:
                x_avg += (a.state.p_pos[0]) / n
                y_avg += (a.state.p_pos[1]) / n

        return np.array([x_avg, y_avg])

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent:
                        continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' +
                                agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(self.width, self.height)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.arc_geoms_xform = []
            self.center_xform = []
            self.render_geoms_xform = []
            self.obs_geoms = []
            for entity in self.world.entities:
                xform = rendering.Transform()
                if isinstance(entity, Agent):
                    geom = rendering.make_triangle(entity.size)
                    geom.set_color(*entity.color)
                    direction = entity.state.p_vel
                    beta = self.agent_rotation(direction)
                    arc_v = []
                    for angle_d in range(0, 180):
                        angle = math.radians(angle_d)

                        x = math.cos(angle) * entity.size * 4
                        y = math.sin(angle) * entity.size * 4
                        arc_v.append([x, y])

                    # Make arc
                    polygon = rendering.make_polygon(arc_v)
                    polygon.set_color(*entity.color_laser)

                    xform_arc = rendering.Transform()
                    polygon.add_attr(xform_arc)
                    xform_arc.set_translation(*entity.state.p_pos)
                    xform_arc.set_rotation(beta)

                    self.render_geoms.append(polygon)
                    self.arc_geoms_xform.append(xform_arc)

                    # Make center point
                    point = rendering.make_circle(
                        radius=0.03)  # rendering.Point()
                    point.set_color(0, 0, 0)

                    xform_center = rendering.Transform()
                    point.add_attr(xform_center)
                    xform_center.set_translation(*entity.state.p_pos)

                    self.render_geoms.append(point)
                    self.center_xform.append(xform_center)

                elif isinstance(entity, Obstacle):
                    print(entity.size)
                    obs = rendering.make_circle(radius=entity.size)
                    obs.set_color(*entity.color)
                    obs_xform = rendering.Transform()
                    obs.add_attr(obs_xform)
                    obs_xform.set_translation(*entity.state.p_pos)
                    # self.obs_geoms.append(obs)
                    self.render_geoms.append(obs)
                    self.render_geoms_xform.append(obs_xform)
                    continue
                else:
                    geom = rendering.make_circle(entity.size)
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                xform.set_rotation(beta)

                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            center_coord = self.centroid(self.world)
            center_circle_geom = rendering.make_circle(0.02)

            center_circle_xform = rendering.Transform()
            center_circle_geom.add_attr(center_circle_xform)
            center_circle_xform.set_translation(*center_coord)

            self.render_geoms.append(center_circle_geom)
            self.render_geoms_xform.append(center_circle_xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent

            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(
                pos[0]-cam_range, pos[0]+cam_range, pos[1]-cam_range, pos[1]+cam_range)

            center_coord = self.centroid(self.world)
            center_circle_xform = self.render_geoms_xform[-1]
            center_circle_xform.set_translation(*center_coord)

            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                if 'agent' in entity.name:
                    direction = entity.state.p_vel
                    beta = self.agent_rotation(direction)

                    xform_arc = self.arc_geoms_xform[e]
                    xform = self.render_geoms_xform[e]

                    # Translate arc
                    xform_arc.set_translation(*entity.state.p_pos)

                    # Rotate triangle and arc
                    xform_arc.set_rotation(beta)
                    xform.set_rotation(beta)

                    # Translate center
                    xform_center = self.center_xform[e]
                    xform_center.set_translation(*entity.state.p_pos)

                self.render_geoms_xform[e].set_translation(
                    *entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(
                return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(
                        distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, np.array(done_n), info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
