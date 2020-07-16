import numpy as np
from multiagent.core import World, Agent, Landmark, Obstacle
from multiagent.scenario import BaseScenario
import math
import random


class Scenario(BaseScenario):
    def make_world(self, num_agents=3, num_obs=3):
        world = World()
        # set any world properties first
        world.dim_c = 0
        self.num_agents = num_agents
        num_landmarks = 1
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.08
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.025
        # add obstacles
        world.obs = [Obstacle() for i in range(num_obs)]
        for i, obstacles in enumerate(world.obs):
            obstacles.id = i
            obstacles.name = 'obstacle %d' % i
            obstacles.collide = True
            obstacles.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def circle_points(self, obs):
        res = 20
        points = []
        for i in range(res):
            ang = 2*math.pi*i / res
            points.append(
                (obs.state.p_pos[0] + math.cos(ang)*obs.size, obs.state.p_pos[1] + math.sin(ang)*obs.size))

        return points

    def reset_world(self, world):

        # print("\n\n\n\n\n == == == == == == == == RESET WORLD == == == == == == == == ===\n\n\n\n\n")
        # set random initial states of the team
        # centroid = np.random.uniform(-1, +1, world.dim_p)
        # centroid = np.array([0, 0])
        # random properties for landmarks
        for _, landmark in enumerate(world.landmarks):
            landmark.color = (np.array([178, 34, 34])) / 255
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # random properties for obstacles
        for _, obs in enumerate(world.obs):
            obs.color = (np.array([192, 192, 192])) / 255
            obs.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            while ((world.landmarks[0].state.p_pos[0] - obs.state.p_pos[0])**2 + (world.landmarks[0].state.p_pos[0] - obs.state.p_pos[0])**2) <= (obs.size**2):
                obs.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            obs.geom = self.circle_points(obs)
            obs.state.p_vel = np.zeros(world.dim_p)
            # obs.state.prev_p_pos = obs.state.p_pos
            # obs.state.prev_p_vel = obs.state.p_vel

         # random properties for agents
        for agent in world.agents:
            color = np.random.choice(range(256), size=3) / 256
            agent.color = color
            agent.color_laser = np.append(color, 0.25)
            while(True):
                # print("Ponto do agente")
                # cx, cy = centroid + np.random.uniform(-0.5, +0.5, world.dim_p)
                cx, cy = np.random.uniform(-0.99, +0.99, world.dim_p)
                # cx = max(min(cx, 1.0), -1.0)
                # cy = max(min(cy, 1.0), -1.0)
                agent.state.p_pos = np.array([cx, cy])
                # print(agent.name, "Ponto do agente", agent.state.p_pos)
                if not self.is_collision(agent, world.obs):
                    break
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.prev_p_pos = list(agent.state.p_pos)
            agent.state.prev_p_vel = list(agent.state.p_vel)
            agent.state.c = np.zeros(world.dim_c)

    def wall(self, pos):
        return np.any(pos > 1.2) or np.any(pos < -1.2)

    def done(self, agent, world):

        goal_zone = 0.15  # Goal Zone
        ctrd_ = self.centroid(world)
        dist_t_ = np.sqrt(
            np.sum(np.square(ctrd_ - world.landmarks[0].state.p_pos)))

        done = self.wall(agent.state.p_pos) or dist_t_ <= goal_zone
        # done = np.any(agent.state.p_pos > 1.2) or np.any(agent.state.p_pos < -1.2) or dist_t_ <= goal_zone
        # done = dist_t_ <= goal_zone
        print(done)
        return done

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, enteties):
        for agent2 in enteties:
            if agent2 is agent1:
                continue
            delta_pos = agent1.state.p_pos - agent2.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = agent1.size + agent2.size
            if dist < dist_min:
                return True
        return False

    def centroid(self, world, previous=False):

        x_list = []
        y_list = []

        for a in world.agents:
            if previous:
                x_list.append(a.state.prev_p_pos[0])
                y_list.append(a.state.prev_p_pos[1])
            else:
                x_list.append(a.state.p_pos[0])
                y_list.append(a.state.p_pos[1])

        sum_x = np.sum(x_list)
        sum_y = np.sum(y_list)

        return np.array([sum_x, sum_y]) / len(x_list)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        re = -2
        wg = 50  # Factor do objectivo
        wc = 80  # Factor de colisões
        wf = 50  # Factor de proximidade entre agentes
        wp = 10  # Movimentos suaves
        r_goal = 5  # Goal reward
        r_collision = -2  # Collisions Penalty
        goal_zone = 0.18  # Goal Zone
        d = 0.2  # Threshold entre agentes

        Rg = 0
        Rc = 0
        Rf = 0
        Rw = 0  # WALL
        Rp = 0  # Movimentos suaves

        # ###  Distancia de centroid à landmark  ###
        # ctr = self.centroid(world)
        # if ctr == world.landmarks[0].state.p_pos:
        #     Rg = 7
        # else:

        ###   Distancia entre o agente e o objectivo ###
        dist_t_ = np.sqrt(
            np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))
        dist_t_ = round(dist_t_, 4)

        if dist_t_ <= goal_zone:
            Rg = r_goal
        else:
            dist_t = np.sqrt(
                np.sum(np.square(agent.state.prev_p_pos - world.landmarks[0].state.p_pos)))
            dist_t = round(dist_t, 4)
            Rg = dist_t - dist_t_

        if(len(world.agents) > 1):
            ###  Colisões com outros agentes  ###
            Rc = r_collision if self.is_collision(agent, world.agents) else Rc

            dists = []
            for a in world.agents:
                if a is agent:
                    continue
                dists.append(
                    min(0, d - np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))))
            ###  Distancia entre agentes    ###
            Rf = np.mean(dists)
            Rf = round(Rf, 4)

        ###  Colisões com outros obstaculos  ###
        if world.obs:
            Rc = r_collision if self.is_collision(agent, world.obs) else Rc

        ### Detect Wall ###
        if self.wall(agent.state.p_pos):
            # print("Agent=", agent.id, "  WALLLLLLLL")
            Rw = -100

        # v_ = math.sqrt(agent.state.p_vel[0]**2 + agent.state.p_vel[1]**2)
        # # print(v_)
        # if dist_t_ <= goal_zone and v_ < 0.01:
        #     Rp = 3
        # elif dist_t_ <= goal_zone and v_ >= 0.01:
        #     Rp = -v_

        Rc = Rc * wc
        Rg = Rg * wg
        Rf = Rf * wf
        # Rp = Rp * wp

        Rc = round(Rc, 4)
        Rg = round(Rg, 4)
        Rf = round(Rf, 4)
        # Rp = round(Rp, 4)

        # rew = re + Rg + Rw
        rew = re + Rg + Rc + Rf + Rw + Rp
        # rew = re + entropy + wg * Rg + wc * Rc + wf * Rf + Rw

        rew = round(rew, 4)

        # print("Agent=", agent.id, "   Rg=", (Rg), "   Rc=", (Rc), "   Rf=", (Rf), "   Rw=", Rw, "   Rp=", Rp, "   REW=", rew)
        return rew

    def reward_v2(self, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        re = -2
        wg = 100
        wc = 1
        wf = 100
        wp = 5
        r_goal = 3  # Goal reward
        r_collision = -50  # Collisions Penalty
        goal_zone = 0.15  # Goal Zone
        d = 0.5  # Threshold entre agentes

        Rg = 0
        Rc = 0
        Rf = 0

        ctrd_ = self.centroid(world)
        dist_t_ = np.sqrt(
            np.sum(np.square(ctrd_ - world.landmarks[0].state.p_pos)))

        if dist_t_ <= goal_zone:
            Rg = r_goal
        else:
            ctrd = self.centroid(world, previous=True)
            dist_t = np.sqrt(
                np.sum(np.square(ctrd - world.landmarks[0].state.p_pos)))
            Rg = dist_t - dist_t_

        dists = []
        for i, a in enumerate(world.agents):
            for b in world.agents[i+1:]:
                if b is a:
                    continue
                Rc = r_collision if self.is_collision(a, b) else Rc
                # if self.is_collision(a, b):
                #    Rc = r_collision
                # else:
                #    Rc = 0

                dists.append(
                    min(0, d - np.sqrt(np.sum(np.square(a.state.p_pos - b.state.p_pos)))))

        Rf = np.mean(dists)

        vel_var = []
        for a in world.agents:
            try:
                pv = np.dot(a.state.prev_p_vel, a.state.p_vel)

                v = math.sqrt(
                    a.state.prev_p_vel[0]**2 + a.state.prev_p_vel[1]**2)
                v_ = math.sqrt(a.state.p_vel[0]**2 + a.state.p_vel[1]**2)
                vel_var.append(- math.acos(pv / (v * v_)))
            except Exception as e:
                print(e)

        Rp = np.mean(vel_var)

        if np.isnan(Rp):
            Rp = 0.00

        # print(Rg, Rc, Rf, Rp)
        rew = re + wg * Rg + wc * Rc + wf * Rf + wp * Rp
        # print(rew)

        return rew

    def range_finder(self, agent, e_x, e_y, step=6):
        # print("============== Agent %d ================" % (agent.id))
        max_limit = math.ceil(2 * math.sqrt(2))
        ax, ay = agent.state.p_pos
        vx, vy = agent.state.p_vel

        # print("Position", ax, ay)
        # print("Direction", vx, vy)

        px = e_x - ax
        py = e_y - ay
        # print("Original position", e_x, e_y)

        if agent.state.p_vel.all() == 0:
            vx, vy = agent.direction
        else:
            agent.direction = agent.state.p_vel

        beta = math.atan2(vy, vx) - math.pi/2

        px_ = math.cos(beta) * px + math.sin(beta) * py
        py_ = -math.sin(beta) * px + math.cos(beta) * py
        # print("Transformed position", px_, py_)

        if py_ < 0:
            return None, None

        alpha_ = int(np.round(math.degrees(
            math.atan2((py_), (px_)))))

        # print("Prev Alpha = ", alpha_)
        alpha_ = math.floor(alpha_*step/180)
        # Quando está no limite e chega a 180 graus
        if alpha_ == step:
            alpha_ = step-1

        # print("New Alpha = ", alpha_)
        dist = np.sqrt(
            np.sum(np.square(agent.state.p_pos - (e_x, e_y))))

        dist = (max_limit - dist) / max_limit
        dist = round(dist, 4)
        return alpha_, dist

    def laser_values(self, agent, world):
        # print("============== Agent %d ================" % (agent.id))
        # print("P_POS=", agent.state.p_pos)
        # print("Direction=", agent.state.p_vel)
        step = 6
        laser = np.ones((step,))*(0)

        for other in world.agents:

            if other is agent:
                continue
            # print("Agent:")
            alpha_, dist = self.range_finder(
                agent, other.state.p_pos[0], other.state.p_pos[1], step=step)
            if alpha_ == None:
                continue
            laser[alpha_] = round(max(laser[alpha_], dist))
            # print("AGENT   Alpha -> ", alpha_, "  Dist -> ", dist)

        # print(world.obs, world.obs[0].geom)
        if (not world.obs) or (not world.obs[0].geom):
            # print("Laser -> ", laser, " Not Geom")
            return laser

        for obstacle_elem in world.obs:
            # print("OBS:")
            obs_points = obstacle_elem.geom
            for point in obs_points:
                # print("Point", point)
                x_, y_ = point
                alpha_, dist = self.range_finder(
                    agent, x_, y_, step=step)
                if alpha_ == None:
                    continue
                laser[alpha_] = round(max(laser[alpha_], dist))
                # print("OBS   Alpha -> ", alpha_, "  Dist -> ", dist)

        # print("Laser -> ", laser)
        return laser

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # agent_pos = [round(elem, 4) for elem in agent.state.p_pos]

        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos)

        # entity_pos = np.around(entity_pos, 4)
        other_agents = []
        for a in world.agents:
            if a is agent:
                continue
            other_agents.append(a.state.p_pos)

        # other_agents = np.around(other_agents, 4)
        laser = self.laser_values(agent, world)

        obs = np.concatenate([agent.state.p_vel] +
                             [agent.state.p_pos] + entity_pos + [laser])
        # obs = np.concatenate([agent.state.p_pos] + other_agents + entity_pos)

        # obs = [round(elem, 4) for elem in obs]
        # print(agent.name, "OBS: ", obs)
        return obs
