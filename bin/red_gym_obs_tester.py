from red_env_constants import *

DISCOVERY_POINTS = [
    (6, 2, 40), (8, 12, 0),
    (8, 10, 0), (16, 18, 0), (15, 7, 0), (15, 2, 0), (8, 5, 0),
    (0, 2, 37), (2, 1, 37), (3, 1, 38), (0, 2, 38), (7, 7, 38),
    (2, 17, 0), (18, 2, 37), (2, 17, 0), (18, 2, 37), (2, 17, 0), (18, 2, 37),
    (6, 1, 39),
    (0, 2, 37), (2, 1, 37),
    (5, 8, 40), (0, 8, 40), (0, 11, 40), (3, 11, 40),  (5, 8, 40), (0, 8, 40), (0, 11, 40), (3, 11, 40), (5, 8, 40), (0, 8, 40), (0, 11, 40), (3, 11, 40),  (5, 8, 40), (0, 8, 40), (0, 11, 40), (3, 11, 40),
]

MAX_DISCOVERY = len(DISCOVERY_POINTS)


class RedGymObsTester:
    def __init__(self, env):
        if env.env.debug:
            print('**** RedGymObsTester ****')

        self.env = env
        self.discovery_index = 0
        self.p2p_found = 0

    def pallet_town_point_nav(self):
        x_pos, y_pos, map_n = self.env.get_current_location()
        reward = 0

        if (DISCOVERY_POINTS[self.discovery_index][0] == x_pos and
                DISCOVERY_POINTS[self.discovery_index][1] == y_pos and
                DISCOVERY_POINTS[self.discovery_index][2] == map_n):
            reward = 10
            self.p2p_found += 1

            self.discovery_index += 1
            if self.discovery_index == MAX_DISCOVERY:
                self.discovery_index = 0

        return reward
