import numpy as np
import matplotlib.pyplot as plt
import math
from functools import lru_cache

class GridMDP:
    def __init__(self, reward_step=-0.04, gamma=0.9):
        self.rows = 3
        self.cols = 4
        self.terminal_states = {(0, 3): 1, (1, 3): -1}
        self.walls = {(1, 1)}
        self.step_reward = reward_step
        self.discount = gamma
        self.actions = ["U", "D", "L", "R"]
        self.p_intended = 0.8
        self.p_side = 0.1

    def inside(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and (r, c) not in self.walls

    def move(self, r, c, a):
        if a == "U": nr, nc = r - 1, c
        elif a == "D": nr, nc = r + 1, c
        elif a == "L": nr, nc = r, c - 1
        else: nr, nc = r, c + 1
        return (nr, nc) if self.inside(nr, nc) else (r, c)

    def sideways(self, a):
        if a in ["U", "D"]: return ["L", "R"]
        return ["U", "D"]

    def transitions(self, r, c, a):
        result = {}
        nr, nc = self.move(r, c, a)
        result[(nr, nc)] = result.get((nr, nc), 0) + self.p_intended
        for s in self.sideways(a):
            sr, sc = self.move(r, c, s)
            result[(sr, sc)] = result.get((sr, sc), 0) + self.p_side
        return result

    def reward(self, r, c):
        if (r, c) in self.terminal_states: return self.terminal_states[(r, c)]
        return self.step_reward


def value_iteration(env, tol=1e-4):
    V = np.zeros((env.rows, env.cols))
    while True:
        d = 0
        newV = V.copy()
        for r in range(env.rows):
            for c in range(env.cols):
                if (r, c) in env.walls or (r, c) in env.terminal_states:
                    continue
                vals = []
                for a in env.actions:
                    s = 0
                    for (nr, nc), p in env.transitions(r, c, a).items():
                        s += p * (env.reward(nr, nc) + env.discount * V[nr, nc])
                    vals.append(s)
                newV[r, c] = max(vals)
                d = max(d, abs(newV[r, c] - V[r, c]))
        V = newV
        if d < tol: break
    return V

def extract_policy(env, V):
    P = np.full((env.rows, env.cols), " ", dtype=str)
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in env.walls: P[r, c] = "W"; continue
            if (r, c) in env.terminal_states: P[r, c] = "T"; continue
            scores = {}
            for a in env.actions:
                s = 0
                for (nr, nc), p in env.transitions(r, c, a).items():
                    s += p * (env.reward(nr, nc) + env.discount * V[nr, nc])
                scores[a] = s
            P[r, c] = max(scores, key=scores.get)
    return P

def show_values(V):
    plt.imshow(V, cmap="viridis")
    plt.colorbar()
    plt.title("Gridworld Values")
    plt.show()

def show_policy(P):
    plt.figure()
    plt.title("Gridworld Policy")
    plt.axis("off")
    t = plt.table(cellText=P, cellLoc="center", loc="center")
    t.scale(1, 2)
    plt.show()

@lru_cache(None)
def pois(l, k):
    return (l**k * math.exp(-l)) / math.factorial(k)

def poisson_dist(mean, maxv=15):
    return np.array([pois(mean, i) for i in range(maxv + 1)])

class GbikeMDP:
    def __init__(self):
        self.maxB = 20
        self.max_move = 5
        self.gamma = 0.9
        self.req = (3, 4)
        self.ret = (3, 2)
        self.rent_val = 10
        self.move_cost = 2
        self.park_cost = 4

    def states(self):
        return [(i, j) for i in range(self.maxB + 1) for j in range(self.maxB + 1)]

    def actions(self):
        return list(range(-self.max_move, self.max_move + 1))

    def move(self, s1, s2, a):
        n1 = min(self.maxB, max(0, s1 - a))
        n2 = min(self.maxB, max(0, s2 + a))
        return n1, n2

    def move_fee(self, a):
        return abs(a) * self.move_cost - (self.move_cost if a > 0 else 0)

    def park_fee(self, a, b):
        return (self.park_cost if a > 10 else 0) + (self.park_cost if b > 10 else 0)

    def expected(self, n1, n2, a):
        m1, m2 = self.move(n1, n2, a)
        r = 0
        pr1 = poisson_dist(self.req[0])
        pr2 = poisson_dist(self.req[1])
        for k, p in enumerate(pr1): r += p * min(k, m1) * self.rent_val
        for k, p in enumerate(pr2): r += p * min(k, m2) * self.rent_val
        r -= self.move_fee(a)
        r -= self.park_fee(m1, m2)
        nxt = {}
        ret1 = poisson_dist(self.ret[0])
        ret2 = poisson_dist(self.ret[1])
        for k1, p1 in enumerate(ret1):
            for k2, p2 in enumerate(ret2):
                s1 = min(self.maxB, m1 + k1)
                s2 = min(self.maxB, m2 + k2)
                nxt[(s1, s2)] = nxt.get((s1, s2), 0) + p1 * p2
        return r, nxt

def eval_policy(mdp, policy, V, tol=1e-3):
    while True:
        d = 0
        for s1 in range(mdp.maxB + 1):
            for s2 in range(mdp.maxB + 1):
                a = policy[(s1, s2)]
                rew, nxt = mdp.expected(s1, s2, a)
                nv = 0
                for (ns1, ns2), p in nxt.items():
                    nv += p * (rew + mdp.gamma * V[(ns1, ns2)])
                d = max(d, abs(nv - V[(s1, s2)]))
                V[(s1, s2)] = nv
        if d < tol: break

def improve(mdp, policy, V):
    stable = True
    for s1 in range(mdp.maxB + 1):
        for s2 in range(mdp.maxB + 1):
            best = policy[(s1, s2)]
            bv = -1e18
            for a in mdp.actions():
                rew, nxt = mdp.expected(s1, s2, a)
                val = 0
                for (ns1, ns2), p in nxt.items():
                    val += p * (rew + mdp.gamma * V[(ns1, ns2)])
                if val > bv:
                    bv = val
                    best = a
            if best != policy[(s1, s2)]:
                policy[(s1, s2)] = best
                stable = False
    return stable

def solve_gbike():
    mdp = GbikeMDP()
    V = {(i, j): 0 for i in range(mdp.maxB + 1) for j in range(mdp.maxB + 1)}
    policy = {(i, j): 0 for i in range(mdp.maxB + 1) for j in range(mdp.maxB + 1)}
    while True:
        eval_policy(mdp, policy, V)
        if improve(mdp, policy, V):
            break
    return V, policy

def plot_values(V):
    M = np.zeros((21, 21))
    for i in range(21):
        for j in range(21):
            M[i, j] = V[(i, j)]
    plt.imshow(M, cmap="plasma")
    plt.colorbar()
    plt.title("Gbike Value Function")
    plt.xlabel("Bikes L2")
    plt.ylabel("Bikes L1")
    plt.show()

if __name__ == "__main__":
    g = GridMDP(0.02)
    V = value_iteration(g)
    P = extract_policy(g, V)
    print(V)
    print(P)
    show_values(V)
    show_policy(P)

    Vg, Pg = solve_gbike()
    plot_values(Vg)
