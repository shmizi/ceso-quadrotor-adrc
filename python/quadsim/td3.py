import copy

import numpy as np


class MLP:
    def __init__(self, sizes, out_act, rng):
        self.W = [rng.uniform(-1, 1, (o, i))/np.sqrt(i) for i, o in zip(sizes[:-1], sizes[1:])]
        self.W[-1] *= 3e-3*np.sqrt(sizes[-2])
        self.b = [np.zeros(o) for o in sizes[1:]]
        self.out_act = out_act

        self.m = [np.zeros_like(p) for p in self.W+self.b]
        self.v = [np.zeros_like(p) for p in self.W+self.b]
        self.t = 0

    def forward(self, x):
        self.cache = [x]
        for j, (w, b) in enumerate(zip(self.W, self.b)):
            x = x @ w.T+b
            if j < len(self.W)-1:
                x = np.maximum(x, 0.)
            elif self.out_act == "tanh":
                x = np.tanh(x)
            self.cache.append(x)
        return x

    def backward(self, grad):
        gw, gb = [None]*len(self.W), [None]*len(self.W)

        for j in reversed(range(len(self.W))):
            out = self.cache[j+1]
            if j < len(self.W)-1:
                grad = grad*(out > 0)
            elif self.out_act == "tanh":
                grad = grad*(1-out**2)

            gw[j], gb[j] = grad.T @ self.cache[j], grad.sum(0)
            grad = grad @ self.W[j]

        self.grads = gw+gb
        return grad

    def step(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k, (p, g) in enumerate(zip(self.W+self.b, self.grads)):
            self.m[k] = b1*self.m[k]+(1-b1)*g
            self.v[k] = b2*self.v[k]+(1-b2)*g*g
            p -= lr*(self.m[k]/(1-b1**self.t))/(np.sqrt(self.v[k]/(1-b2**self.t))+eps)

    def soft_update(self, source, tau):
        for p, q in zip(self.W+self.b, source.W+source.b):
            p *= 1-tau
            p += tau*q

    def export(self):
        return [{"W": w.tolist(), "b": b.tolist()} for w, b in zip(self.W, self.b)]


class Replay:
    def __init__(self, size, obs_dim, act_dim):
        self.s, self.s2 = np.zeros((size, obs_dim)), np.zeros((size, obs_dim))
        self.a, self.r, self.d = np.zeros((size, act_dim)), np.zeros((size, 1)), np.zeros((size, 1))
        self.size, self.count = size, 0

    def add(self, s, a, r, s2, d):
        j = self.count % self.size
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j] = s, a, r, s2, d
        self.count += 1

    def sample(self, n, rng):
        j = rng.integers(0, min(self.count, self.size), n)
        return self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j]


class Agent:
    def __init__(self, h, rng):
        self.h, self.rng = h, rng
        self.obs_dim, self.act_dim = (15 if h["observe_gains"] else 12), 3
        hidden = list(h["hidden_layers"])

        self.actor = MLP([self.obs_dim]+hidden+[self.act_dim], "tanh", rng)
        self.critics = [MLP([self.obs_dim+self.act_dim]+hidden+[1], "linear", rng) for _ in range(2)]
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_targets = [copy.deepcopy(q) for q in self.critics]

        self.buffer = Replay(h["buffer_size"], self.obs_dim, self.act_dim)
        self.updates = 0

    def act(self, obs):
        return self.actor.forward(obs[None])[0]

    def explore(self, obs):
        if self.buffer.count < self.h["warmup_steps"]:
            return self.rng.uniform(-1, 1, self.act_dim)
        return np.clip(self.act(obs)+self.rng.normal(0, self.h["exploration_std"], self.act_dim), -1, 1)

    def train_step(self):
        for _ in range(self.h["updates_per_step"]):
            self.update()

    def update(self):
        h, rng = self.h, self.rng
        if self.buffer.count < max(h["warmup_steps"], h["batch_size"]):
            return

        s, a, r, s2, d = self.buffer.sample(h["batch_size"], rng)
        noise = np.clip(rng.normal(0, h["target_noise_std"], a.shape),
                        -h["target_noise_clip"], h["target_noise_clip"])
        a2 = np.clip(self.actor_target.forward(s2)+noise, -1., 1.)

        x2 = np.hstack([s2, a2])
        y = r+h["discount"]*(1-d)*np.minimum(self.critic_targets[0].forward(x2),
                                             self.critic_targets[1].forward(x2))

        x = np.hstack([s, a])
        for q in self.critics:
            q.backward(2*(q.forward(x)-y)/len(s))
            q.step(h["learning_rate"])

        self.updates += 1
        if self.updates % h["policy_delay"] == 0:
            self.update_actor(s)

    def update_actor(self, s):
        h = self.h
        pi = self.actor.forward(s)
        self.critics[0].forward(np.hstack([s, pi]))
        grad = self.critics[0].backward(-np.ones((len(s), 1))/len(s))

        self.actor.backward(grad[:, self.obs_dim:])
        self.actor.step(h["learning_rate"])

        self.actor_target.soft_update(self.actor, h["soft_update"])
        for q, qt in zip(self.critics, self.critic_targets):
            qt.soft_update(q, h["soft_update"])
