import numpy as np
import matplotlib.pyplot as plt
import os
from fpdf import FPDF

os.makedirs("lab6_results", exist_ok=True)

class Hopfield:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))
    def train(self, patterns):
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            p = p.reshape(-1, 1)
            self.W += p @ p.T
        np.fill_diagonal(self.W, 0)
        self.W /= self.n
    def recall(self, pattern, steps=20):
        s = pattern.copy()
        energy = []
        for _ in range(steps):
            for i in range(self.n):
                h = np.dot(self.W[i], s)
                s[i] = 1 if h >= 0 else -1
            e = -0.5 * np.dot(s.T, self.W @ s)
            energy.append(e)
        return s, energy

def gen_patterns(n, k):
    return [np.random.choice([-1, 1], size=n) for _ in range(k)]

def q1():
    n = 100
    net = Hopfield(n)
    pats = gen_patterns(n, 5)
    net.train(pats)
    base = pats[0]
    noisy = base.copy()
    frac = 0.25
    idx = np.random.choice(n, int(n * frac), replace=False)
    noisy[idx] *= -1
    rec, e = net.recall(noisy)
    acc = np.sum(rec == base) / n
    plt.plot(e)
    plt.title("Energy Q1")
    plt.savefig("lab6_results/Q1_energy.png")
    plt.close()
    return acc, frac

def q2():
    N = 8
    M = N * N
    A = B = 4
    W = np.zeros((M, M))
    th = np.zeros(M)
    idx = lambda i, j: i * N + j
    for i in range(N):
        for j in range(N):
            u = idx(i, j)
            for k in range(N):
                if k != j: W[u, idx(i, k)] -= A
                if k != i: W[u, idx(k, j)] -= B
            th[u] = -(A + B)
    v = np.random.choice([0, 1], size=M)
    e = []
    for _ in range(400):
        i = np.random.randint(0, M)
        h = np.dot(W[i], v) - th[i]
        v[i] = 1 if h > 0 else 0
        en = -0.5 * np.dot(v.T, W @ v) + np.dot(th, v)
        e.append(en)
    board = v.reshape((N, N))
    plt.subplot(1, 2, 1)
    plt.plot(e)
    plt.title("Energy Q2")
    plt.subplot(1, 2, 2)
    plt.imshow(board, cmap="gray_r")
    plt.title("Rook Board")
    plt.tight_layout()
    plt.savefig("lab6_results/Q2_rook.png")
    plt.close()
    return board

def q3():
    n = 10
    D = np.random.randint(10, 100, (n, n))
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    V = np.random.rand(n, n)
    A = B = 500
    C = 200
    sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))
    E = []
    for _ in range(3000):
        dV = np.zeros_like(V)
        for i in range(n):
            for p in range(n):
                t1 = A * (np.sum(V[i]) - 1)
                t2 = B * (np.sum(V[:, p]) - 1)
                t3 = sum(D[i, j] * (V[j, (p + 1) % n] + V[j, (p - 1) % n]) for j in range(n) if j != i)
                dV[i, p] = -A*t1 - B*t2 - C*t3
        V = sig(V + dV * 0.001)
        E.append((A/2)*np.sum((np.sum(V, axis=1)-1)**2) + (B/2)*np.sum((np.sum(V, axis=0)-1)**2))
    tour = np.argmax(V, axis=0)
    length = sum(D[tour[i], tour[(i+1) % n]] for i in range(n))
    plt.subplot(1, 2, 1)
    plt.plot(E)
    plt.title("Energy Q3")
    plt.subplot(1, 2, 2)
    plt.imshow(V, cmap="plasma")
    plt.title("Activation")
    plt.tight_layout()
    plt.savefig("lab6_results/Q3_tsp.png")
    plt.close()
    return tour, length

def pdf(acc, frac, board, tour, length):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Lab 6: Hopfield Network Experiments", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Q1 recovered accuracy: {acc*100:.2f}%, corruption: {int(frac*100)}%\n")
    pdf.image("lab6_results/Q1_energy.png", w=170)
    pdf.multi_cell(0, 8, "Q2 Eight-Rook Solution:")
    pdf.image("lab6_results/Q2_rook.png", w=170)
    pdf.multi_cell(0, 8, f"Q3 TSP tour: {tour.tolist()}\nTotal length: {length:.2f}")
    pdf.image("lab6_results/Q3_tsp.png", w=170)
    pdf.output("lab6_results/Lab6_Report.pdf")

if __name__ == "__main__":
    acc, frac = q1()
    board = q2()
    tour, length = q3()
    pdf(acc, frac, board, tour, length)
