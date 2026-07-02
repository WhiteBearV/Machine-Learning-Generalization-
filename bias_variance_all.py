# -*- coding: utf-8 -*-
"""
Bias-Variance & Learning Curves  (ใช้ normal equation ตามโจทย์)
โครงไฟล์:
    ส่วน A : TOOLKIT  (ฟังก์ชันของผู้ใช้ + เพิ่มเวอร์ชันผ่านจุดกำเนิด)
    ส่วน B : นิยาม 3 แบบจำลอง + 2 ฟังก์ชันเป้าหมาย ให้เรียกใช้แบบเดียวกัน
    ส่วน C : Bias-Variance (analytical + simulation)   -> ส่วนที่ 1-2 ของโจทย์
    ส่วน D : Learning Curves (+ ทดลองใส่ noise)         -> ส่วนที่ 3 ของโจทย์
รันไฟล์นี้ครั้งเดียวได้รูปทั้งสองใบ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# =====================================================================
# ส่วน A : TOOLKIT
#   convention ของ theta ตกลงกันทั้งไฟล์:  theta[0] = b (intercept)
#                                          theta[1:] = a (สัมประสิทธิ์)
# =====================================================================
def linear_regression_predict(X, theta):
    """ทำนายค่า: y_hat = X @ (สัมประสิทธิ์) + intercept"""
    return X @ theta[1:] + theta[0]

def compute_cost_linear(X, y, theta):
    """cost = MSE/2 (ครึ่งหนึ่งของ mean squared error)"""
    p = linear_regression_predict(X, theta)
    newY = p - y
    m = X.shape[0]
    return (1 / (2 * m)) * np.sum(newY ** 2)     # ใช้ sum(**2) ให้ทั่วไปกว่า dot

def compute_r2(y_true, y_pred):
    """ค่า R^2 ไว้ประเมินคุณภาพแบบจำลอง"""
    y_mean = np.mean(y_true)
    SSR = np.sum((y_true - y_pred) ** 2)
    SST = np.sum((y_true - y_mean) ** 2)
    if SST == 0:
        return 1.0
    return 1 - (SSR / SST)

def normal_equation(X, y):
    """normal equation แบบมี intercept: theta = pinv(Xa^T Xa) Xa^T y
       Xa = แปะคอลัมน์ 1 ไว้หน้า X  -> theta = [b, a1, a2, ...]"""
    m = X.shape[0]
    Xaug = np.hstack((np.ones((m, 1)), X))
    theta = np.linalg.pinv(Xaug.T @ Xaug) @ Xaug.T @ y
    return theta

def normal_equation_origin(X, y):
    """normal equation แบบผ่านจุดกำเนิด: ไม่มี intercept (บังคับ b=0)
       คืน theta = [0, a] ให้ convention ตรงกับตัวอื่น ใช้ predict ตัวเดิมได้"""
    theta_a = np.linalg.pinv(X.T @ X) @ X.T @ y
    return np.concatenate(([0.0], np.atleast_1d(theta_a)))

# =====================================================================
# ส่วน B : 3 แบบจำลอง + 2 ฟังก์ชันเป้าหมาย
#   ทุกแบบจำลองรับ (x, y) เป็น 1 มิติ แล้วคืน theta = [b, a] เสมอ
#   (แบบไหนไม่มีพจน์นั้น ค่าจะเป็น 0 เอง)
# =====================================================================
def fit_const(x, y):
    X0 = x.reshape(-1, 1)[:, :0]          # feature ว่าง (m,0) -> เหลือแต่ intercept
    return normal_equation(X0, y)          # theta = [b]  (a ไม่มี)

def fit_linear(x, y):
    X = x.reshape(-1, 1)                    # feature = x
    return normal_equation(X, y)           # theta = [b, a]

def fit_origin(x, y):
    X = x.reshape(-1, 1)
    return normal_equation_origin(X, y)    # theta = [0, a]

MODELS = {"constant":   fit_const,
          "linear":     fit_linear,
          "lin-origin": fit_origin}

TARGETS = {"sin(pi*x)": lambda x: np.sin(np.pi * x),
           "x^2":       lambda x: x ** 2}

def predict_on_grid(theta, xs):
    """ช่วยทำนายบน grid: รองรับทั้ง theta ที่มี/ไม่มีพจน์ a
       theta[0]=b เสมอ, ถ้ามี theta[1] คือ a ไม่งั้นถือ a=0"""
    a = theta[1] if len(theta) > 1 else 0.0
    return a * xs + theta[0]

# =====================================================================
# ส่วน C : BIAS-VARIANCE
# =====================================================================
rng = np.random.default_rng(42)

def simulate(f, fit, M=100_000):
    """Monte Carlo: สุ่มชุดข้อมูล 2 จุด M รอบ แล้วประมาณ bias, variance"""
    xs = np.linspace(-1, 1, 1001)
    G = np.zeros((M, len(xs)))                    # เก็บค่าทำนายทุกรอบ
    for m in range(M):
        x = rng.uniform(-1, 1, 2)
        theta = fit(x, f(x))
        G[m] = predict_on_grid(theta, xs)
    gbar = G.mean(axis=0)                         # แบบจำลองเฉลี่ย
    bias = np.mean((gbar - f(xs)) ** 2)           # เฉลี่ยเหนือ x
    var  = np.mean(G.var(axis=0))                 # var เหนือ D แล้วเฉลี่ยเหนือ x
    return bias, var, xs, gbar, G

def analytical(f, fit):
    """คำนวณ bias, variance ด้วยสูตรค่าคาดหวัง (อินทิเกรตตรง ๆ)
       เฉลี่ย a, b เหนือ (x1,x2)~U[-1,1]^2 (ความหนาแน่นร่วม 1/4)"""
    def ab(x1, x2):
        theta = fit(np.array([x1, x2]), f(np.array([x1, x2])))
        a = theta[1] if len(theta) > 1 else 0.0
        return a, theta[0]

    def E(g):
        v, _ = integrate.dblquad(lambda x2, x1: g(*ab(x1, x2)) * 0.25,
                                 -1, 1, -1, 1)
        return v

    Ea  = E(lambda a, b: a)
    Eb  = E(lambda a, b: b)
    Ea2 = E(lambda a, b: a * a)
    Eb2 = E(lambda a, b: b * b)
    Eab = E(lambda a, b: a * b)

    bias, _ = integrate.quad(
        lambda x: (Ea * x + Eb - f(x)) ** 2 * 0.5, -1, 1)
    var, _ = integrate.quad(
        lambda x: (Ea2 * x**2 + 2 * Eab * x + Eb2
                   - (Ea * x + Eb) ** 2) * 0.5, -1, 1)
    return bias, var

def run_bias_variance():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    print(f"{'target':<10} {'model':<11} | {'bias(ana)':>9} {'bias(sim)':>9}"
          f" | {'var(ana)':>9} {'var(sim)':>9} | {'Eout':>7}")
    print("-" * 78)
    for i, (fname, f) in enumerate(TARGETS.items()):
        for j, (mname, fit) in enumerate(MODELS.items()):
            b_a, v_a = analytical(f, fit)
            b_s, v_s, xs, gbar, G = simulate(f, fit)
            print(f"{fname:<10} {mname:<11} | {b_a:9.4f} {b_s:9.4f}"
                  f" | {v_a:9.4f} {v_s:9.4f} | {b_a+v_a:7.4f}")
            ax = axes[i, j]
            for g in G[:60]:
                ax.plot(xs, g, color="gray", alpha=0.15, lw=0.8)
            sd = G.std(axis=0)
            ax.fill_between(xs, gbar - sd, gbar + sd, color="tab:blue",
                            alpha=0.3, label="gbar ± sd")
            ax.plot(xs, f(xs), "g", lw=2, label="f(x)")
            ax.plot(xs, gbar, "r--", lw=2, label="gbar(x)")
            ax.set_ylim(-2, 2)
            ax.set_title(f"{fname}, {mname}\nbias={b_s:.3f}, var={v_s:.3f}")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("bias_variance.png", dpi=120)
    print("-> บันทึก bias_variance.png\n")

# =====================================================================
# ส่วน D : LEARNING CURVES (+ noise)
# =====================================================================
def learning_curve(f, fit, Ns, sigma=0.0, trials=3000):
    """คืน (Ein เฉลี่ย, Eout เฉลี่ย) ต่อแต่ละ N
       Ein  = MSE บนชุดฝึก (มี noise)
       Eout = MSE เทียบ f จริง บน grid (คือค่าคาดหวังเหนือ x)"""
    xs = np.linspace(-1, 1, 400)
    Ein, Eout = [], []
    for N in Ns:
        ein = eout = 0.0
        for _ in range(trials):
            x = rng.uniform(-1, 1, N)
            y = f(x) + sigma * rng.standard_normal(N)
            theta = fit(x, y)
            X = x.reshape(-1, 1)
            # Ein: เทียบกับ y ที่มี noise -> ใช้ compute_cost*2 = MSE
            ein  += 2 * compute_cost_linear(X, y, _pad(theta))
            # Eout: เทียบกับ f จริงบน grid
            eout += np.mean((predict_on_grid(theta, xs) - f(xs)) ** 2)
        Ein.append(ein / trials)
        Eout.append(eout / trials)
    return np.array(Ein), np.array(Eout)

def _pad(theta):
    """ทำให้ theta ยาว 2 เสมอ ([b] -> [b,0]) เพื่อให้ predict ใช้ได้"""
    return theta if len(theta) > 1 else np.array([theta[0], 0.0])

def run_learning_curves(fname="sin(pi*x)"):
    f = TARGETS[fname]
    Ns = np.arange(2, 61, 2)
    sigmas = [0.0, 0.3]
    colors = {"constant": "tab:blue", "linear": "tab:red",
              "lin-origin": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, sigma in zip(axes, sigmas):
        for name, fit in MODELS.items():
            Ein, Eout = learning_curve(f, fit, Ns, sigma)
            ax.plot(Ns, Eout, "-",  color=colors[name], label=f"{name} Eout")
            ax.plot(Ns, Ein, "--",  color=colors[name], alpha=0.6,
                    label=f"{name} Ein")
        if sigma > 0:
            ax.axhline(sigma**2, color="k", ls=":", lw=1,
                       label=f"noise floor={sigma**2:.2f}")
        ax.set_title(f"f(x)={fname}, noise sigma={sigma}")
        ax.set_xlabel("N (training size)")
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("MSE")
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=120)
    print("-> บันทึก learning_curves.png")

# =====================================================================
if __name__ == "__main__":
    run_bias_variance()
    run_learning_curves()
