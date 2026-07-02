# -*- coding: utf-8 -*-
"""
ส่วนที่ 1-2: หา Bias และ Variance ของแบบจำลอง 3 แบบ
    H0 : ค่าคงที่            h(x) = b
    H1 : เชิงเส้น             h(x) = a*x + b
    H1o: เชิงเส้นผ่านจุดกำเนิด  h(x) = a*x
กับฟังก์ชันเป้าหมาย 2 ฟังก์ชัน
    f1(x) = sin(pi*x)
    f2(x) = x^2
ข้อมูล: สุ่ม x จำนวน N=2 จุด จากการแจกแจงเอกรูปในช่วง [-1, 1]

วิธี analytical  -> คำนวณค่าคาดหวัง (integral) ตรง ๆ ด้วย numerical integration
วิธี simulation  -> Monte Carlo สุ่มชุดข้อมูลหลาย ๆ ชุดแล้วเฉลี่ย
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

rng = np.random.default_rng(42)   # ตัวสุ่ม (กำหนด seed ให้ผลซ้ำได้)

# ---------------------------------------------------------------
# 1) ฟังก์ชัน fit แบบจำลองด้วย normal equation: w = (X^T X)^-1 X^T y
#    ใช้ np.linalg.lstsq ซึ่งแก้สมการเดียวกันแต่เสถียรกว่า
#    คืนค่า w = [a, b] เสมอ (แบบจำลองไหนไม่มีพจน์นั้นให้เป็น 0)
# ---------------------------------------------------------------
def fit_const(x, y):
    X = np.ones((len(x), 1))              # มีแต่คอลัมน์ค่าคงที่
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.array([0.0, w[0]])          # a=0, b=w

def fit_linear(x, y):
    X = np.column_stack([x, np.ones(len(x))])   # คอลัมน์ [x, 1]
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.array([w[0], w[1]])          # a, b

def fit_linear_origin(x, y):
    X = x.reshape(-1, 1)                   # มีแต่คอลัมน์ x (ไม่มี bias)
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.array([w[0], 0.0])           # a=w, b=0

MODELS = {"constant  h(x)=b":       fit_const,
          "linear    h(x)=ax+b":    fit_linear,
          "lin-origin h(x)=ax":     fit_linear_origin}

TARGETS = {"sin(pi*x)": lambda x: np.sin(np.pi * x),
           "x^2":       lambda x: x ** 2}

# ---------------------------------------------------------------
# 2) วิธี SIMULATION (Monte Carlo)
#    - สุ่มชุดข้อมูล M ชุด ชุดละ N=2 จุด แล้ว fit เก็บ (a, b) ไว้
#    - gbar(x)  = ค่าเฉลี่ยของ g_D(x) เหนือทุกชุดข้อมูล
#    - bias     = E_x[ (gbar(x) - f(x))^2 ]
#    - variance = E_x[ E_D[ (g_D(x) - gbar(x))^2 ] ]
#    ค่าเฉลี่ยเหนือ x ประมาณด้วย grid ละเอียดบน [-1,1]
# ---------------------------------------------------------------
def simulate(f, fit, M=100_000, N=2):
    W = np.zeros((M, 2))                       # เก็บ (a, b) ของทุกรอบ
    for m in range(M):
        x = rng.uniform(-1, 1, N)              # สุ่มข้อมูล 2 จุด
        y = f(x)                               # ไม่มี noise ในส่วนนี้
        W[m] = fit(x, y)

    xs = np.linspace(-1, 1, 1001)              # grid สำหรับเฉลี่ยเหนือ x
    G = np.outer(W[:, 0], xs) + W[:, 1:2]      # ค่าทำนายทุกรอบ: shape (M, 1001)

    gbar = G.mean(axis=0)                      # แบบจำลองเฉลี่ย gbar(x)
    bias = np.mean((gbar - f(xs)) ** 2)        # เฉลี่ยเหนือ x
    var  = np.mean(G.var(axis=0))              # var เหนือ D แล้วเฉลี่ยเหนือ x
    return bias, var, xs, gbar, G

# ---------------------------------------------------------------
# 3) วิธี ANALYTICAL
#    ใช้สูตรค่าคาดหวังโดยตรง: E[...] = integral ... p(x) dx
#    โดย p(x1) = p(x2) = 1/2 บน [-1,1] (เอกรูป)
#    ขั้นตอน:
#      3.1 หา gbar(x) = E_D[g_D(x)] = อินทิเกรตพารามิเตอร์เหนือ (x1, x2)
#      3.2 bias = ∫ (gbar(x)-f(x))^2 * (1/2) dx
#      3.3 var  = ∫ E_D[g_D(x)^2] - gbar(x)^2 * (1/2) dx
#    ที่นี่คำนวณ double integral เหนือ (x1, x2) ด้วย scipy (dblquad)
# ---------------------------------------------------------------
def analytical(f, fit):
    def params(x1, x2):
        """fit จากข้อมูล 2 จุด (x1, x2) แล้วคืน (a, b)"""
        x = np.array([x1, x2])
        return fit(x, f(x))

    # ค่าคาดหวังของ a, b, a^2, b^2, ab เหนือ (x1,x2) ~ U[-1,1]^2
    # ความหนาแน่นร่วม = 1/4 บนสี่เหลี่ยม [-1,1]x[-1,1]
    def E(func):
        val, _ = integrate.dblquad(
            lambda x2, x1: func(*params(x1, x2)) * 0.25,
            -1, 1, -1, 1)
        return val

    Ea  = E(lambda a, b: a)
    Eb  = E(lambda a, b: b)
    Ea2 = E(lambda a, b: a * a)
    Eb2 = E(lambda a, b: b * b)
    Eab = E(lambda a, b: a * b)

    # gbar(x) = Ea*x + Eb
    # bias = ∫ (Ea*x + Eb - f(x))^2 /2 dx
    bias, _ = integrate.quad(
        lambda x: (Ea * x + Eb - f(x)) ** 2 * 0.5, -1, 1)

    # E_D[g(x)^2] = Ea2*x^2 + 2*Eab*x + Eb2 ,  gbar(x)^2 = (Ea*x+Eb)^2
    var, _ = integrate.quad(
        lambda x: (Ea2 * x**2 + 2 * Eab * x + Eb2
                   - (Ea * x + Eb) ** 2) * 0.5, -1, 1)
    return bias, var

# ---------------------------------------------------------------
# 4) รันทุกคู่ (ฟังก์ชันเป้าหมาย x แบบจำลอง) + วาดรูป
# ---------------------------------------------------------------
if __name__ == "__main__":
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    print(f"{'target':<10} {'model':<22} | {'bias(ana)':>9} {'bias(sim)':>9} | "
          f"{'var(ana)':>9} {'var(sim)':>9} | {'Eout=b+v':>9}")
    print("-" * 90)

    for i, (fname, f) in enumerate(TARGETS.items()):
        for j, (mname, fit) in enumerate(MODELS.items()):
            b_ana, v_ana = analytical(f, fit)
            b_sim, v_sim, xs, gbar, G = simulate(f, fit)

            print(f"{fname:<10} {mname:<22} | {b_ana:9.4f} {b_sim:9.4f} | "
                  f"{v_ana:9.4f} {v_sim:9.4f} | {b_ana+v_ana:9.4f}")

            # ---- วาดรูป: f(x), ตัวอย่าง g_D, gbar และแถบ +-sd ----
            ax = axes[i, j]
            for g in G[:60]:                       # เส้นตัวอย่าง 60 เส้นจาง ๆ
                ax.plot(xs, g, color="gray", alpha=0.15, lw=0.8)
            sd = G.std(axis=0)
            ax.fill_between(xs, gbar - sd, gbar + sd,
                            color="tab:blue", alpha=0.3, label="gbar ± sd")
            ax.plot(xs, f(xs), "g", lw=2, label="f(x)")
            ax.plot(xs, gbar, "r--", lw=2, label="gbar(x)")
            ax.set_ylim(-2, 2)
            ax.set_title(f"f={fname}, {mname}\n"
                         f"bias={b_sim:.3f}, var={v_sim:.3f}")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("bias_variance.png", dpi=120)
    print("\nบันทึกรูป bias_variance.png แล้ว")
