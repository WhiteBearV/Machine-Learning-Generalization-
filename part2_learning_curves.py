# -*- coding: utf-8 -*-
"""
ส่วนที่ 3: Learning Curves เปรียบเทียบ 3 แบบจำลอง
    - แกน x: จำนวนตัวอย่างฝึก N
    - แกน y: ค่าเฉลี่ยของ E_in (error บนชุดฝึก) และ E_out (error จริง)
    - ทดลอง 2 เงื่อนไข: ไม่มี noise และมี noise ~ N(0, sigma^2)

E_out ประมาณด้วย MSE บน grid ละเอียดของ x (คือค่าคาดหวังเหนือ x จริง ๆ)
หมายเหตุ 1: เมื่อมี noise ค่า E_out ที่ดีที่สุดที่เป็นไปได้คือ sigma^2 (noise floor)
หมายเหตุ 2: linear + noise ที่ N=2 ค่าเฉลี่ย E_out จะระเบิดได้
             เพราะบางรอบสุ่มได้ x1 ใกล้ x2 มาก -> ความชัน (y2-y1)/(x2-x1)
             พุ่งสูงมาก นี่คือ "variance สูง" ของแบบจำลองซับซ้อนเมื่อข้อมูลน้อย
             (ในกราฟค่าพวกนี้จะหลุดกรอบ ylim ออกไปเอง)
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ---------- แบบจำลองทั้ง 3 (fit ด้วย lstsq = normal equation) ----------
def fit_const(x, y):
    w = np.linalg.lstsq(np.ones((len(x), 1)), y, rcond=None)[0]
    return np.array([0.0, w[0]])                    # (a, b)

def fit_linear(x, y):
    X = np.column_stack([x, np.ones(len(x))])
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.array([w[0], w[1]])

def fit_linear_origin(x, y):
    w = np.linalg.lstsq(x.reshape(-1, 1), y, rcond=None)[0]
    return np.array([w[0], 0.0])

MODELS = {"constant":   fit_const,
          "linear":     fit_linear,
          "lin-origin": fit_linear_origin}

f = lambda x: np.sin(np.pi * x)        # ฟังก์ชันเป้าหมาย (เปลี่ยนเป็น x**2 ได้)

# ---------- ทดลอง 1 เงื่อนไข noise แล้วคืน learning curve ----------
def learning_curve(fit, Ns, sigma=0.0, trials=3000):
    """คืนค่า (Ein เฉลี่ย, Eout เฉลี่ย) สำหรับแต่ละ N ใน Ns"""
    xs = np.linspace(-1, 1, 400)                  # grid สำหรับวัด E_out
    Ein, Eout = [], []
    for N in Ns:
        ein = eout = 0.0
        for _ in range(trials):
            x = rng.uniform(-1, 1, N)             # สุ่มชุดฝึก N จุด
            y = f(x) + sigma * rng.standard_normal(N)   # ใส่ noise
            a, b = fit(x, y)
            ein  += np.mean((a * x + b - y) ** 2)         # error บนชุดฝึก
            eout += np.mean((a * xs + b - f(xs)) ** 2)    # error เทียบ f จริง
        Ein.append(ein / trials)
        Eout.append(eout / trials)
    return np.array(Ein), np.array(Eout)

# ---------- รันและวาดรูป ----------
if __name__ == "__main__":
    Ns = np.arange(2, 61, 2)
    sigmas = [0.0, 0.3]                            # ไม่มี noise / มี noise
    colors = {"constant": "tab:blue",
              "linear": "tab:red",
              "lin-origin": "tab:green"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, sigma in zip(axes, sigmas):
        for name, fit in MODELS.items():
            Ein, Eout = learning_curve(fit, Ns, sigma)
            ax.plot(Ns, Eout, "-",  color=colors[name], label=f"{name} Eout")
            ax.plot(Ns, Ein,  "--", color=colors[name], alpha=0.6,
                    label=f"{name} Ein")
            print(f"sigma={sigma}  {name:<10}  "
                  f"Eout(N=2)={Eout[0]:.3f}  Eout(N=60)={Eout[-1]:.3f}")
        if sigma > 0:                              # เส้น noise floor
            ax.axhline(sigma**2, color="k", ls=":", lw=1,
                       label=f"noise floor σ²={sigma**2:.2f}")
        ax.set_title(f"f(x)=sin(pi*x),  noise sigma={sigma}")
        ax.set_xlabel("N (training set size)")
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean Squared Error")

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=120)
    print("\nบันทึกรูป learning_curves.png แล้ว")
