# Gannet Optimization Algorithm

## Initialization

$$
X = 
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,Dim} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,Dim} \\
\vdots  & \vdots  & \ddots & \vdots  \\
x_{N,1} & x_{N,2} & \cdots & x_{N,Dim} 
\end{bmatrix}\\\\
x_{i,j}=r_1\cdot (ub_j-lb_j)+lb_j\\i=1,2,\dots,N\ j=1,2,\dots,Dim
$$

## Exploration

$$
V(x) =
\begin{cases}
	-\displaystyle\frac{1}{\pi}\cdot x + 1,\ x\in(0, \pi)\\\\
	\displaystyle\frac{1}{\pi}\cdot x - 1,\ x\in(\pi, 2\pi)
\end{cases}
\\\\
t = 1 - \displaystyle\frac{Iter}{T_{max\_iter}}\\\\
a = 2\cdot \cos(2\pi r_2)\cdot t\\\\
b = 2\cdot V(2\pi r_3)\cdot t\\\\
A = (2r_4 - 1)\cdot a\\\\
B = (2r_5 - 1)\cdot b\\\\
X_m(t) = \displaystyle\frac{1}{N}\sum_{i=1}^N X_i(t)\\\\
u_2 = A\cdot (X_i(t) - X_r(t))\\\\
v_2 = B\cdot(X_i(t) - X_m(t))\\\\
MX_i(t + 1) =
\begin{cases}
	X_i(t) + u_1 + u_2,\ q \ge 0.5\\\\
	X_i(t) + v_1 + v_2,\ q < 0.5
\end{cases}
$$

## Exploitation

$$
L = 0.2 + (2 - 0.2)\cdot r_6\\\\
R = \displaystyle\frac{M\cdot velocity^2}{L}\ (M=2.5kg,\ velocity=1.5m/s)\\\\
t_2 = 1 + \displaystyle\frac{Iter}{T_{max\_iter}}\\\\
Capturability = \displaystyle\frac{1}{R\cdot t_2}\\\\
\sigma = \displaystyle(\frac{\Gamma(1+\beta)\cdot \sin(\frac{\pi\beta}{2})}{\Gamma(\frac{1+\beta}{2}\cdot\beta\cdot 2^{\frac{\beta-1}{2}})})^{\frac{1}{\beta}}\\\\
Levy(Dim) = 0.01\cdot \displaystyle\frac{\mu\cdot \sigma}{|v|^{\frac{1}{\beta}}}\\\\
P = Levy(Dim)\\\\
delta = Captureability\cdot |X_i(t)-X_{best}(t)|\\\\
MX_i(t+1) = 
\begin{cases}
	t\cdot delta\cdot (X_i(t)-X_{best}(t))+X_i(t)\ \text{ , Capturability}\ge c\\\\
	X_{best}-(X_i(t)-X_{best})\cdot P\cdot t\ \text{ , Capturability}< c\ (c = 0.2)
\end{cases}
$$

