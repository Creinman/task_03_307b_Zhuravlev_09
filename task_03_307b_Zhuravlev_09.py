#библиотеки
import math as mt
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

#константы и переменные
W0 = 120.0 * np.pi
Sc = 1.0
C = 300000000
maxSize_m = 5.2
maxTime = 500
dx = 7e-3
maxSize = mt.floor(maxSize_m / dx + 0.5)
dt = Sc * dx / C
tlist = np.arange(0, maxTime * dt, dt)
df = 1.0 / (maxTime * dt)
freq = np.arange(-maxTime / 2 * df, maxTime / 2 * df, df)

#настройка параметров сигнала, датчика, полей
A0 = 100
Amax = 150
Fmax = 3e9
wg = np.sqrt(np.log(Amax)) / (np.pi * Fmax)
dg = wg * np.sqrt(np.log(A0))
sourcePosM = 0.5
sourcePos = mt.floor(sourcePosM / dx + 0.5)
probePosM = 2.6
probePos = mt.floor(probePosM / dx + 0.5)
probeEz = np.zeros(maxTime)
probeHy = np.zeros(maxTime)
Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize)

#гаусов импульс в вакууме
def gaus(q, m, d_gaus, w_gaus, d_t):
    return np.exp(-((((q - m) - (d_gaus / d_t))/(w_gaus / dt)) ** 2))

#настройка графических окон
xlist = np.arange(0, maxSize_m, dx)
plt.ion()
fig, ax0 = plt.subplots()
ax0.set_xlim(0, maxSize_m)
ax0.set_ylim(-1.1, 1.1)
ax0.set_xlabel('x, м')
ax0.set_ylabel('Ez, В/м')
ax0.grid()
ax0.plot(sourcePosM, 0, 'ok')
ax0.plot(probePosM, 0, 'xr')
ax0.legend(['Источник ({:.2f} м)'.format(sourcePosM),
           'Датчик ({:.2f} м)'.format(probePosM)],
          loc='lower right')
line, = ax0.plot(xlist, Ez)
for t in range(1, maxTime):
    Hy[-1] = Hy[-2]
    Hy[:-1] = Hy[:-1] + (Ez[1:] - Ez[:-1]) * Sc / W0
    Hy[sourcePos - 1] -= (Sc / W0) * gaus(t, sourcePos, dg, wg, dt)
    Ez[0] = Ez[1]
    Ez[1:] = Ez[1:] + (Hy[1:] - Hy[:-1]) * Sc * W0
    Ez[sourcePos] += Sc * gaus(t + 1, sourcePos, dg, wg, dt)  
    probeHy[t] = Hy[probePos]
    probeEz[t] = Ez[probePos] 
    if t % 4 == 0:
       plt.title(format(t * dt * 1e9, '.3f') + ' нc')
       line.set_ydata(Ez)
       fig.canvas.draw()
       fig.canvas.flush_events()
plt.ioff()
EzSpec = fftshift(np.abs(fft(probeEz)))
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xlim(0, maxTime * dt)
ax1.set_ylim(0, 1.1)
ax1.set_xlabel('t, с')
ax1.set_ylabel('Ez, В/м')
ax1.plot(tlist, probeEz)
ax1.minorticks_on()
ax1.grid()
ax2.set_xlim(0, maxTime * df / 8)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('f, Гц')
ax2.set_ylabel('|S| / |Smax|, б/р')
ax2.plot(freq, EzSpec / np.max(EzSpec))
ax2.minorticks_on()
ax2.grid()
plt.show()
