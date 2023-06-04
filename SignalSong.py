import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import math

t = np.linspace(0,3, 12 * 1024)


Freq_array = np.array([261.63,261.63,293.66,261.63,329.63])
freq_array = np.array([2 *261.63,2 *261.63,2 *293.66, 2* 261.63, 2 * 329.63])
time_array = np.array([0,0.4,0.8,1.2,1.6,2])
Time_array = np.array([0.3,0.3,0.3,0.3,0.3])


N = 5
i = 0
p = 0

while(i < N):
    Fi = Freq_array[i]
    fi = freq_array[i]
    ti = time_array[i]
    Ti = Time_array[i]
    Funci = np.sin(2 * np.pi * Fi * t)
    funci = np.sin(2 * np.pi * fi *  t)
    p = p + (Funci + funci)*((t >= ti) & (t <= (Ti + ti))) 
    i += 1
    
plt.plot(t,p)
sd.play(p , 3*1024)

N = 3*1024
f = np.linspace(0,512,int(N/2))

x_f = fft(p)
x_f = 2/N * np.abs(x_f[0:np.int(N/2)])

fn1,fn2 = np.random.randint(0,512,2)
n = np.sin(2*  fn1 * np.pi *t) + np.sin(2* fn2* np.pi*t)

xn = p + n
xn_f = fft(xn)
xn_f = 2/N * np.abs(xn_f[0:np.int(N/2)])

z = np.where(xn_f > math.ceil(np.max(p)))


index1 = z[0][0]
index2 = z[0][1]
found1 = int(f[index1])
found2 = int(f[index2])

pFilter = xn - (np.sin(2* found1 * np.pi* t) + np.sin(2 * found2* np.pi * t))
sd.play(pFilter, 3 * 1024)

pFilter_f = fft(pFilter)
pFilter_f = 2/N * np.abs(pFilter_f[0:np.int(N/2)])


plt.figure()
plt.subplot(3,1,1)
plt.plot(t,p)
plt.subplot(3,1,2)
plt.plot(t,xn)
plt.subplot(3,1,3)
plt.plot(t,pFilter)
plt.figure()
plt.subplot(3,1,1)
plt.plot(f,x_f)
plt.subplot(3,1,2)
plt.plot(f,xn_f)
plt.subplot(3,1,3)
plt.plot(f,pFilter_f)






