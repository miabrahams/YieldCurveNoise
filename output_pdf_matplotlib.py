import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stat
import matplotlib.backends.backend_pdf as bq

pp = bq.PdfPages('furnace_Python_out.pdf')

furnace0 = pd.read_csv('furnace.csv', header=0, na_values=['.'])
furnace = furnace0[["CHArea","CHHght","Age","BTUIn","BTUOut","Damper"]]

print(furnace.head(25))
print(" ")

print(furnace[["CHArea","CHHght","Age","BTUIn","BTUOut"]].describe())
print(" ")
print("Missing CHArea %-tiles in pandas release 0.18.1:")
print(np.nanpercentile(furnace['CHArea'],[25,50,75]))   
print(" ") 
# pandas release 0.18.1 "describe" function doesn't return percentiles when columns contain NaN 

print(furnace['Damper'].value_counts())
print(" ")

plt.ioff()               # suppresses empty plot in next line
fig, ax = plt.subplots()
evd = ax.scatter(furnace.BTUIn[furnace.Damper==1], furnace.BTUOut[furnace.Damper==1], facecolor='blue', edgecolor='black', linewidth=0.25)
tvd = ax.scatter(furnace.BTUIn[furnace.Damper==2], furnace.BTUOut[furnace.Damper==2], facecolor='red',  edgecolor='black', linewidth=0.25)
ax.set_aspect(1./ax.get_data_ratio())
ax.set_xticks(np.linspace(0, 20, 11, endpoint=True))
ax.set_yticks(np.linspace(2, 22, 11, endpoint=True))
ax.legend(([evd, tvd]), ("EVD","TVD"), loc='lower right')
ax.set_title('BTUIn vs BTUOut for Stratified Furnace Data')
ax.set_xlabel('BTUIn')
ax.set_ylabel('BTUOut')
pp.savefig(fig, dpi=1200)

M = np.mean(furnace.BTUIn)
S = np.std(furnace.BTUIn,ddof=1)

plt.ioff()               
fig, ax = plt.subplots()
bins = np.arange(2,21,2)
bars = ax.hist(furnace.BTUIn, normed=1, bins=np.arange(2,21,2), facecolor='white', edgecolor='black', linewidth=0.10)
x = np.arange(2,20,0.01)
y = mlab.normpdf(x, M, S)
curv = ax.plot(x, y, 'm--')
ax.set_title('BTUIn Histogram & Normal Fit')
ax.set_xlabel("BTUIn")
ax.set_ylabel("Probability")
pp.savefig(fig, dpi=1200)

N = len(furnace.BTUIn)
A = M+S*stat.t.ppf(0.05,N-1)/np.sqrt(N)
B = M+S*stat.t.ppf(0.95,N-1)/np.sqrt(N)
print("90 pct CI about BTUIn mean: "+str(round(A,5))+" "+str(round(M,5))+" "+str(round(B,5)))
print(" ")

A = S*np.sqrt((N-1)/stat.chi2.ppf(0.95,N-1))
B = S*np.sqrt((N-1)/stat.chi2.ppf(0.05,N-1))
print("90 pct CI about BTUIn dtdv: "+str(round(A,5))+" "+str(round(S,5))+" "+str(round(B,5)))
print(" ")

slope, intercept, r_value, p_value, std_err = stat.linregress(furnace.BTUIn,furnace.BTUOut)
print("Slope: "+str(round(slope,5)))
print("Intercept: "+str(round(intercept,5)))
print("Multiple R-squared: "+str(round(r_value**2,4)))

plt.ioff()               
fig, ax = plt.subplots()
lnr = ax.plot([0,20],[intercept,intercept+slope*20], color='red')
dat = ax.scatter(furnace.BTUIn, furnace.BTUOut, facecolor='green', marker=".", edgecolor='black', linewidth=0.15)
ax.set_aspect(1./ax.get_data_ratio())
ax.set_xticks(np.linspace(0, 20, 11, endpoint=True))
ax.set_yticks(np.linspace(2, 22, 11, endpoint=True))
ax.set_title('BTUIn vs BTUOut for Stratified Furnace Data')
ax.set_xlabel('BTUIn')
ax.set_ylabel('BTUOut')
pp.savefig(fig, dpi=1200)

pp.close()