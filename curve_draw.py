# coding=utf-8
import pylab
import numpy as np
import sys, os
from scipy.optimize import curve_fit
import argparse

def cls_func(x, a, b,c,d):
  q = 1/x+c
  return -b/q**a+d

def reg_func(x, a, b,c):
  q = x
  return b/q**a+c

command = 'sat_3layer'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drawing scaling curves')
    parser.add_argument('--filename', type=str, default="",
                        help='filename of output result (default: )')
    args = parser.parse_args()
    print(args)

    file_path = 'results/'+args.filename
    array = np.load(file_path)
    x = array[0]
    y = array[1]

    if "regression" in args.filename:
       func = reg_func
    if "classification" in args.filename:
       func = cls_func
    print(np.min(x))
    print(np.max(x))
    step= (np.max(x)-np.min(x))/1000

    command = args.filename.split('.')[0]


    popt, pcov = curve_fit(func, x, y,maxfev=50000000,bounds=(0,np.inf))               # 曲线拟合，popt为函数的参数list
    
    if "classification" in args.filename:
        y_pred = [func(i, popt[0], popt[1], popt[2], popt[3]) for i in range(int(np.min(x)-2*step),int(np.max(x)+2*step),int(step))]    # 直接用函数和函数参数list来进行y值的计算
        x_pred = np.arange(int(np.min(x)-2*step),int(np.max(x)+2*step),int(step))
        print(popt)
        y_data = [func(i, popt[0], popt[1], popt[2], popt[3]) for i in x]
    if "regression" in args.filename:
        y_pred = [func(i, popt[0], popt[1], popt[2]) for i in range(int(np.min(x)-2*step),int(np.max(x)+2*step),int(step))]    # 直接用函数和函数参数list来进行y值的计算
        x_pred = np.arange(int(np.min(x)-2*step),int(np.max(x)+2*step),int(step))
        print(popt)
        y_data = [func(i, popt[0], popt[1], popt[2]) for i in x]


    residuals = y-y_data
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('R^2',r_squared)
    plot1 = pylab.plot(x, y, '*', label='original values')
    plot2 = pylab.plot(x_pred, y_pred, 'r', label='fit values')
    pylab.title(command+'    '+r' R^2: '+str(r_squared))

    if "model" in args.filename:
       pylab.xlabel('model size')
    if "data" in args.filename:
       pylab.xlabel("number of nodes")
    
    metric = command.split('_')[-1]
    pylab.ylabel(metric)
    pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
    pylab.xscale('log')
    pylab.show()
    pylab.savefig('figures/'+command+'.png', dpi=200, bbox_inches='tight')
