from __future__ import division 
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt


def dmp_to_dat(dmp, dat, tot_lns, lns_per_time, bad_lns, cols, tsteps_on=False, first_tstep=None, last_tstep=None, 
               tstep_size=None):
    # Pull in the .dmp file
    toskip = []
    for idx in range(tot_lns - 1):
        if (idx % lns_per_time) in [i for i in range(bad_lns)]:
            toskip.append(idx)
    data=pd.read_csv(dmp, sep = " ", skiprows=toskip, usecols=cols.keys(), names=cols.values(), engine='python')

    if tsteps_on:
        # Add timesteps to df
        timesteps = np.linspace(first_tstep, last_tstep, int((last_tstep - first_tstep) / tstep_size) + 1) # lazy?
        timesteps_list = []
        for timestep in timesteps:
            timesteps_list.append(round(timestep))
        data["timestep"] = timesteps_list
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]

    # Return df and write to output file
    data.to_csv(dat, sep=' ', index=False)
    return data


def thermo_to_dat(thermo, dat, header_len, last_ln, tot_len, cols, first_tstep, last_tstep, tstep_size):
    # Pull in log.dmp
    data=pd.read_fwf(thermo, skiprows=header_len, skipfooter=tot_len-last_ln-1, usecols=cols.keys(), names=cols.values(), engine='python', dtype=float)

    # Add timesteps to df, hacky fix should just do above but whatever
    timesteps = np.linspace(first_tstep, last_tstep, int((last_tstep - first_tstep) / tstep_size) + 1)
    timesteps_list = []
    for timestep in timesteps:
        timesteps_list.append(round(timestep))
    data["timestep"] = timesteps_list
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]

    # Return df and write to output file
    data.to_csv(dat, sep=' ', index=False)
    return data


def plot_block_stats(param, block_width, data, timesteps, data_name, CONST, output, max_box=1001, full_graph=True, avgs_on=True, sigmas_on=False, rs_on=False):
    half_block = int(0.5*(block_width-1))
    avgs = []
    sigmas = []
    rs = []
    step_range = range(len(data))[half_block:-half_block]
    for step in step_range:
        block = data[step-half_block:step+half_block+1]
        avg = np.average(block)
        avgs.append(avg)
        sigmas.append(np.std(block, ddof=1))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            timesteps[step-half_block:step+half_block+1], block)
        rs.append(r_value)
    avgs_raw, rs_raw = avgs, rs
    timesteps_raw = [timesteps[i] for i in step_range]
    
    if full_graph:
        timestep_range = timesteps
        avgs_avg = np.average(avgs)
        sigmas_avg = np.average(sigmas)
        rs_avg = np.average(rs)
        avgs = [avgs_avg]*half_block + avgs + [avgs_avg]*half_block
        sigmas = [sigmas_avg]*half_block + sigmas + [sigmas_avg]*half_block
        rs = [rs_avg]*half_block + rs + [rs_avg]*half_block
    else:        
        timestep_range = [timesteps[i] for i in step_range]
    
    if avgs_on:
        # Avgs
        fig, ax = plt.subplots(figsize=[15, 10])
        ax.plot(timestep_range, avgs, label="Average %s of %d timesteps calculated with this center" % (data_name, block_width), 
                color="black", linestyle="solid")
        ax.set_xlabel("Timesteps", fontsize=20)
        plt.legend(loc="upper left")
        plt.savefig(output + "figures/%s%d/%sblock_avg.png" % (param, CONST, data_name), bbox_inches="tight")
        plt.show()
#     # Relative avgs
#     fig, ax = plt.subplots(figsize=[15, 10])
#     rel_avgs = [a/]
#     ax.plot(timestep_range, avgs, label="Relative average %s of %d timesteps calculated with this center" % (data_name, 
#                                                                                                              block_width),
#                                                                                            color="black", linestyle="solid")
#     ax.set_xlabel("Timesteps", fontsize=20)
#     plt.legend(loc="upper left")
#     plt.savefig("figures/%s%d/%s_ravg_block.png" % (CONST, data_name), bbox_inches="tight")
#     plt.show()

    if sigmas_on:
        #Sigmas
        fig, ax = plt.subplots(figsize=[15, 10])
        ax.plot(timestep_range, sigmas, label="Sigma of %s of %d timesteps calculated with this center" % (data_name,
                                                                          block_width), color="blue", linestyle="solid")
        plt.legend(loc="upper left")
        plt.savefig(output + "figures/%s%d/sigmas_%s_block.png" % (param, CONST, data_name), bbox_inches="tight")
        plt.show()
    
    if rs_on:
        # r and r^2 values
        fig, ax = plt.subplots(figsize=[15, 10])
        ax.plot(timestep_range, rs, label="r of %s of %d timesteps calculated with this center" % (data_name, block_width), 
                color="green", linestyle="solid")
        r2s = [r**2 for r in rs]
        ax.plot(timestep_range, r2s, label="r^2 of %s of %d timesteps calculated with this center" % (data_name, block_width), 
                color="purple", linestyle="solid")
        ax.plot(timestep_range, [0]*len(timestep_range), label="0", color = "black", linestyle="solid")
        ax.set_ylim(-1,1)
        plt.legend(loc="upper left")
        plt.savefig(output + "figures/%s%d/%sblock_r.png" % (param, CONST, data_name), bbox_inches="tight")
        plt.show()
    
    return avgs_raw, rs_raw, timesteps_raw
    

# Intercept code, copy and pasted
def find_intercepts(x, y1, y2):
    def interpolated_intercepts(x, y1, y2):
        def intercept(point1, point2, point3, point4):
            def line(p1, p2):
                A = (p1[1] - p2[1])
                B = (p2[0] - p1[0])
                C = (p1[0]*p2[1] - p2[0]*p1[1])
                return A, B, -C
            def intersection(L1, L2):
                D  = L1[0] * L2[1] - L1[1] * L2[0]
                Dx = L1[2] * L2[1] - L1[1] * L2[2]
                Dy = L1[0] * L2[2] - L1[2] * L2[0]
                x = Dx / D
                y = Dy / D
                return x,y
            L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
            L2 = line([point3[0],point3[1]], [point4[0],point4[1]])
            R = intersection(L1, L2)
            return R
        idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
        xcs = []
        ycs = []
        for idx in idxs:
            xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], 
                                                                                                 y2[idx+1])))
            xcs.append(xc)
            ycs.append(yc)
        return np.array(xcs), np.array(ycs)
    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
    xcs, ycs = interpolated_intercepts(x,y1,y2)
    return xcs, ycs


def num_derivative(x_list, y_list):
    y_prime = np.diff(y_list)/np.diff(x_list)
    x_prime = []
    for i in range(len(y_prime)):
        temp = (x_list[i+1] + x_list[i])/2
        x_prime = np.append(x_prime, temp)
    return x_prime, y_prime