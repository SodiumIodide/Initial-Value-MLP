#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def get_data(filename):
    with open(filename) as datafile:
        file_lines = list(map(lambda x: x.strip().split(","), datafile.readlines()))
        x_values = np.fromiter(map(lambda y: float(y[0]), file_lines), dtype=np.float)
        y_values = np.fromiter(map(lambda y: float(y[1]), file_lines), dtype=np.float)
    return (x_values, y_values)

def main():
    # Transport data
    try:
        total_flux_distance, total_flux = get_data("out/total_flux.out")
        flux_1_distance, flux_1 = get_data("out/flux_1.out")
        flux_2_distance, flux_2 = get_data("out/flux_2.out")
        plt.plot(total_flux_distance, total_flux, label="Total Material")
        plt.plot(flux_1_distance, flux_1, label="Material 1")
        plt.plot(flux_2_distance, flux_2, label="Material 2")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Flux (1/cm^2-s-MeV)")
        plt.legend(loc="best")
        plt.grid(b=True, which="both", axis="both")
        #plt.ylim(bottom=0.0)
        plt.savefig("img/flux_profile.png")
        plt.cla()
        plt.clf()
        psi_1_m_distance, psi_1_m = get_data("out/psi_1_m.out")
        psi_1_p_distance, psi_1_p = get_data("out/psi_1_p.out")
        psi_2_m_distance, psi_2_m = get_data("out/psi_2_m.out")
        psi_2_p_distance, psi_2_p = get_data("out/psi_2_p.out")
        plt.plot(psi_1_p_distance, psi_1_p, label="Psi 1+")
        plt.plot(psi_1_m_distance, psi_1_m, label="Psi 1-")
        plt.plot(psi_2_p_distance, psi_2_p, label="Psi 2+")
        plt.plot(psi_2_m_distance, psi_2_m, label="Psi 2-")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Angular Flux (1/cm^3-s-MeV-strad)")
        plt.legend(loc="best")
        plt.grid(b=True, which="both", axis="both")
        #plt.ylim(bottom=0.0)
        plt.savefig("img/angular_flux_profile.png")
        plt.cla()
        plt.clf()
    except:
        pass
    try:
        # "Initial condition" data
        ic_total_flux_distance, ic_total_flux = get_data("out/ic_total_flux.out")
        ic_flux_1_distance, ic_flux_1 = get_data("out/ic_flux_1.out")
        ic_flux_2_distance, ic_flux_2 = get_data("out/ic_flux_2.out")
        plt.plot(ic_total_flux_distance, ic_total_flux, label="Total Material")
        plt.plot(ic_flux_1_distance, ic_flux_1, label="Material 1")
        plt.plot(ic_flux_2_distance, ic_flux_2, label="Material 2")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Flux (1/cm^2-s-MeV)")
        plt.legend(loc="best")
        plt.grid(b=True, which="both", axis="both")
        #plt.ylim(bottom=0.0)
        plt.xlim(right=10.0)
        plt.savefig("img/ic_flux_profile.png")
        plt.cla()
        plt.clf()
        ic_psi_1_m_distance, ic_psi_1_m = get_data("out/ic_psi_1_m.out")
        ic_psi_1_p_distance, ic_psi_1_p = get_data("out/ic_psi_1_p.out")
        ic_psi_2_m_distance, ic_psi_2_m = get_data("out/ic_psi_2_m.out")
        ic_psi_2_p_distance, ic_psi_2_p = get_data("out/ic_psi_2_p.out")
        plt.plot(ic_psi_1_p_distance, ic_psi_1_p, label="Psi 1+")
        plt.plot(ic_psi_1_m_distance, ic_psi_1_m, label="Psi 1-")
        plt.plot(ic_psi_2_p_distance, ic_psi_2_p, label="Psi 2+")
        plt.plot(ic_psi_2_m_distance, ic_psi_2_m, label="Psi 2-")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Angular Flux (1/cm^3-s-MeV-strad)")
        plt.legend(loc="best")
        plt.grid(b=True, which="both", axis="both")
        #plt.ylim(bottom=0.0)
        plt.savefig("img/ic_angular_profile.png")
        plt.cla()
        plt.clf()
    except:
        pass
    try:
        # LP Transport data
        total_flux_distance_lp, total_flux_lp = get_data("out/total_flux_lp.out")
        flux_1_distance_lp, flux_1_lp = get_data("out/flux_1_lp.out")
        flux_2_distance_lp, flux_2_lp = get_data("out/flux_2_lp.out")
        plt.plot(total_flux_distance_lp, total_flux_lp, label="Total Material")
        plt.plot(flux_1_distance_lp, flux_1_lp, label="Material 1")
        plt.plot(flux_2_distance_lp, flux_2_lp, label="Material 2")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Flux (1/cm^2-s-MeV)")
        plt.legend(loc="best")
        plt.grid(b=True, which="both", axis="both")
        #plt.ylim(bottom=0.0)
        plt.savefig("img/flux_profile_lp.png")
        plt.cla()
        plt.clf()
        psi_1_m_distance_lp, psi_1_m_lp = get_data("out/psi_1_m_lp.out")
        psi_1_p_distance_lp, psi_1_p_lp = get_data("out/psi_1_p_lp.out")
        psi_2_m_distance_lp, psi_2_m_lp = get_data("out/psi_2_m_lp.out")
        psi_2_p_distance_lp, psi_2_p_lp = get_data("out/psi_2_p_lp.out")
        plt.plot(psi_1_p_distance_lp, psi_1_p_lp, label="Psi 1+")
        plt.plot(psi_1_m_distance_lp, psi_1_m_lp, label="Psi 1-")
        plt.plot(psi_2_p_distance_lp, psi_2_p_lp, label="Psi 2+")
        plt.plot(psi_2_m_distance_lp, psi_2_m_lp, label="Psi 2-")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Angular Flux (1/cm^3-s-MeV-strad)")
        plt.legend(loc="best")
        plt.grid(b=True, which="both", axis="both")
        #plt.ylim(bottom=0.0)
        plt.savefig("img/angular_profile_lp.png")
        plt.cla()
        plt.clf()
    except:
        pass

if __name__ == '__main__':
    main()
