#!/bin/bash
unset SLURM
export NOSLURM=1
# LOIHI_GEN must be either N3B3 for KPs delivered in 2022
export LOIHI_GEN=N3B3
# NXSDKHOST must be the IP address assigned to the host
export NXSDKHOST=192.168.1.18
# HOST_BINARY must be the path to the nx_driver_server file on the host
# Check that this file is accessible using SSH
export HOST_BINARY=/opt/nxcore/bin/nx_driver_server