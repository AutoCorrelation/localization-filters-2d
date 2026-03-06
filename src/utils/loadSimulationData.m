function [ranging, x_hat_LLS, z_LLS, R_LLS, Q, P0, processNoise, toaNoise, processbias] = loadSimulationData(h5File)
    %LOADSIMULATIONDATA Load simulation data from HDF5 file
    %   Loads all simulation data from H5 file and returns as individual variables
    %   Usage: [ranging, x_hat_LLS, z_LLS, R_LLS, Q, P0, processNoise, toaNoise, processbias] = loadSimulationData(h5File)
    
    ranging = h5read(h5File, '/ranging');
    x_hat_LLS = h5read(h5File, '/x_hat_LLS');
    z_LLS = h5read(h5File, '/z_LLS');
    R_LLS = h5read(h5File, '/R_LLS');
    Q = h5read(h5File, '/Q');
    P0 = h5read(h5File, '/P0');
    processNoise = h5read(h5File, '/processNoise');
    toaNoise = h5read(h5File, '/toaNoise');
    processbias = h5read(h5File, '/processbias');
end
