function data = loadSimulationData(h5File)
    %LOADSIMULATIONDATA Load simulation data from HDF5 file
    %   Loads all simulation data from H5 file and returns as individual variables
    %   Usage: data = loadSimulationData(h5File)

    data.ranging = h5read(h5File, '/ranging');
    data.x_hat_LLS = h5read(h5File, '/x_hat_LLS');
    data.z_LLS = h5read(h5File, '/z_LLS');
    data.R_LLS = h5read(h5File, '/R_LLS');
    data.Q = h5read(h5File, '/Q');
    data.P0 = h5read(h5File, '/P0');
    data.processNoise = h5read(h5File, '/processNoise');
    data.toaNoise = h5read(h5File, '/toaNoise');
    data.processbias = h5read(h5File, '/processbias');
    data.true_state = h5read(h5File, '/true_state');
    data.mode_history = h5read(h5File, '/mode_history');

    data.true_position = [];
    if localHasDataset(h5File, 'true_position')
        data.true_position = h5read(h5File, '/true_position');
    end
end

function tf = localHasDataset(h5File, datasetName)
    info = h5info(h5File);
    tf = any(strcmp({info.Datasets.Name}, datasetName));
end
