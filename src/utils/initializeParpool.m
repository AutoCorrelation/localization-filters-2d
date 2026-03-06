function initializeParpool(numWorkers)
    %INITIALIZEPARPOOL Initialize parallel pool for computation
    %   Initialize parpool with specified number of workers if not already active
    
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        parpool(numWorkers);
    end
end
