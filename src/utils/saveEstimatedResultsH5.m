function savedPaths = saveEstimatedResultsH5(resultDir, particleCount, filterNames, estimatedPosByFilter, noiseVariance, motionModel)
%SAVEESTIMATEDRESULTSH5 Save per-noise estimate tensors to an HDF5 file.

    if nargin < 6 || isempty(motionModel)
        motionModel = 'cv';
    end

    particleCountTag = sprintf('N%d', round(particleCount));
    motionPrefix = sprintf('%s_', motionModel);
    h5Path = fullfile(resultDir, sprintf('benchmark_%s%s_estimates.h5', motionPrefix, particleCountTag));

    if exist(h5Path, 'file')
        delete(h5Path);
    end

    h5create(h5Path, '/particleCount', [1 1], 'Datatype', 'double');
    h5write(h5Path, '/particleCount', double(round(particleCount)));

    h5create(h5Path, '/noiseVariance', size(noiseVariance(:)), 'Datatype', 'double');
    h5write(h5Path, '/noiseVariance', noiseVariance(:));

    for fIdx = 1:numel(filterNames)
        filterName = char(filterNames{fIdx});
        estimateTensor = estimatedPosByFilter{fIdx};
        if isempty(estimateTensor)
            continue;
        end

        numNoise = size(estimateTensor, 4);
        for noiseIdx = 1:numNoise
            dsetName = localBuildDatasetName(filterName, noiseVariance(noiseIdx));
            estSlice = estimateTensor(:, :, :, noiseIdx);
            h5create(h5Path, dsetName, size(estSlice), 'Datatype', 'double');
            h5write(h5Path, dsetName, estSlice);
        end
    end

    savedPaths = struct();
    savedPaths.estimatesH5Path = h5Path;
end

function dsetName = localBuildDatasetName(filterName, noiseValue)
    safeFilterName = regexprep(filterName, '[^A-Za-z0-9_]', '_');
    safeNoiseTag = regexprep(sprintf('%.6g', noiseValue), '[^A-Za-z0-9_]', '_');
    dsetName = sprintf('/estimatedPos_%s_var_%s', safeFilterName, safeNoiseTag);
end