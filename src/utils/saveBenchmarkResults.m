function savedPaths = saveBenchmarkResults(resultDir, particleCount, rmseTable, runtimeTable, motionModel)
% SAVEBENCHMARKRESULTS Save per-particle-count benchmark outputs.
% Saves RMSE CSV (with embedded runtime row).
% motionModel: 'cv' or 'imm' for filename prefix

    if nargin < 5 || isempty(motionModel)
        motionModel = 'cv';
    end
    
    particleCountTag = sprintf('N%d', round(particleCount));
    motionPrefix = sprintf('%s_', motionModel);
    resultBaseName = sprintf('benchmark_%s%s', motionPrefix, particleCountTag);

    rmseWithRuntime = localInsertRuntimeUnderVariance100(rmseTable, runtimeTable);

    rmseCsvPath = fullfile(resultDir, [resultBaseName '_RMSE.csv']);

    writetable(rmseWithRuntime, rmseCsvPath);

    savedPaths = struct();
    savedPaths.rmseCsvPath = rmseCsvPath;
    savedPaths.rmseTableWithRuntime = rmseWithRuntime;
end

function outTable = localInsertRuntimeUnderVariance100(rmseTable, runtimeTable)
    if ~ismember('RowType', rmseTable.Properties.VariableNames)
        rmseTable = addvars(rmseTable, repmat("RMSE", height(rmseTable), 1), ...
            'Before', 1, 'NewVariableNames', 'RowType');
    else
        rmseTable.RowType(:) = "RMSE";
    end

    runtimeRow = array2table(nan(1, width(rmseTable)), 'VariableNames', rmseTable.Properties.VariableNames);
    runtimeRow.RowType = "RuntimeSec";
    runtimeRow.NoiseVariance = 100;

    variableNames = rmseTable.Properties.VariableNames;
    for cIdx = 1:numel(variableNames)
        filterName = variableNames{cIdx};
        if strcmp(filterName, 'RowType') || strcmp(filterName, 'NoiseVariance')
            continue;
        end

        rowMask = strcmp(runtimeTable.FilterName, filterName);
        if any(rowMask)
            runtimeRow{1, cIdx} = runtimeTable.RuntimeSec(find(rowMask, 1, 'first'));
        end
    end

    insertAfter = find(rmseTable.NoiseVariance == 100, 1, 'first');
    if isempty(insertAfter)
        outTable = [rmseTable; runtimeRow];
        return;
    end

    if insertAfter == height(rmseTable)
        outTable = [rmseTable; runtimeRow];
    else
        outTable = [rmseTable(1:insertAfter, :); runtimeRow; rmseTable(insertAfter+1:end, :)];
    end
end
