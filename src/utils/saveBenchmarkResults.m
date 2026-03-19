function savedPaths = saveBenchmarkResults(resultDir, particleCount, maeTable, runtimeTable)
% SAVEBENCHMARKRESULTS Save per-particle-count benchmark outputs.
% Saves MAE CSV (with embedded runtime row).

    particleCountTag = sprintf('N%d', round(particleCount));
    resultBaseName = sprintf('benchmark_%s', particleCountTag);

    maeWithRuntime = localInsertRuntimeUnderVariance100(maeTable, runtimeTable);

    maeCsvPath = fullfile(resultDir, [resultBaseName '_MAE.csv']);

    writetable(maeWithRuntime, maeCsvPath);

    savedPaths = struct();
    savedPaths.maeCsvPath = maeCsvPath;
    savedPaths.maeTableWithRuntime = maeWithRuntime;
end

function outTable = localInsertRuntimeUnderVariance100(maeTable, runtimeTable)
    if ~ismember('RowType', maeTable.Properties.VariableNames)
        maeTable = addvars(maeTable, repmat("MAE", height(maeTable), 1), ...
            'Before', 1, 'NewVariableNames', 'RowType');
    else
        maeTable.RowType(:) = "MAE";
    end

    runtimeRow = array2table(nan(1, width(maeTable)), 'VariableNames', maeTable.Properties.VariableNames);
    runtimeRow.RowType = "RuntimeSec";
    runtimeRow.NoiseVariance = 100;

    variableNames = maeTable.Properties.VariableNames;
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

    insertAfter = find(maeTable.NoiseVariance == 100, 1, 'first');
    if isempty(insertAfter)
        outTable = [maeTable; runtimeRow];
        return;
    end

    if insertAfter == height(maeTable)
        outTable = [maeTable; runtimeRow];
    else
        outTable = [maeTable(1:insertAfter, :); runtimeRow; maeTable(insertAfter+1:end, :)];
    end
end
