function savedPaths = saveBenchmarkResults(resultDir, particleCount, apeTable, runtimeTable)
% SAVEBENCHMARKRESULTS Save per-particle-count benchmark outputs.
% Saves APE CSV (with embedded runtime row).

    particleCountTag = sprintf('N%d', round(particleCount));
    resultBaseName = sprintf('benchmark_%s', particleCountTag);

    apeWithRuntime = localInsertRuntimeUnderVariance100(apeTable, runtimeTable);

    apeCsvPath = fullfile(resultDir, [resultBaseName '_APE.csv']);

    writetable(apeWithRuntime, apeCsvPath);

    savedPaths = struct();
    savedPaths.apeCsvPath = apeCsvPath;
    savedPaths.apeTableWithRuntime = apeWithRuntime;
end

function outTable = localInsertRuntimeUnderVariance100(apeTable, runtimeTable)
    if ~ismember('RowType', apeTable.Properties.VariableNames)
        apeTable = addvars(apeTable, repmat("APE", height(apeTable), 1), ...
            'Before', 1, 'NewVariableNames', 'RowType');
    else
        apeTable.RowType(:) = "APE";
    end

    runtimeRow = array2table(nan(1, width(apeTable)), 'VariableNames', apeTable.Properties.VariableNames);
    runtimeRow.RowType = "RuntimeSec";
    runtimeRow.NoiseVariance = 100;

    variableNames = apeTable.Properties.VariableNames;
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

    insertAfter = find(apeTable.NoiseVariance == 100, 1, 'first');
    if isempty(insertAfter)
        outTable = [apeTable; runtimeRow];
        return;
    end

    if insertAfter == height(apeTable)
        outTable = [apeTable; runtimeRow];
    else
        outTable = [apeTable(1:insertAfter, :); runtimeRow; apeTable(insertAfter+1:end, :)];
    end
end
