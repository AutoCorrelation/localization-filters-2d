function [RMSE, MAE] = evaluateFilter(estimatedPos, startPoint)
    % EVALUATEFILTER Compute RMSE/MAE for filter performance
    % Ground truth: diagonal constant velocity path (1,1) to (10,10)
    %   
    % Inputs:
    %   estimatedPos - (2, numPoints, numIterations) estimated positions
    %   startPoint   - Starting point index for evaluation (default: 3)
    %
    % Outputs:
    %   RMSE    - Root Mean Square Error
    %   MAE     - Mean Absolute Error (optional 3rd output)
    
    if nargin < 2 || isempty(startPoint)
        startPoint = 3;
    end
    
    numPoints = size(estimatedPos, 2);
    numIterations = size(estimatedPos, 3);
    
    % Collect per-sample Euclidean errors; ground truth is [k; k] for point k
    numEvaluations = (numPoints - startPoint + 1) * numIterations;
    errors = zeros(numEvaluations, 1);
    writeIdx = 1;
    
    for iterIdx = 1:numIterations
        for pointIdx = startPoint:numPoints
            error = norm(estimatedPos(:, pointIdx, iterIdx) - [pointIdx; pointIdx]);
            errors(writeIdx) = error;
            writeIdx = writeIdx + 1;
        end
    end

    MAE = mean(errors);
    RMSE = sqrt(mean(errors .^ 2));
end
