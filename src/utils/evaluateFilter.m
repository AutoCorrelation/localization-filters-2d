function [RMSE, metrics] = evaluateFilter(estimatedPos, startPoint)
    % EVALUATEFILTER Compute RMSE for filter performance
    % Ground truth: diagonal constant velocity path (1,1) to (10,10)
    %   
    % Inputs:
    %   estimatedPos - (2, numPoints, numIterations) estimated positions
    %   startPoint   - Starting point index for evaluation (default: 3)
    %
    % Outputs:
    %   RMSE    - Root Mean Square Error
    %   metrics - Struct with detailed error metrics (optional)
    
    if nargin < 2 || isempty(startPoint)
        startPoint = 3;
    end
    
    numPoints = size(estimatedPos, 2);
    numIterations = size(estimatedPos, 3);
    
    % Compute RMSE: ground truth is [k; k] for point k
    errorSum = 0;
    
    for iterIdx = 1:numIterations
        for pointIdx = startPoint:numPoints
            error = norm(estimatedPos(:, pointIdx, iterIdx) - [pointIdx; pointIdx]);
            errorSum = errorSum + error;
        end
    end
    
    numEvaluations = (numPoints - startPoint + 1) * numIterations;
    RMSE = errorSum / numEvaluations;
    
    % Optional detailed metrics
    if nargout > 1
        errorSumSquared = 0;
        maxError = 0;
        errorByPoint = zeros(numPoints, 1);
        
        for iterIdx = 1:numIterations
            for pointIdx = startPoint:numPoints
                error = norm(estimatedPos(:, pointIdx, iterIdx) - [pointIdx; pointIdx]);
                errorSumSquared = errorSumSquared + error^2;
                maxError = max(maxError, error);
                errorByPoint(pointIdx) = errorByPoint(pointIdx) + error;
            end
        end
        
        metrics.RMSE = RMSE;
        metrics.MSE = errorSumSquared / numEvaluations;
        metrics.maxError = maxError;
        metrics.avgErrorByPoint = errorByPoint / numIterations;
    end
end
