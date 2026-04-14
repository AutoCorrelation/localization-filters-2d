function [RMSE, APE] = evaluateFilter(estimatedPos, startPoint, truePos)
    % EVALUATEFILTER Compute RMSE/APE for filter performance
    % Ground truth: diagonal constant velocity path (1,1) to (10,10)
    %   
    % Inputs:
    %   estimatedPos - (2, numPoints, numIterations) estimated positions
    %   startPoint   - Starting point index for evaluation (default: 3)
    %   truePos      - Optional ground truth positions:
    %                  (2, numPoints) or (2, numPoints, numIterations)
    %
    % Outputs:
    %   RMSE    - Root Mean Square Error (primary metric)
    %   APE     - Average Position Error
    
    if nargin < 2 || isempty(startPoint)
        startPoint = 3;
    end
    if nargin < 3
        truePos = [];
    end
    
    numPoints = size(estimatedPos, 2);
    numIterations = size(estimatedPos, 3);
    
    % Collect per-sample Euclidean errors.
    hasTruePos = ~isempty(truePos);
    if hasTruePos
        if ndims(truePos) == 2
            truePos = repmat(truePos, 1, 1, numIterations);
        end
        if size(truePos, 1) < 2 || size(truePos, 2) < numPoints || size(truePos, 3) < numIterations
            error('evaluateFilter:InvalidTruePos', ...
                'truePos must be size (2, numPoints) or (2, numPoints, numIterations).');
        end
    end

    numEvaluations = (numPoints - startPoint + 1) * numIterations;
    errors = zeros(numEvaluations, 1);    % Euclidean (for APE)
    errorsSq = zeros(numEvaluations, 1);  % squared Euclidean (for RMSE)
    writeIdx = 1;
    
    for iterIdx = 1:numIterations
        for pointIdx = startPoint:numPoints
            if hasTruePos
                gt = truePos(1:2, pointIdx, iterIdx);
            else
                gt = [pointIdx; pointIdx];
            end
            d = estimatedPos(:, pointIdx, iterIdx) - gt;
            errSq = d(1)^2 + d(2)^2;        % squared 2D error (dx^2 + dy^2)
            errorsSq(writeIdx) = errSq;
            errors(writeIdx) = sqrt(errSq); % Euclidean distance (for APE)
            writeIdx = writeIdx + 1;
        end
    end

    RMSE = sqrt(mean(errorsSq));
    APE = mean(errors);
end
