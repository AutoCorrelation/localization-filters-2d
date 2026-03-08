function value = rmsev(estimatedPos, startPoint)
    if nargin < 2
        startPoint = 1;
    end
    
    numPoints = size(estimatedPos, 2);
    numIterations = size(estimatedPos, 3);
    numPointsCalc = numPoints - startPoint + 1;

    % Build ground truth: 2 x numPointsCalc (each point p -> [p; p])
    truePos = reshape([startPoint:numPoints; startPoint:numPoints], 2, numPointsCalc, 1);

    % Compute errors for every (point, iteration) pair at once
    errors = estimatedPos(:, startPoint:numPoints, :) - truePos;  % 2 x numPointsCalc x numIterations

    % Sum of Euclidean norms over all points and iterations
    value = sum(sqrt(sum(errors.^2, 1)), 'all') / (numPointsCalc * numIterations);
end
