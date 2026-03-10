function value = rmsev(estimatedPos, startPoint)
    if nargin < 2
        startPoint = 1;
    end
    
    rmseSum = 0;
    numPoints = size(estimatedPos, 2);
    numIterations = size(estimatedPos, 3);

    for countIter = 1:numIterations
        for countPoint = startPoint:numPoints
            rmseSum = rmseSum + norm(estimatedPos(:, countPoint, countIter) - [countPoint; countPoint]);
        end
    end

    numPointsCalc = numPoints - startPoint + 1;
    value = rmseSum / (numPointsCalc * numIterations);
end
