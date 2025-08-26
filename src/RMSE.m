classdef RMSE
    properties
        
    end

    methods
        function obj = RMSE()
        end

        function y = getRMSE(~, estimatedPos)
            y = zeros(size(estimatedPos, 4),1);
            for i = 1:size(estimatedPos, 3)
                for p = 1:size(estimatedPos, 2)
                    for n = 1:size(estimatedPos, 4)
                        y(n) = y(n) + norm(estimatedPos(:, p, i, n) - [p; p]);
                    end
                end
            end
            y = y / (size(estimatedPos, 2) * size(estimatedPos, 3));
        end
    end


end