classdef Env
    properties
        numIterations
        numPoints
        Anchor
        noiseVariance
    end

    methods
        function obj = Env(numIterations)
            obj.numIterations = numIterations;
            obj.numPoints = 10;
            obj.Anchor = [0 0 10 10; 10 0 0 10];
            obj.noiseVariance = [0.01; 0.1; 1; 10; 100];
        end

        function preSimulate(obj)
            ranging = zeros(4, 1);
            lAnchor = obj.Anchor;
            lnoiseVariance = obj.noiseVariance;
            lnumIterations = obj.numIterations;
            lnumPoints = obj.numPoints;
            
            z = zeros(6, lnumIterations, lnumPoints, 5);
            toaPos = zeros(2, lnumIterations, lnumPoints, 5);
            R = zeros(6, 6, lnumIterations, lnumPoints, 5);
            H = [...
                0, -20
                20, -20
                20, 0
                20, 0
                20, 20
                0, 20];
            pinvH = pinv(H);
            for i = 1:lnumIterations
                for p = 1:lnumPoints
                    realPos = [p; p];
                    for n = 1:5
                        for a = 1:4
                            ranging(a,1) = norm(realPos - lAnchor(:,a)) + sqrt(lnoiseVariance(n)) * randn;
                        end
                        z(:,i,p,n) = [...
                        ranging(1, 1)^2 - ranging(2, 1)^2 - 10^2;
                        ranging(1, 1)^2 - ranging(3, 1)^2;
                        ranging(1, 1)^2 - ranging(4, 1)^2 + 10^2;
                        ranging(2, 1)^2 - ranging(3, 1)^2 + 10^2;
                        ranging(2, 1)^2 - ranging(4, 1)^2 + 2*(10^2);
                        ranging(3, 1)^2 - ranging(4, 1)^2 + 10^2];

                        toaPos(:,i,p,n) = pinvH * z(:,i,p,n);

                        R(:,:,i,p,n) = [4*obj.noiseVariance(n)*(ranging(1,1)^2+ranging(2,1)^2) 4*obj.noiseVariance(n)*(ranging(1,1)^2) 4*obj.noiseVariance(n)*(ranging(1,1)^2) -4*obj.noiseVariance(n)*(ranging(2,1)^2) -4*obj.noiseVariance(n)*(ranging(2,1)^2) 0;...
                                    4*obj.noiseVariance(n)*(ranging(1,1)^2) 4*obj.noiseVariance(n)*(ranging(1,1)^2+ranging(3,1)^2) 4*obj.noiseVariance(n)*(ranging(1,1)^2) 4*obj.noiseVariance(n)*(ranging(3,1)^2) 0 -4*obj.noiseVariance(n)*(ranging(3,1)^2);...
                                    4*obj.noiseVariance(n)*(ranging(1,1)^2) 4*obj.noiseVariance(n)*(ranging(1,1)^2) 4*obj.noiseVariance(n)*(ranging(1,1)^2+ranging(4,1)^2) 0 4*obj.noiseVariance(n)*(ranging(4,1)^2) 4*obj.noiseVariance(n)*(ranging(4,1)^2);...
                                    -4*obj.noiseVariance(n)*(ranging(2,1)^2) 4*obj.noiseVariance(n)*(ranging(3,1)^2) 0 4*obj.noiseVariance(n)*(ranging(2,1)^2+ranging(3,1)^2) 4*obj.noiseVariance(n)*(ranging(2,1)^2) -4*obj.noiseVariance(n)*(ranging(3,1)^2);...
                                    -4*obj.noiseVariance(n)*(ranging(2,1)^2) 0 4*obj.noiseVariance(n)*(ranging(4,1)^2) 4*obj.noiseVariance(n)*(ranging(2,1)^2) 4*obj.noiseVariance(n)*(ranging(2,1)^2+ranging(4,1)^2) 4*obj.noiseVariance(n)*(ranging(4,1)^2);...
                                    0 -4*obj.noiseVariance(n)*(ranging(3,1)^2) 4*obj.noiseVariance(n)*(ranging(4,1)^2) -4*obj.noiseVariance(n)*(ranging(3,1)^2) 4*obj.noiseVariance(n)*(ranging(4,1)^2) 4*obj.noiseVariance(n)*(ranging(3,1)^2+ranging(4,1)^2)
                                    ];
                    end
                end
            end
            save('../data/z.mat','z');
            save('../data/toaPos.mat','toaPos');
            save('../data/R.mat','R');

            Q = zeros(2, 2, 5);
            P0 = zeros(2,2,5);
            eeT = zeros(2, 2, lnumIterations, 5);
            xxT = zeros(2, 2, lnumIterations, 5);
            vel = toaPos(:,:,2,:) - toaPos(:,:,1,:);
            vel = squeeze(vel);
            p3 = 3 * ones(size(vel));
            p2 = 2 * ones(size(vel));
            processNoise = p3 - squeeze(toaPos(:,:,2,:)) - vel;
            toaNoise = p2 - squeeze(toaPos(:,:,2,:));
            for i = 1:lnumIterations
                for n = 1:5
                    eeT(:, :, i, n) = processNoise(:, i, n) * processNoise(:, i, n)';
                    xxT(:, :, i, n) = toaNoise(:, i, n) * toaNoise(:, i, n)';
                end
            end
            EeeT = squeeze(mean(eeT, 3));
            ExxT = squeeze(mean(xxT, 3));
            processbias = squeeze(mean(processNoise, 2));
            toabias = squeeze(mean(toaNoise, 2));
            for n = 1:5
                Q(:, :, n) = EeeT(:, :, n) - processbias(:, n) * processbias(:, n)';
                P0(:, :, n) = ExxT(:, :, n) - toabias(:, n) * toabias(:, n)';
                writematrix(Q(:, :, n), strcat('../data/Q', num2str(n), '.csv'));
                writematrix(P0(:, :, n), strcat('../data/P0', num2str(n), '.csv'));
                writematrix(processNoise(:, :, n), strcat('../data/processNoise', num2str(n), '.csv'));
                writematrix(processbias(:, n), strcat('../data/processbias', num2str(n), '.csv'));
                writematrix(toaNoise(:, :, n), strcat('../data/toaNoise', num2str(n), '.csv'));
            end
        end
    end
end