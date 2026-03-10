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
            lAnchor = obj.Anchor;
            lnoiseVariance = obj.noiseVariance;
            lnumIterations = obj.numIterations;
            lnumPoints = obj.numPoints;
            
            % Pre-compute H and pinvH outside parfor
            H = [...
                0, -20
                20, -20
                20, 0
                20, 0
                20, 20
                0, 20];
            pinvH = pinv(H);
            
            % Initialize output arrays
            z = zeros(6, lnumIterations, lnumPoints, 5);
            toaPos = zeros(2, lnumIterations, lnumPoints, 5);
            ranging = zeros(4, lnumIterations, lnumPoints, 5);
            R = zeros(6, 6, lnumIterations, lnumPoints, 5);
            
            % Initialize parpool
            poolobj = gcp('nocreate');
            if isempty(poolobj)
                parpool(5);  % 5 workers for 5 noise levels
            end
            
            % Parallelize over noise levels
            tic;
            z_all = cell(5, 1);
            toaPos_all = cell(5, 1);
            ranging_all = cell(5, 1);
            R_all = cell(5, 1);
            
            parfor n = 1:5
                z_n = zeros(6, lnumIterations, lnumPoints);
                toaPos_n = zeros(2, lnumIterations, lnumPoints);
                ranging_n = zeros(4, lnumIterations, lnumPoints);
                R_n = zeros(6, 6, lnumIterations, lnumPoints);
                
                for i = 1:lnumIterations
                    for p = 1:lnumPoints
                        actualPos = [p; p];
                        ranging_temp = zeros(4, 1);
                        
                        for a = 1:4
                            ranging_temp(a,1) = norm(actualPos - lAnchor(:,a)) + sqrt(lnoiseVariance(n)) * randn;
                        end
                        ranging_n(:, i, p) = ranging_temp;
                        
                        z_n(:,i,p) = [...
                            ranging_temp(1, 1)^2 - ranging_temp(2, 1)^2 - 10^2;
                            ranging_temp(1, 1)^2 - ranging_temp(3, 1)^2;
                            ranging_temp(1, 1)^2 - ranging_temp(4, 1)^2 + 10^2;
                            ranging_temp(2, 1)^2 - ranging_temp(3, 1)^2 + 10^2;
                            ranging_temp(2, 1)^2 - ranging_temp(4, 1)^2 + 2*(10^2);
                            ranging_temp(3, 1)^2 - ranging_temp(4, 1)^2 + 10^2];
                        
                        toaPos_n(:,i,p) = pinvH * z_n(:,i,p);
                        
                        R_n(:,:,i,p) = [4*lnoiseVariance(n)*(ranging_temp(1,1)^2+ranging_temp(2,1)^2) 4*lnoiseVariance(n)*(ranging_temp(1,1)^2) 4*lnoiseVariance(n)*(ranging_temp(1,1)^2) -4*lnoiseVariance(n)*(ranging_temp(2,1)^2) -4*lnoiseVariance(n)*(ranging_temp(2,1)^2) 0;...
                                        4*lnoiseVariance(n)*(ranging_temp(1,1)^2) 4*lnoiseVariance(n)*(ranging_temp(1,1)^2+ranging_temp(3,1)^2) 4*lnoiseVariance(n)*(ranging_temp(1,1)^2) 4*lnoiseVariance(n)*(ranging_temp(3,1)^2) 0 -4*lnoiseVariance(n)*(ranging_temp(3,1)^2);...
                                        4*lnoiseVariance(n)*(ranging_temp(1,1)^2) 4*lnoiseVariance(n)*(ranging_temp(1,1)^2) 4*lnoiseVariance(n)*(ranging_temp(1,1)^2+ranging_temp(4,1)^2) 0 4*lnoiseVariance(n)*(ranging_temp(4,1)^2) 4*lnoiseVariance(n)*(ranging_temp(4,1)^2);...
                                        -4*lnoiseVariance(n)*(ranging_temp(2,1)^2) 4*lnoiseVariance(n)*(ranging_temp(3,1)^2) 0 4*lnoiseVariance(n)*(ranging_temp(2,1)^2+ranging_temp(3,1)^2) 4*lnoiseVariance(n)*(ranging_temp(2,1)^2) -4*lnoiseVariance(n)*(ranging_temp(3,1)^2);...
                                        -4*lnoiseVariance(n)*(ranging_temp(2,1)^2) 0 4*lnoiseVariance(n)*(ranging_temp(4,1)^2) 4*lnoiseVariance(n)*(ranging_temp(2,1)^2) 4*lnoiseVariance(n)*(ranging_temp(2,1)^2+ranging_temp(4,1)^2) 4*lnoiseVariance(n)*(ranging_temp(4,1)^2);...
                                        0 -4*lnoiseVariance(n)*(ranging_temp(3,1)^2) 4*lnoiseVariance(n)*(ranging_temp(4,1)^2) -4*lnoiseVariance(n)*(ranging_temp(3,1)^2) 4*lnoiseVariance(n)*(ranging_temp(4,1)^2) 4*lnoiseVariance(n)*(ranging_temp(3,1)^2+ranging_temp(4,1)^2)];
                    end
                end
                
                z_all{n} = z_n;
                toaPos_all{n} = toaPos_n;
                ranging_all{n} = ranging_n;
                R_all{n} = R_n;
            end
            
            % Merge cell results back to arrays
            for n = 1:5
                z(:, :, :, n) = z_all{n};
                toaPos(:, :, :, n) = toaPos_all{n};
                ranging(:, :, :, n) = ranging_all{n};
                R(:, :, :, :, n) = R_all{n};
            end
            ptime = toc;
            fprintf('Parallel data generation completed in %.2f seconds\n', ptime);
            
            % Save main data files
            save('../data/z.mat','z');
            save('../data/toaPos.mat','toaPos');
            save('../data/ranging.mat','ranging');
            save('../data/R.mat','R');

            % Post-processing: statistics and noise characterization (sequential)
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
            
            tic;
            eeT_all = cell(5, 1);
            xxT_all = cell(5, 1);
            
            parfor n = 1:5
                eeT_n = zeros(2, 2, lnumIterations);
                xxT_n = zeros(2, 2, lnumIterations);
                
                for i = 1:lnumIterations
                    eeT_n(:, :, i) = processNoise(:, i, n) * processNoise(:, i, n)';
                    xxT_n(:, :, i) = toaNoise(:, i, n) * toaNoise(:, i, n)';
                end
                
                eeT_all{n} = eeT_n;
                xxT_all{n} = xxT_n;
            end
            
            % Merge cell results back to arrays
            for n = 1:5
                eeT(:, :, :, n) = eeT_all{n};
                xxT(:, :, :, n) = xxT_all{n};
            end
            ptime2 = toc;
            fprintf('Parallel statistics computation completed in %.2f seconds\n', ptime2);
            
            EeeT = squeeze(mean(eeT, 3));
            ExxT = squeeze(mean(xxT, 3));
            processbias = squeeze(mean(processNoise, 2));
            toabias = squeeze(mean(toaNoise, 2));
            for n = 1:5
                Q(:, :, n) = EeeT(:, :, n) - processbias(:, n) * processbias(:, n)';
                P0(:, :, n) = ExxT(:, :, n) - toabias(:, n) * toabias(:, n)';
            end
            
            % Parallel file I/O for noise characterization
            tic;
            parfor n = 1:5
                writematrix(Q(:, :, n), strcat('../data/Q', num2str(n), '.csv'));
                writematrix(P0(:, :, n), strcat('../data/P0', num2str(n), '.csv'));
                writematrix(processNoise(:, :, n), strcat('../data/processNoise', num2str(n), '.csv'));
                writematrix(processbias(:, n), strcat('../data/processbias', num2str(n), '.csv'));
                writematrix(toaNoise(:, :, n), strcat('../data/toaNoise', num2str(n), '.csv'));
                T=array2table(processNoise(:,:, n).', 'VariableNames',{'x','y'});
                writetable(T, strcat('../data/processNoise_table', num2str(n), '.csv'));
            end
            ptime3 = toc;
            fprintf('Parallel file I/O completed in %.2f seconds\n', ptime3);
        end
%{
        function preSimulateH5(obj)
            ranging = zeros(4, 1);
            lAnchor = obj.Anchor;
            lnoiseVariance = obj.noiseVariance;
            lnumIterations = obj.numIterations;
            lnumPoints = obj.numPoints;
            
            z = zeros(6, lnumIterations, lnumPoints, 5);
            toaPos = zeros(2, lnumIterations, lnumPoints, 5);
            realPos = zeros(2, lnumIterations, lnumPoints, 5);  % toaPos와 동일 shape
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
                    actualPos = [p; p];
                    for n = 1:5
                        realPos(:, i, p, n) = actualPos;  % 모든 노이즈에 동일하게 저장
                        for a = 1:4
                            ranging(a,1) = norm(actualPos - lAnchor(:,a)) + sqrt(lnoiseVariance(n)) * randn;
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
            
            % Create HDF5 file
            h5filename = '../data/simulation_data.h5';
            if isfile(h5filename)
                delete(h5filename);
            end
            
            % Save z, toaPos, realPos, R to HDF5
            h5create(h5filename, '/toaPos', size(toaPos), 'DataType', 'double');
            h5write(h5filename, '/toaPos', toaPos);
            h5create(h5filename, '/realPos', size(realPos), 'DataType', 'double');
            h5write(h5filename, '/realPos', realPos);

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
            
            % Save Q, P0, processNoise, toaNoise, processbias to HDF5
            
            
            for n = 1:5
                Q(:, :, n) = EeeT(:, :, n) - processbias(:, n) * processbias(:, n)';
                P0(:, :, n) = ExxT(:, :, n) - toabias(:, n) * toabias(:, n)';
            end
            
            
        end
        %}
    end
end