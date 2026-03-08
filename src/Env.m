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
            
            % Pre-compute true (noiseless) ranging distances: 4 x numPoints.
            % true_dists(a,p) = norm([p;p] - anchor_a)
            p_vals = 1:lnumPoints;  % 1 x numPoints
            true_dists = sqrt((p_vals - lAnchor(1, :)').^2 + (p_vals - lAnchor(2, :)').^2);  % 4 x numPoints

            % Fixed coefficient matrices that decompose R as a weighted sum:
            %   R = 4*sigma^2 * (A1*r1^2 + A2*r2^2 + A3*r3^2 + A4*r4^2)
            A1 = [1 1 1 0 0 0; 1 1 1 0 0 0; 1 1 1 0 0 0; zeros(3, 6)];
            A2 = [1 0 0 -1 -1 0; zeros(2,6); -1 0 0 1 1 0; -1 0 0 1 1 0; zeros(1,6)];
            A3 = [zeros(1,6); 0 1 0 1 0 -1; zeros(1,6); 0 1 0 1 0 -1; zeros(1,6); 0 -1 0 -1 0 1];
            A4 = [zeros(2,6); 0 0 1 0 1 1; zeros(1,6); 0 0 1 0 1 1; 0 0 1 0 1 1];

            % Reshape to 6x6x1x1 for broadcasting over (iterations x points)
            A1_4d = reshape(A1, 6, 6, 1, 1);
            A2_4d = reshape(A2, 6, 6, 1, 1);
            A3_4d = reshape(A3, 6, 6, 1, 1);
            A4_4d = reshape(A4, 6, 6, 1, 1);

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
                % --- Ranging: 4 x numIterations x numPoints ---
                % Broadcast true_dists (4x1xnumPoints) over iterations dimension
                noise = sqrt(lnoiseVariance(n)) * randn(4, lnumIterations, lnumPoints);
                ranging_n = reshape(true_dists, 4, 1, lnumPoints) + noise;  % 4 x I x P

                % --- TDOA measurement vector z: 6 x numIterations x numPoints ---
                r_sq = ranging_n.^2;
                z_n = zeros(6, lnumIterations, lnumPoints);
                z_n(1,:,:) = r_sq(1,:,:) - r_sq(2,:,:) - 100;
                z_n(2,:,:) = r_sq(1,:,:) - r_sq(3,:,:);
                z_n(3,:,:) = r_sq(1,:,:) - r_sq(4,:,:) + 100;
                z_n(4,:,:) = r_sq(2,:,:) - r_sq(3,:,:) + 100;
                z_n(5,:,:) = r_sq(2,:,:) - r_sq(4,:,:) + 200;
                z_n(6,:,:) = r_sq(3,:,:) - r_sq(4,:,:) + 100;

                % --- LLS position estimate: 2 x numIterations x numPoints ---
                z_flat = reshape(z_n, 6, lnumIterations * lnumPoints);
                toaPos_flat = pinvH * z_flat;
                toaPos_n = reshape(toaPos_flat, 2, lnumIterations, lnumPoints);

                % --- Measurement covariance R: 6 x 6 x numIterations x numPoints ---
                % R = 4*sigma^2 * (A1*r1^2 + A2*r2^2 + A3*r3^2 + A4*r4^2)
                % Reshape each r_sq row to 1x1xIxP for broadcasting with 6x6x1x1 A matrices
                r1_sq = reshape(r_sq(1,:,:), 1, 1, lnumIterations, lnumPoints);
                r2_sq = reshape(r_sq(2,:,:), 1, 1, lnumIterations, lnumPoints);
                r3_sq = reshape(r_sq(3,:,:), 1, 1, lnumIterations, lnumPoints);
                r4_sq = reshape(r_sq(4,:,:), 1, 1, lnumIterations, lnumPoints);
                R_n = 4 * lnoiseVariance(n) * ...
                    (A1_4d .* r1_sq + A2_4d .* r2_sq + A3_4d .* r3_sq + A4_4d .* r4_sq);

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
                % Compute batch outer products via broadcasting instead of a loop.
                % processNoise(:,:,n) is 2 x I; reshape to (2x1xI) and (1x2xI)
                % to get the (2x2xI) stack of outer products in one shot.
                pn = processNoise(:, :, n);  % 2 x I
                tn = toaNoise(:, :, n);      % 2 x I
                eeT_all{n} = reshape(pn, 2, 1, lnumIterations) .* reshape(pn, 1, 2, lnumIterations);
                xxT_all{n} = reshape(tn, 2, 1, lnumIterations) .* reshape(tn, 1, 2, lnumIterations);
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