clear all;
close all;
clc;
disp('do preSimulate? Y/N: ')
yesorno = input('', 's');
if yesorno == 'Y'
    Env = Env(1e4);
    Env.preSimulate();
end

%% load data
load('../data/z.mat');
load('../data/toaPos.mat');
load('../data/R.mat');
%% test
RMSE = RMSE();
% parameters
params = struct();
params.numParticles = 1e3;
params.numIterations = 1e3; %size(toaPos, 2);
params.pfIterations = 1e3;
params.numPoints = size(toaPos, 3);
params.numNoise = size(toaPos, 4);
params.H = [...
    0, -20
    20, -20
    20, 0
    20, 0
    20, 20
    0, 20];
pinvH = pinv(params.H);


%% Particlefilter-------------------------------------------------------------------
pf_data = struct();
pf_data.particles = zeros(2, params.numParticles, params.numPoints, params.pfIterations, params.numNoise);
pf_data.vel = zeros(size(pf_data.particles));
pf_data.weights = params.numParticles \ ones(params.numParticles, params.numPoints, params.pfIterations, params.numNoise);
pf_data.estimatedPos = zeros(2, params.numPoints, params.pfIterations, params.numNoise);
pf_data.RMSE = zeros(params.numNoise, 1);


for countNoise = 1:params.numNoise
    pf = ParticleFilter(countNoise, params.numParticles);
    pf = thresholding(pf,countNoise); % 
    for countIter = 1:params.pfIterations
        for countPoint = 2:params.numPoints  
            meas = z(:, countIter, countPoint, countNoise);
            Rmat = R(:, :, countIter, countPoint, countNoise);
            particles_prev = pf_data.particles(:, :, countPoint-1, countIter, countNoise);
            vel_prev = pf_data.vel(:, :, countPoint-1, countIter, countNoise);
            weights_curr = pf_data.weights(:, countPoint, countIter, countNoise);
            est_prev = pf_data.estimatedPos(:, countPoint-1, countIter, countNoise);
            
            if countPoint < 3
                pf_data.estimatedPos(:, countPoint-1, countIter, countNoise) ...
                    = toaPos(:, countIter, countPoint-1, countNoise);
                pf_data.estimatedPos(:, countPoint, countIter, countNoise) ...
                    = toaPos(:, countIter, countPoint, countNoise);
                pf_data.particles(:, :, countPoint-1, countIter, countNoise) ...
                    = pf.sampling(toaPos(:, countIter, countPoint-1, countNoise)); % compare rand with countIter
                pf_data.particles(:, :, countPoint, countIter, countNoise) ...
                    = pf.sampling(toaPos(:, countIter, countPoint, countNoise));
                pf_data.vel(:, :, countPoint, countIter, countNoise) ...
                    = pf_data.particles(:, :, countPoint, countIter, countNoise) - pf_data.particles(:, :, countPoint-1, countIter, countNoise);
            else
                % particles_pred = pf.predict(particles_prev, vel_prev, 1);
                particles_pred = pf.predictParam(particles_prev, vel_prev, 1, countPoint, 0.3);

                weights_upd = pf.update(particles_pred, weights_curr, meas, params.H, Rmat);
                est = pf.estimate(particles_pred, weights_upd);
                particles_res = pf.resample(particles_pred, weights_upd);

                vel_new = est*ones(1,params.numParticles) - particles_prev;

                pf_data.particles(:, :, countPoint, countIter, countNoise) = particles_res;
                pf_data.weights(:, countPoint, countIter, countNoise) = weights_upd;
                pf_data.estimatedPos(:, countPoint, countIter, countNoise) = est;
                pf_data.vel(:, :, countPoint, countIter, countNoise) = vel_new;
            end
        end
    end
end
pf_data.RMSE = RMSE.getRMSE(pf_data.estimatedPos);


%% Kalman Filter---------------------------------------------------- 
% %{
kf_data = struct();
kf_data.estimatedPos = zeros(2, params.numPoints, params.numIterations, params.numNoise);
kf_data.errCov = zeros(2, 2, params.numPoints, params.numIterations, params.numNoise);
kf_data.vel = zeros(size(kf_data.estimatedPos));
kf_data.RMSE = zeros(params.numNoise, 1);

for countNoise = 1:params.numNoise
% for countNoise = 5
    kf = KalmanFilter(countNoise, params.H);
    for countIter = 1:params.numIterations
        for countPoint = 2:params.numPoints
            if countPoint < 3
                kf_data.estimatedPos(:, countPoint-1, countIter, countNoise) = toaPos(:, countIter, countPoint-1, countNoise);
                kf_data.estimatedPos(:, countPoint, countIter, countNoise) = toaPos(:, countIter, countPoint, countNoise);
                kf_data.vel(:, countPoint, countIter, countNoise) = kf_data.estimatedPos(:, countPoint, countIter, countNoise) - kf_data.estimatedPos(:, countPoint-1, countIter, countNoise);
            else
                [xhat, Phat] = kf.predict(kf_data.estimatedPos(:, countPoint-1, countIter, countNoise), kf_data.errCov(:, :, countPoint-1, countIter, countNoise), kf_data.vel(:, countPoint-1, countIter, countNoise), 1);
                kf = kf.update(Phat, R(:, :, countIter, countPoint, countNoise));
                [kf_data.estimatedPos(:, countPoint, countIter, countNoise), kf_data.errCov(:,:, countPoint, countIter, countNoise)] = kf.estimate(xhat, Phat, z(:, countIter, countPoint, countNoise));
                kf_data.vel(:, countPoint, countIter, countNoise) = kf_data.estimatedPos(:, countPoint, countIter, countNoise) - kf_data.estimatedPos(:, countPoint-1, countIter, countNoise);
            end
        end
    end
end

kf_data.RMSE = RMSE.getRMSE(kf_data.estimatedPos);


%% Kalman Filter modified----------------------------------------------------
kf1_data = struct();
kf1_data.estimatedPos = zeros(2, params.numPoints, params.numIterations, params.numNoise);
kf1_data.errCov = zeros(2, 2, params.numPoints, params.numIterations, params.numNoise);
kf1_data.vel = zeros(size(kf1_data.estimatedPos));
kf1_data.RMSE = zeros(params.numNoise, 1);
optimal_gamma = [0.5, 0.5, 0.4, 0.3, 0.5];

for countNoise = 1:params.numNoise
% for countNoise = 5
    kf = KalmanFilter1(countNoise, params.H);
    for countIter = 1:params.numIterations
        for countPoint = 2:params.numPoints
            if countPoint < 3
                kf1_data.estimatedPos(:, countPoint-1, countIter, countNoise) = toaPos(:, countIter, countPoint-1, countNoise);
                kf1_data.estimatedPos(:, countPoint, countIter, countNoise) = toaPos(:, countIter, countPoint, countNoise);
                kf1_data.vel(:, countPoint, countIter, countNoise) = kf1_data.estimatedPos(:, countPoint, countIter, countNoise) - kf1_data.estimatedPos(:, countPoint-1, countIter, countNoise);
            else
                [xhat, Phat] = kf.predict(kf1_data.estimatedPos(:, countPoint-1, countIter, countNoise), kf1_data.errCov(:, :, countPoint-1, countIter, countNoise), kf1_data.vel(:, countPoint-1, countIter, countNoise), 1, countPoint, optimal_gamma(countNoise));
                kf = kf.update(Phat, R(:, :, countIter, countPoint, countNoise));
                [kf1_data.estimatedPos(:, countPoint, countIter, countNoise), kf1_data.errCov(:,:, countPoint, countIter, countNoise)] = kf.estimate(xhat, Phat, z(:, countIter, countPoint, countNoise));
                kf1_data.vel(:, countPoint, countIter, countNoise) = kf1_data.estimatedPos(:, countPoint, countIter, countNoise) - kf1_data.estimatedPos(:, countPoint-1, countIter, countNoise);
            end
        end
    end
end

kf1_data.RMSE = RMSE.getRMSE(kf1_data.estimatedPos);

%}
%% Plotting-----------------------------------------------------
noisevalue = [0.01;0.1;1;10;100];
% figure;
semilogx(noisevalue,kf_data.RMSE,'DisplayName','Kalman Filter');
hold on;
semilogx(noisevalue,pf_data.RMSE,'DisplayName','Particle Filter');
semilogx(noisevalue,kf1_data.RMSE,'DisplayName','Kalman Filter 1');
legend show;
