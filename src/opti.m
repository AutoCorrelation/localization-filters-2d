% this file is only For optimizing parameter gamma
clear all;
close all;
clc;
yesorno = input('do preSimulate? Y/N: ','s');
if yesorno == 'Y'
    Env = Env(1e4);
    Env.preSimulate();
end
%%
% load data
load('../data/z.mat');
load('../data/toaPos.mat');
load('../data/R.mat');
%
% parameters
params = struct();
params.numParticles = 2000;
params.numIterations = 1e4; %size(toaPos, 2);
alphaMax = 9;
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
%{%
% Particlefilter
RMSE = RMSE();
pf_RMSE = zeros(params.numNoise, alphaMax);

for a = 1:alphaMax
    alpha = 0.1*a;
    for countNoise = 1:params.numNoise
        % for countNoise = 5
        pf = ParticleFilter(countNoise, params.numParticles);
        pf_particles = zeros(2, params.numParticles, params.numPoints, params.numIterations);
        pf_vel = zeros(size(pf_particles));
        pf_weights = params.numParticles\ones(params.numParticles, params.numPoints, params.numIterations);
        pf_estimatedPos = zeros(2, params.numPoints, params.numIterations);
        pf_errorPos = zeros(size(pf_estimatedPos));

        for countIter = 1:params.numIterations
            for countPoint = 2:params.numPoints
                if countPoint < 3
                    pf_estimatedPos(:, countPoint-1, countIter) = toaPos(:, countIter, countPoint-1, countNoise);
                    pf_estimatedPos(:, countPoint, countIter) = toaPos(:, countIter, countPoint, countNoise);
                    pf_particles(:, :, countPoint-1, countIter) = sampling(pf, toaPos(:, countIter, countPoint-1, countNoise)); % compare rand with countIter
                    pf_particles(:, :, countPoint, countIter) = sampling(pf, toaPos(:, countIter, countPoint, countNoise));
                    pf_vel(:, :, countPoint, countIter) = pf_particles(:, :, countPoint, countIter) - pf_particles(:, :, countPoint-1, countIter);
                else
                    % pf_particles(:, :, countPoint, countIter) = predict(pf, pf_particles(:, :, countPoint-1, countIter), pf_vel(:, :, countPoint-1, countIter), 1);
                    pf_particles(:, :, countPoint, countIter) = predictParam(pf, pf_particles(:, :, countPoint-1, countIter), pf_vel(:, :, countPoint-1, countIter), 1, countPoint, alpha);
                    pf_weights(:, countPoint, countIter) = update(pf, pf_particles(:, :, countPoint, countIter), pf_weights(:, countPoint, countIter), z(:, countIter, countPoint, countNoise), params.H, R(:, :, countIter, countPoint, countNoise));
                    % pf_weights(:, countPoint, countIter) = updateParam(pf, pf_particles(:, :, countPoint, countIter), pf_weights(:, countPoint, countIter), z(:, countIter, countPoint, countNoise), params.H, R(:, :, countIter, countPoint, countNoise),0.3);
                    pf_estimatedPos(:, countPoint, countIter) = estimate(pf, pf_particles(:, :, countPoint, countIter), pf_weights(:, countPoint, countIter));
                    pf_particles(:, :, countPoint, countIter) = resample(pf, pf_particles(:, :, countPoint, countIter), pf_weights(:, countPoint, countIter));
                    % pf_particles(:, :, countPoint, countIter) = resample2(pf, pf_particles(:, :, countPoint, countIter), pf_weights(:, countPoint, countIter));
                    pf_vel(:, :, countPoint, countIter) = pf_particles(:, :, countPoint, countIter) - pf_particles(:, :, countPoint-1, countIter);
                end
            end
        end

        % error
        pf_RMSE(countNoise, a) = getRMSE(RMSE, pf_estimatedPos);
    end
end
%}

%{
kf1_RMSE = zeros(params.numNoise, alphaMax);
for a = 1:alphaMax
    alpha = 0.1*a;
    kf1_data = struct();
    kf1_data.estimatedPos = zeros(2, params.numPoints, params.numIterations, params.numNoise);
    kf1_data.errCov = zeros(2, 2, params.numPoints, params.numIterations, params.numNoise);
    kf1_data.vel = zeros(size(kf1_data.estimatedPos));
    kf1_data.RMSE = zeros(params.numNoise, alphaMax);


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
                    [xhat, Phat] = kf.predict(kf1_data.estimatedPos(:, countPoint-1, countIter, countNoise), kf1_data.errCov(:, :, countPoint-1, countIter, countNoise), kf1_data.vel(:, countPoint-1, countIter, countNoise), 1, countPoint, alpha);
                    kf = kf.update(Phat, R(:, :, countIter, countPoint, countNoise));
                    [kf1_data.estimatedPos(:, countPoint, countIter, countNoise), kf1_data.errCov(:,:, countPoint, countIter, countNoise)] = kf.estimate(xhat, Phat, z(:, countIter, countPoint, countNoise));
                    kf1_data.vel(:, countPoint, countIter, countNoise) = kf1_data.estimatedPos(:, countPoint, countIter, countNoise) - kf1_data.estimatedPos(:, countPoint-1, countIter, countNoise);
                end
            end
        end
    end
    kf1_RMSE(: , a) = getRMSE(RMSE, kf1_data.estimatedPos);
end
%}

% find minimum RMSE
% [minvalue, minindex] = min(pf_RMSE,[],2);
[minvalue, minindex] = min(kf1_RMSE,[],2);
% semilogx(Env.noiseVariance, minvalue);
disp(minindex);

