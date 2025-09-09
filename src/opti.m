% this file is only For optimizing parameter gamma
clear all;
close all;
clc;
% disp('do preSimulate? Y/N: ')
% yesorno = input('', 's');
% if yesorno == 'Y'
%     Env = Env(1e5);
%     Env.preSimulate();
% end

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
alphaMax = 9;
pf_RMSE = zeros(params.numNoise, alphaMax);
pfopti_w_gamma = [0.6 0.6 0.4 0.2 0.3]; %분산이 클때만 거의 유효?
% Ess 는 분산이 클 때 성능향상은 0.6~0.7에 근접 
pfopti_ess_gamma = [0.8 0.8 0.8 0.7 0.7];
pfopti_ess_increase = [0.55 0.55 0.55 0.55 0.55];
pfopti_ess_decrease = [0.02 0.04 0.03 0.05 0.01];
pfopti_roughening = [];
for a = 1:alphaMax
    alpha = 0.1*a;
    pf_data = struct();
    pf_data.estimatedPos = zeros(2, params.numPoints, params.pfIterations, params.numNoise);
    % pf_data.RMSE = zeros(params.numNoise, 1);

    for countNoise = 1:params.numNoise
        pf = ParticleFilter(countNoise, params.numParticles); % 클래스 객체 불러오기
        % pf_particles = zeros(2, params.numParticles, params.numPoints, params.numIterations);
        % pf_vel = zeros(size(pf_particles));
        % pf_weights = params.numParticles\ones(params.numParticles, params.numPoints, params.numIterations);
        % pf_estimatedPos = zeros(2, params.numPoints, params.numIterations);
        % pf_errorPos = zeros(size(pf_estimatedPos));

        for countIter = 1:params.pfIterations
            particles_prev = [];
            vel_prev = [];
            weights_curr = ones(params.numParticles, 1) / params.numParticles; % 초기 가중치 균일분포
            for countPoint = 2:params.numPoints
                meas = z(:, countIter, countPoint, countNoise); % 변수가 복잡해서 따로 변수에 저장함.
                Rmat = R(:, :, countIter, countPoint, countNoise);

                if countPoint < 3 % 1, 2 스텝에서 추정값을 toa 값으로 두고, 파티클 생성 후 속도 계산.
                    % 1,2 스텝: 추정값을 TOA로 지정하고 파티클은 TOA 주변 샘플링만 수행
                    pf_data.estimatedPos(:, countPoint-1, countIter, countNoise) = toaPos(:, countIter, countPoint-1, countNoise);
                    pf_data.estimatedPos(:, countPoint,   countIter, countNoise) = toaPos(:, countIter, countPoint,   countNoise);

                    % 첫 두 포인트에 대해 각각 샘플링하여 초기 파티클 및 속도 계산
                    p_prev = pf.sampling(toaPos(:, countIter, countPoint-1, countNoise)); % size 2 x N
                    p_curr = pf.sampling(toaPos(:, countIter, countPoint,   countNoise));
                    particles_prev = p_curr; % 다음 단계의 "이전 파티클"로 사용
                    vel_prev = p_curr - p_prev; % 초기 속도 추정 (2 x N)

                else
                    % particles_pred = ...
                    %     pf.predict(particles_prev, vel_prev, 1); % 예측 : 그냥 예측 프로세스 노이즈 들어감 , f(x,u,w_k)
                    particles_pred = ...
                    pf.predictParam(particles_prev, vel_prev, 1, countPoint, alpha); % 예측 : 스텝에 따라 process 노이즈를 줄여가면서 (gamma 최적화)
                    weights_upd = ...
                        pf.update(particles_pred, weights_curr, meas, params.H, Rmat); % 측정값 반영(p(y|x), 가중치 업데이트는 여러방법이 있음 최적화 필요)
                    est = ...
                        pf.estimate(particles_pred, weights_upd); % 각 파티클의 가중치를 가지고 가중합 (posteriori)
                    % [particles_res,weights_upd] = ...
                    %     pf.resampling(particles_pred, weights_upd); % 리샘플링 (기본적으로 SIR 적용, 최적화 가능성 있음)
                    [particles_res,weights_upd] = ...
                        pf.resamplingEss(particles_pred, weights_upd); % 리샘플링 (기본적으로 SIR 적용, 최적화 가능성 있음)
                    % [particles_res,weights_upd] = ...
                    %     pf.resampling_param(particles_pred, weights_upd, countPoint,alpha); % 리샘플링 (기본적으로 SIR 적용, 최적화 가능성 있음)
                    vel_new = ...
                        est*ones(1,params.numParticles) - particles_prev;% 속도 추정: 추정값 - 이전 리샘플링 파티클(파티클 빈곤현상 있을 수 있음) roughening 해볼 수 있음.

                    % 다음 반복을 위해 로컬 변수 갱신
                    particles_prev = particles_res;
                    vel_prev = vel_new;
                    weights_curr = weights_upd;

                    pf_data.estimatedPos(:, countPoint, countIter, countNoise) = est;

                end
            end
        end
    end
    pf_RMSE(:, a) = getRMSE(RMSE, pf_data.estimatedPos);
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
[minvalue, minindex] = min(pf_RMSE,[],2);
% [minvalue, minindex] = min(kf1_RMSE,[],2);
% semilogx(Env.noiseVariance, minvalue);
disp(minindex);

