clear all;
close all;
clc;
disp('do preSimulate? Y/N: ')
yesorno = input('', 's');
if yesorno == 'Y'
    Env = Env(1e5);
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
params.numParticles = 1e4;
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
pf_data.estimatedPos = zeros(2, params.numPoints, params.pfIterations, params.numNoise);
pf_data.RMSE = zeros(params.numNoise, 1);
pfopti_w_gamma = [0.6 0.6 0.4 0.2 0.3];
pfopti_ess_gamma = [0.55 0.55 0.55 0.55 0.55];


for countNoise = 1:params.numNoise
    pf = ParticleFilter(countNoise, params.numParticles); % 클래스 객체 불러오기
    % pf = thresholding(pf,countNoise); % 쓰레기홀딩
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
                    pf.predictParam(particles_prev, vel_prev, 1, countPoint, pfopti_w_gamma(countNoise)); % 예측 : 스텝에 따라 process 노이즈를 줄여가면서 (gamma 최적화)
                weights_upd = ...
                    pf.update(particles_pred, weights_curr, meas, params.H, Rmat); % 측정값 반영(p(y|x), 가중치 업데이트는 여러방법이 있음 최적화 필요)
                est = ...
                    pf.estimate(particles_pred, weights_upd); % 각 파티클의 가중치를 가지고 가중합 (posteriori)
                % [particles_res,weights_upd] = ...
                %     pf.resampling(particles_pred, weights_upd); % 리샘플링 (기본적으로 SIR 적용, 최적화 가능성 있음)
                [particles_res,weights_upd] = ...
                    pf.resamplingEss(particles_pred, weights_upd); % 리샘플링 (기본적으로 SIR 적용, 최적화 가능성 있음)
                % [particles_res,weights_upd] = ...
                %     pf.resampling_param(particles_pred, weights_upd, countPoint, pfopti_ess_gamma(countNoise)); % 리샘플링 (기본적으로 SIR 적용, 최적화 가능성 있음)

                % particles_res = particles_pred; % 리샘플링 안함 (파티클 빈곤현상 고려안함)
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
pf_data.RMSE = RMSE.getRMSE(pf_data.estimatedPos); % 성능 지표 RMSE


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
semilogx(noisevalue,kf1_data.RMSE,'DisplayName','Kalman Filter 1');
semilogx(noisevalue,pf_data.RMSE,'DisplayName','Particle Filter','LineWidth',1.5);
legend show;
