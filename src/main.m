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
params.numParticles = 5e3;
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
pf_data = struct(); % pf 구조체 선언 (파티클 필터 관련된 데이터는 여기다가 다 저장함, 변수 차원 순서는 보통 변수, 파티클개수, 포인트 개수, 반복횟수, 노이즈 순번)
pf_data.particles = zeros(2, params.numParticles, params.numPoints, params.pfIterations, params.numNoise);
pf_data.vel = zeros(size(pf_data.particles));
pf_data.weights = params.numParticles \ ones(params.numParticles, params.numPoints, params.pfIterations, params.numNoise); % 기본 가중치 1/N 으로 통일
pf_data.estimatedPos = zeros(2, params.numPoints, params.pfIterations, params.numNoise);
pf_data.RMSE = zeros(params.numNoise, 1);


for countNoise = 1:params.numNoise
    pf = ParticleFilter(countNoise, params.numParticles); % 클래스 객체 불러오기
    pf = thresholding(pf,countNoise); % 쓰레기홀딩
    for countIter = 1:params.pfIterations
        for countPoint = 2:params.numPoints  
            meas = z(:, countIter, countPoint, countNoise); % 변수가 복잡해서 따로 변수에 저장함.
            Rmat = R(:, :, countIter, countPoint, countNoise);
            particles_prev = pf_data.particles(:, :, countPoint-1, countIter, countNoise);
            vel_prev = pf_data.vel(:, :, countPoint-1, countIter, countNoise);
            weights_curr = pf_data.weights(:, countPoint, countIter, countNoise);
            est_prev = pf_data.estimatedPos(:, countPoint-1, countIter, countNoise);
            
            if countPoint < 3 % 1, 2 스텝에서 추정값을 toa 값으로 두고, 파티클 생성 후 속도 계산.
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
                particles_pred = pf.predict(particles_prev, vel_prev, 1); % 예측 : 그냥 예측 프로세스 노이즈 들어감 , f(x,u,w_k)
                % particles_pred = pf.predictParam(particles_prev, vel_prev, 1, countPoint, 0.3); % 예측 : 스텝에 따라 process 노이즈를 줄여가면서 (gamma 최적화)

                weights_upd = pf.update(particles_pred, weights_curr, meas, params.H, Rmat); % 측정값 반영(p(y|x), 가중치 업데이트는 여러방법이 있음 최적화 필요)
                est = pf.estimate(particles_pred, weights_upd); % 각 파티클의 가중치를 가지고 가중합 (posteriori)
                particles_res = pf.resample(particles_pred, weights_upd); % 리샘플링 (기본적으로 SIR 적용, 최적화 가능성 있음)  

                % particles_res = pf.roughening(particles_res, 0.2); % roughening

                vel_new = est*ones(1,params.numParticles) - particles_prev;% 속도 추정: 추정값 - 이전 리샘플링 파티클(파티클 빈곤현상 있을 수 있음) roughening 해볼 수 있음.

                pf_data.particles(:, :, countPoint, countIter, countNoise) = particles_res; % 리샘플링한 파티클을 이전 파티클로 지정
                pf_data.weights(:, countPoint, countIter, countNoise) = weights_upd; % 
                pf_data.estimatedPos(:, countPoint, countIter, countNoise) = est;
                pf_data.vel(:, :, countPoint, countIter, countNoise) = vel_new;
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
semilogx(noisevalue,pf_data.RMSE,'DisplayName','Particle Filter');
semilogx(noisevalue,kf1_data.RMSE,'DisplayName','Kalman Filter 1');
legend show;
