clear all;
close all;
clc;

load('../data/toaPos.mat');

RMSE = RMSE();

params = struct();
params.numParticles = 400;
params.numIterations = 1e4;
params.numPoints = size(toaPos, 3);
params.numNoise = size(toaPos, 4);
params.transportIters = 12;
params.epsilonScale = 1.0;
params.gamma = 0.6;
params.anchors = [0 0 10 10; 10 0 0 10];
params.noiseVar = [0.01; 0.1; 1; 10; 100];

gtpf_data = struct();
gtpf_data.estimatedPos = zeros(2, params.numPoints, params.numIterations, params.numNoise);
gtpf_data.RMSE = zeros(params.numNoise, 1);

rng(7);

for countNoise = 1:params.numNoise
    betaSchedule = linspace(0.2, 1.0, params.transportIters);
    gtpf = GTParticleFilter(countNoise, params.numParticles, params.transportIters, params.epsilonScale, params.gamma, betaSchedule);

    sigma = sqrt(params.noiseVar(countNoise));
    Rrange = params.noiseVar(countNoise) * eye(size(params.anchors, 2));

    for countIter = 1:params.numIterations
        particles_prev = [];
        vel_prev = [];

        for countPoint = 2:params.numPoints
            truePos = [countPoint; countPoint];
            idealRanges = zeros(size(params.anchors, 2), 1);
            for m = 1:size(params.anchors, 2)
                idealRanges(m) = norm(truePos - params.anchors(:, m));
            end
            measRange = idealRanges + sigma * randn(size(idealRanges));

            if countPoint < 3
                initPrev = toaPos(:, countIter, countPoint-1, countNoise);
                initCurr = toaPos(:, countIter, countPoint, countNoise);

                gtpf_data.estimatedPos(:, countPoint-1, countIter, countNoise) = initPrev;
                gtpf_data.estimatedPos(:, countPoint, countIter, countNoise) = initCurr;

                p_prev = gtpf.sampling(initPrev);
                p_curr = gtpf.sampling(initCurr);
                particles_prev = p_curr;
                vel_prev = p_curr - p_prev;
            else
                particles_pred = gtpf.predict(particles_prev, vel_prev, 1, countPoint);
                [particles_trans, ~] = gtpf.transport(particles_pred, params.anchors, measRange, Rrange);

                est = gtpf.estimate(particles_trans);
                vel_new = est * ones(1, params.numParticles) - particles_prev;

                particles_prev = particles_trans;
                vel_prev = vel_new;

                gtpf_data.estimatedPos(:, countPoint, countIter, countNoise) = est;
            end
        end
    end
end

gtpf_data.RMSE = RMSE.getRMSE(gtpf_data.estimatedPos);

noisevalue = [0.01; 0.1; 1; 10; 100];
figure;
semilogx(noisevalue, gtpf_data.RMSE, 'DisplayName', 'GT-PF', 'LineWidth', 1.5);
grid on;
legend show;
title('Gradient Transport Particle Filter RMSE');
xlabel('Noise variance');
ylabel('RMSE');
