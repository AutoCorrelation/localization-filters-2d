clc; clear; format long;

%% HDF5 reading
% allRanging(4x11x11000x5), distance(4x11), true_position(2x11)
% ranging_001(4x11x11000) ... ranging_100(4x11x11000)
h5path = 'ranging_data_cv.h5';

info = h5info(h5path);
data.allRanging = h5read(h5path, '/allRanging');
data.allRanging_corrected = h5read(h5path, '/allRanging_corrected');
data.distance = h5read(h5path, '/distance');
data.true_position = h5read(h5path, '/true_position');


%% Filter estimation
% particle filter
STEP = size(data.allRanging, 2);
ITERATION = size(data.allRanging, 3);
NOISE_SIZE = size(data.allRanging, 4);
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool(10);
end
RUN_BASELINE = true; % baseline (LLS)
RUN_PF  = true;   % Particle Filter
RUN_EKF = true;   % EKF
RUN_DNN = false;  %

results = struct();
estimatedPF = zeros([size(data.true_position),ITERATION, NOISE_SIZE]);

% for N = 1:NOISE_SIZE
%     parfor I = 1:ITERATION
%         estimatedPF(:,:,I,N) = (data.allRanging_corrected(:,:,I,N));
%     end
% end

% results.estimatedPF = estimatedPF;
% clear estimatedPF;

addpath('src');
addpath('src/utils');
% 기본 경로 자동 탐색 + 기본 저장

% 경로/저장파일 지정
stats = estimateCorrectedMeasurementCovariance('ranging_data_cv.h5', 'R_corrected_stats.mat');
opts = struct();
opts.filterList = {'Baseline'; 'NonlinearParticleFilter'};
opts.compareCorrected = true;
opts.disablePredictNoiseCorrected = true;
% 필요시 직접 주입 가능:
load('R_corrected_stats.mat');
opts.RCorrectedPoint = stats.R_corrected_point; 

results = quick_eval('ranging_data_cv.h5', opts);


%% HDF5 (re)writing
dset = '/estimatedPF';
try
    h5info(h5path, dset);
    fid = H5F.open(h5path, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
    H5L.delete(fid, dset, 'H5P_DEFAULT');
    H5F.close(fid);
catch

end

% rewrite
h5create(h5path, dset, size(results.estimatedPF), 'Datatype', class(results.estimatedPF));
h5write(h5path, dset, results.estimatedPF);

h5disp(h5path);