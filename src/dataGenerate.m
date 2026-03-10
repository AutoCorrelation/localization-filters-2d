function dataGenerate(config)
pathData = config.pathData;
pathResult = config.pathResult;
numSamples = config.numSamples;
noiseVariance = config.noiseVariance;
numPoints = config.numPoints;
Anchor = config.Anchor;
H = config.H;
pinvH = config.pinvH;

if ~exist(pathData, 'dir')
    mkdir(pathData)
end
if ~exist(pathResult, 'dir')
    mkdir(pathResult)
end
numNoises = length(noiseVariance);

% 셀 배열 사용 (parfor 호환)
ranging_cell = cell(numNoises, 1);
x_hat_LLS_cell = cell(numNoises, 1);
z_LLS_cell = cell(numNoises, 1);
R_LLS_cell = cell(numNoises, 1);

% CV motion model
parfor i = 1:numNoises
    ranging_temp = zeros(4, numPoints, numSamples);
    z_LLS_temp = zeros(6, numPoints, numSamples);
    R_LLS_temp = zeros(6, 6, numPoints, numSamples);
    x_hat_LLS_temp = zeros(2, numPoints, numSamples);
    noiseVar = noiseVariance(i);

    for j = 1:numSamples
        for k = 1:numPoints
            d = vecnorm([k; k] - Anchor, 2, 1).';   % 4x1
            ranging_temp(:, k, j) = d + sqrt(noiseVar) * randn(4, 1);
            z_LLS_temp(:, k, j) = [...
                ranging_temp(1, k, j)^2 - ranging_temp(2, k, j)^2 - 10^2;
                ranging_temp(1, k, j)^2 - ranging_temp(3, k, j)^2;
                ranging_temp(1, k, j)^2 - ranging_temp(4, k, j)^2 + 10^2;
                ranging_temp(2, k, j)^2 - ranging_temp(3, k, j)^2 + 10^2;
                ranging_temp(2, k, j)^2 - ranging_temp(4, k, j)^2 + 2 * 10^2;
                ranging_temp(3, k, j)^2 - ranging_temp(4, k, j)^2 + 10^2];
            R_LLS_temp(:, :, k, j) = [...
                4*noiseVar*(ranging_temp(1, k, j)^2+ranging_temp(2, k, j)^2) 4*noiseVar*(ranging_temp(1, k, j)^2) 4*noiseVar*(ranging_temp(1, k, j)^2) -4*noiseVar*(ranging_temp(2, k, j)^2) -4*noiseVar*(ranging_temp(2, k, j)^2) 0;...
                4*noiseVar*(ranging_temp(1, k, j)^2) 4*noiseVar*(ranging_temp(1, k, j)^2+ranging_temp(3, k, j)^2) 4*noiseVar*(ranging_temp(1, k, j)^2) 4*noiseVar*(ranging_temp(3, k, j)^2) 0 -4*noiseVar*(ranging_temp(3, k, j)^2);...
                4*noiseVar*(ranging_temp(1, k, j)^2) 4*noiseVar*(ranging_temp(1, k, j)^2) 4*noiseVar*(ranging_temp(1, k, j)^2+ranging_temp(4, k, j)^2) 0 4*noiseVar*(ranging_temp(4, k, j)^2) 4*noiseVar*(ranging_temp(4, k, j)^2);...
                -4*noiseVar*(ranging_temp(2, k, j)^2) 4*noiseVar*(ranging_temp(3, k, j)^2) 0 4*noiseVar*(ranging_temp(2, k, j)^2+ranging_temp(3, k, j)^2) 4*noiseVar*(ranging_temp(2, k, j)^2) -4*noiseVar*(ranging_temp(3, k, j)^2);...
                -4*noiseVar*(ranging_temp(2, k, j)^2) 0 4*noiseVar*(ranging_temp(4, k, j)^2) 4*noiseVar*(ranging_temp(2, k, j)^2) 4*noiseVar*(ranging_temp(2, k, j)^2+ranging_temp(4, k, j)^2) 4*noiseVar*(ranging_temp(4, k, j)^2);...
                0 -4*noiseVar*(ranging_temp(3, k, j)^2) 4*noiseVar*(ranging_temp(4, k, j)^2) -4*noiseVar*(ranging_temp(3, k, j)^2) 4*noiseVar*(ranging_temp(4, k, j)^2) 4*noiseVar*(ranging_temp(3, k, j)^2+ranging_temp(4, k, j)^2)];
            x_hat_LLS_temp(:, k, j) = pinvH * z_LLS_temp(:, k, j);
        end
    end

    % 슬라이스 변수: 루프 인덱스 i만 사용
    ranging_cell{i} = ranging_temp;
    x_hat_LLS_cell{i} = x_hat_LLS_temp;
    z_LLS_cell{i} = z_LLS_temp;
    R_LLS_cell{i} = R_LLS_temp;
end

% parfor 이후: 셀 배열을 다차원 배열로 변환
ranging = cat(4, ranging_cell{:});
x_hat_LLS = cat(4, x_hat_LLS_cell{:});
z_LLS = cat(4, z_LLS_cell{:});
R_LLS = cat(5, R_LLS_cell{:});

Q = zeros(2, 2, 5);
P0 = zeros(2,2,5);
vel = x_hat_LLS(:,2,:,:) - x_hat_LLS(:,1,:,:);
vel = squeeze(vel);
p3 = 3 * ones(size(vel));
p2 = 2 * ones(size(vel));
processNoise = p3 - squeeze(x_hat_LLS(:,2,:,:)) - vel;
toaNoise = p2 - squeeze(x_hat_LLS(:,2,:,:));
eeT_all = cell(5, 1);
xxT_all = cell(5, 1);
parfor n = 1:5
    eeT_n = zeros(2, 2, numSamples);
    xxT_n = zeros(2, 2, numSamples);

    for i = 1:numSamples
        eeT_n(:, :, i) = processNoise(:, i, n) * processNoise(:, i, n)';
        xxT_n(:, :, i) = toaNoise(:, i, n) * toaNoise(:, i, n)';
    end

    eeT_all{n} = eeT_n;
    xxT_all{n} = xxT_n;
end

eeT = cat(4, eeT_all{:});
xxT = cat(4, xxT_all{:});
EeeT = squeeze(mean(eeT, 3));
ExxT = squeeze(mean(xxT, 3));

processbias = squeeze(mean(processNoise, 2));
toabias = squeeze(mean(toaNoise, 2));
for n = 1:5
    Q(:, :, n) = EeeT(:, :, n) - processbias(:, n) * processbias(:, n)';
    P0(:, :, n) = ExxT(:, :, n) - toabias(:, n) * toabias(:, n)';
end

% H5 파일 생성 및 저장
h5File = fullfile(pathData, 'simulation_data.h5');

% 기존 파일 삭제
if isfile(h5File)
    delete(h5File)
end

% 데이터 저장
h5create(h5File, '/ranging', size(ranging), 'DataType', 'double');
h5write(h5File, '/ranging', ranging);

h5create(h5File, '/x_hat_LLS', size(x_hat_LLS), 'DataType', 'double');
h5write(h5File, '/x_hat_LLS', x_hat_LLS);

h5create(h5File, '/z_LLS', size(z_LLS), 'DataType', 'double');
h5write(h5File, '/z_LLS', z_LLS);

h5create(h5File, '/R_LLS', size(R_LLS), 'DataType', 'double');
h5write(h5File, '/R_LLS', R_LLS);

h5create(h5File, '/Q', size(Q), 'DataType', 'double');
h5write(h5File, '/Q', Q);

h5create(h5File, '/P0', size(P0), 'DataType', 'double');
h5write(h5File, '/P0', P0);

h5create(h5File, '/processNoise', size(processNoise), 'DataType', 'double');
h5write(h5File, '/processNoise', processNoise);

h5create(h5File, '/toaNoise', size(toaNoise), 'DataType', 'double');
h5write(h5File, '/toaNoise', toaNoise);

h5create(h5File, '/processbias', size(processbias), 'DataType', 'double');
h5write(h5File, '/processbias', processbias);

fprintf('Data saved to %s\n', h5File);
end