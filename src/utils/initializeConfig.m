function config = initializeConfig(numParticle)
    %INITIALIZECONFIG Initialize configuration parameters
    %   Returns a struct containing all simulation parameters
    
    config.pathData = '../data/';
    config.pathResult = '../result/';
    config.numSamples = 1e5;
    config.iterations = 1e3;
    config.noiseVariance = [1e-2, 1e-1, 1, 1e1, 1e2];
    config.numPoints = 10;
    config.Anchor = [0 10; 0 0; 10 0; 10 10]';
    config.H = [...
        0, -20
        20, -20
        20, 0
        20, 0
        20, 20
        0, 20];
    config.pinvH = pinv(config.H);

    % Motion model selector: set ONE of the following as active
    config.motionModel = 'cv';          % Use Constant-Velocity trajectory (default)
    % config.motionModel = 'imm';       % Use IMM (CV/CT) trajectory (uncomment to use)

    if nargin > 0
        config.numParticles = numParticle;
    else
        config.numParticles = 150;
    end
    config.resampleThresholdRatio = 0.2;
    config.decayGamma = [0.4 0.5 0.5 0.3 0.5];

    % IAE + MAP adaptive PF defaults
    config.iaeWindowLength = 20;
    config.iaeQFloor = 1e-6;
    config.iaeQCeil = 10;
    config.iaeRegLambda = 1e-8;
    config.iaeProcessQScale = 1.0;
    config.mapParameterJitterStd = 1e-3;
    config.mapFeedbackGain = 0.2;

    % KLD likelihood-adaptive PF defaults (A-BPF / A-GPF style)
    config.kldThetaMin = 0.0;
    config.kldThetaMax = 1.0;
    config.kldRegLambda = 1e-8;
    config.kldQFloor = 1e-8;
    config.kldQCeil = 1e3;
    config.kldThetaFallback = 0.5;

    % Belief-Q-shrink adaptive PF defaults
    config.beliefQShrinkGain = 0.3;
    config.beliefQShrinkMinScale = 0.35;

    % R-diag prior-edit adaptive PF defaults
    config.rdiagPriorSigmaGate = 6.0;
    config.rdiagPriorMaxRetry = 20;
    config.rdiagRougheningK = 0.2;

    % Belief-roughening adaptive PF defaults
    config.beliefRougheningKBase = 0.2;
    config.beliefRougheningGain = 0.6;
    config.beliefRougheningKMax = 1.5;

    % Roughening + Prior Editing PF defaults
    config.rougheningK = 0.2;
    config.priorSigmaGate = 6.0;
    config.priorMaxRetry = 30;

    % EKF-proposal PF defaults
    config.ekfEnabled = true;
    config.ekfQScale = 1.0;
    config.ekfUseDataQ = true;
    config.ekfUseDataP0 = true;

    % IMM trajectory generation defaults (used when motionModel='imm')
    config.immDt = 1.0;
    config.immSigmaV = 0.5;
    config.immOmegas = [0, 0.15, -0.15];
    config.immTPM = [0.90, 0.05, 0.05; 0.10, 0.90, 0.00; 0.10, 0.00, 0.90];
    config.immInitialModeProb = [1; 0; 0];
    config.immInitialState = [0; 0; 10; 10];
end
