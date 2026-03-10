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

    if nargin > 0
        config.numParticles = numParticle;
    else
        config.numParticles = 150;
    end
    config.resampleThresholdRatio = 0.5;
    config.decayGamma = [0.6 0.6 0.6 0.4 0.2];
end
