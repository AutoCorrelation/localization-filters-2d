function [beta, lambdaR] = getBestParams(noiseIdx)
    % noiseVar에 따른 최적 하이퍼파라미터 매핑 (그리드 서치 결과 반영)
    % CV model
    % switch noiseIdx
    %     case 1
    %         beta = 0.999; lambdaR = 200;
    %     case 2
    %         beta = 0.995; lambdaR = 50;
    %     case 3
    %         beta = 0.995; lambdaR = 50;
    %     case 4
    %         beta = 0.98; lambdaR = 10;
    %     case 5
    %         beta = 0.98; lambdaR = 5;
    %     otherwise
    %         % 범위를 벗어나는 경우 기본값 설정
    %         beta = 0.90; lambdaR = 1.0; 
    % end

    % IMM model
    switch noiseIdx
        case 1
            beta = 0.999; lambdaR = 0.01;
        case 2
            beta = 0.995; lambdaR = 0.1;
        case 3
            beta = 0.990; lambdaR = 2;
        case 4
            beta = 0.999; lambdaR = 50;
        case 5
            beta = 0.995; lambdaR = 0.5;
        otherwise
            % 범위를 벗어나는 경우 기본값 설정
            beta = 0.90; lambdaR = 1.0; 
    end
end