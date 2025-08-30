function writematrix(matrix, filename)
% writematrix - Octave compatibility function for MATLAB's writematrix
% This function provides basic CSV writing functionality for Octave
    csvwrite(filename, matrix);
end