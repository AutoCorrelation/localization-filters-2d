classdef PF
    properties
        Property1
    end

    methods
        function obj = PF(inputArg1)
            obj.Property1 = inputArg1^2;
            disp(obj.Property1);
            
        end
    end
end 