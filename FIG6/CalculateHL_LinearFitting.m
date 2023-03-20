function [outdata]=CalculateHL_LinearFitting(sourceData,Fequency,ht,HearingNum)
% if (sourceData > 6000)
% 		sourceData = 6000;
% end
for i = 1:HearingNum-1
    if (sourceData >= Fequency(i) && sourceData <= Fequency(i + 1))
        if (Fequency(i) == Fequency(i+1))
            outdata =  ht(i);
        else
            k = (ht(i + 1) - ht(i)) / (Fequency(i+1) - Fequency(i));
            b = ht(i) - k * Fequency(i);
            outdata = k * sourceData + b;
        end
    end
end
end
