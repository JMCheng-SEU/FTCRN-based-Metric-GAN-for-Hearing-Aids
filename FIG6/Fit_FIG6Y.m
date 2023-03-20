function [k,b] = Fit_FIG6Y(audiogram_k, audiogram_ht, ChannelNum)
if (ChannelNum==4)
    ChannelNum_fc = [ 500,1000,2000,4000 ];
end

if (ChannelNum==6)
    ChannelNum_fc = [ 250,500,1000,2000,3000,4000 ];
end

if (ChannelNum==8)
    ChannelNum_fc = [ 250,500,750,1125,1750,2500,4000,6000 ];
end

if (ChannelNum==12)
    ChannelNum_fc = [ 250,375,500,750,1000,1375,1750,2250,3000,3875,4875,6250 ];
end

if (ChannelNum==16)
    ChannelNum_fc = [ 250,375,500,625,750,1000,1125,1375,1750,2125,2625,3125,3875,4625,5500,6625 ];
end


audiogram_ft=[0 audiogram_k];
audiogram_ht=[0 audiogram_ht];
htn=zeros(1,ChannelNum);
for j = 1:ChannelNum
	htn(j) = CalculateHL_LinearFitting(ChannelNum_fc(j), audiogram_ft, audiogram_ht,length(audiogram_ht));
end


k = zeros(ChannelNum,3);                                                   
b = zeros(ChannelNum,3);                                                    
tklin = 40;                                                                 
tkhin = 60;                                                               
for i = 1:ChannelNum
    ht = htn(i);
	k(i,1) = 1;
    if ht < 20
        b(i,1) = 0;
        k(i,2) = 1;
        b(i,2) = 0;
        k(i,3) = 1;
        b(i,3) = 0;
    elseif ht < 40
        ig40 = ht - 20;
        splout40 = 40 + ig40;
        b(i,1) = splout40 - 40;
        tklout = tklin + b(i,1);

        ig65 = 0.6 * (ht - 20);
        splout65 = 65 + ig65;
        %k(i,2) = (splout65 - tklout) / (65 - tklin);
        %b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin);
        k(i,2) = (splout65 - tklout) / (65 - tklin)-1.5;            
        b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin)-30;
        tkhout = k(i,2) * tkhin + b(i,2);

        ig95 = 0;
        splout95 = 95 + ig95;
        k(i,3) = (splout95 - tkhout) / (95 - tkhin);
        b(i,3) = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin);
    elseif ht < 60
        ig40 = ht - 20;

        splout40 = 40 + ig40;
        b(i,1) = splout40 - 40;
        tklout = tklin + b(i,1);

        ig65 = 0.6 * (ht - 20);
        splout65 = 65 + ig65;
        %k(i,2) = (splout65 - tklout) / (65 - tklin);
        %b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin);
        k(i,2) = (splout65 - tklout) / (65 - tklin)-1.5;            
        b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin)-30;
        tkhout = k(i,2) * tkhin + b(i,2);

        ig95 = 0.1*(ht - 40)^1.4;

        splout95 = 95 + ig95;
        k(i,3) = (splout95 - tkhout) / (95 - tkhin);
        b(i,3) = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin);
    else
        ig40 = ht - 20 - 0.5 * (ht - 60);

        splout40 = 40 + ig40;
        b(i,1) = splout40 - 40;
        tklout = tklin + b(i,1);

        ig65 = 0.8 * ht - 23;
        splout65 = 65 + ig65;
        %k(i,2) = (splout65 - tklout) / (65 - tklin);
        %b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin);
        k(i,2) = (splout65 - tklout) / (65 - tklin)-1.5;            
        b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin)-30;
        tkhout = k(i,2) * tkhin + b(i,2);

        ig95 = 0.1*(ht - 40)^1.4;

        splout95 = 95 + ig95;
        k(i,3) = (splout95 - tkhout) / (95 - tkhin);
        b(i,3) = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin);
    end
end