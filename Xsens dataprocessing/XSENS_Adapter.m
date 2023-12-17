% import XSENS data.
%% Start of script
 
addpath('ximu_matlab_library');     % include library

temp = importdata(XSENSPath, '\t', 6);  %tab,第7行开始为数值
XSENSDATA = temp.data;
row=size(XSENSDATA,1);%第一个维度

SamplePeriod = 1/100;

XSENS = struct;
XSENS.time = ((0:row-1)*SamplePeriod)';
% XSENS.Accelerometer = XSENSDATA(:,2:4)/9.8;  % convert the unit to 'g'
% XSENS.Accelerometer = XSENSDATA(:,2:4); 
XSENS_x = XSENSDATA(:,2); 
XSENS_y = XSENSDATA(:,3); 
XSENS_z = XSENSDATA(:,4); 
XSENS_total = sqrt(XSENS_x.*XSENS_x+XSENS_y.*XSENS_y+XSENS_z.*XSENS_z)-9.8;
% XSENS.Gyroscope = XSENSDATA(:,5:7)*180/pi;   % convert the unit to degree, compatible with XIMU and YISHEKUO.
% XSENS.Gyroscope = XSENSDATA(:,5:7);   % convert the unit to degree, compatible with XIMU and YISHEKUO
XSENS_x_g = XSENSDATA(:,5);
XSENS_y_g = XSENSDATA(:,6);
XSENS_z_g = XSENSDATA(:,7);
% XSENS_total_g = sqrt(XSENS_x_g.*XSENS_x_g+XSENS_y_g.*XSENS_y_g+XSENS_z_g.*XSENS_z_g)*180/pi;
XSENS_total_g = -sqrt(XSENS_x_g.*XSENS_x_g+XSENS_y_g.*XSENS_y_g+XSENS_z_g.*XSENS_z_g);
%XSENS.Magnetometer = XSENSDATA(:,8:10);
%XSENS.Ref_quaternion = XSENSDATA(:,11:14);   % 这里保存的与软件界面里的刚好是相反的，
%XSENS.Ref_quaternion = quaternConj(XSENS.Ref_quaternion);   % 需要共轭一下，可能Xsens软件的问题吧。
%XSENS.euler = quatern2euler(XSENS.Ref_quaternion) * (180/pi);

% plot all the data.
if 1
figure('Name', 'XSENS Sensor Data');
axis(1) = subplot(3,1,1);
hold on;
% plot(XSENS.time, XSENS.Gyroscope(:,1), 'b');
% plot(XSENS.time, XSENS.Gyroscope(:,2), 'g');
plot(XSENS.time,XSENS_total_g, 'r');
% legend('X', 'Y', 'Z');
legend('gyr-total');
xlabel('Time (s)');
ylabel('Angular rate (deg/s)');
title('Gyroscope');
hold off;

axis(2) = subplot(3,1,2);
hold on;
% plot(XSENS.time, XSENS.Accelerometer(:,1), 'b');
% plot(XSENS.time, XSENS.Accelerometer(:,2), 'g');
% plot(XSENS.time, XSENS.Accelerometer(:,3), 'r');
plot(XSENS.time, XSENS_total, 'r');
% legend('X', 'Y', 'Z');
legend('acc-total');
xlabel('Time (s)');
ylabel('Acceleration (m/s2)');
title('Accelerometer');
hold off;
% axis(3) = subplot(3,1,3);
% hold on;
% plot(XSENS.time, XSENS.Magnetometer(:,1), 'r');
% plot(XSENS.time, XSENS.Magnetometer(:,2), 'g');
% plot(XSENS.time, XSENS.Magnetometer(:,3), 'b');
% legend('X', 'Y', 'Z');
% xlabel('Time (s)');
% ylabel('Flux (G)');
% title('Magnetometer');
% hold off;
% linkaxes(axis, 'x');


% figure('Name', 'XSENS euler Angles');
% hold on;
% plot(XSENS.time, XSENS.euler(:,1), 'r');
% plot(XSENS.time, XSENS.euler(:,2), 'g');
% plot(XSENS.time, XSENS.euler(:,3), 'b');
% title('XSENS euler Angles');
% xlabel('Time (s)');
% ylabel('Angle (deg)');
% legend('\phi', '\theta', '\psi');
% hold off;
%
end

% plot Magnetometer magnitude and dipangle, from XIMU
% if 0
%     len = size(XSENS.Magnetometer,1);
%     %always using online data
%         
%     for t = 1:len
%         mag(t) = norm(XSENS.Magnetometer(t,:));
%         Qtemp = XSENS.Ref_quaternion(t,:);
%         XSENS.Gravity(t,:) = quaternProd(Qtemp,quaternProd([0,0,0,1],quaternConj(Qtemp)));
%         TempG = XSENS.Gravity(t,2:4); %变成3维的。
%         dipangle(t) = acos(dot(XSENS.Magnetometer(t,:),TempG) / mag(t))*180/pi;
%     end
%     % ylim([0,0.3]);
%     % ylim('auto');
%     figure;
%     hold on;
%     plot(XSENS.time,mag, 'b');
%     % ylim([0,0.3]);
%     fprintf('magetic field\n');
%     xlabel('Time (s)');
%     ylabel('Flux(G)');
%     title('Magnetometer');
%     hold off;
    
    % plot dip angle
%     figure;
%     hold on;
%     plot(XSENS.time,dipangle, 'b');
%     % ylim([0,0.3]);
%     fprintf('dipangle\n');
%     xlabel('Time (s)');
%     ylabel('degree(\circ)');
%     title('dipangle');
%     hold off;    

% end


fprintf('XSENS data import success\n');